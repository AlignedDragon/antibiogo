"""
Calibration study for the heteroscedastic + MC-dropout `xyr` radius model.

This is the dedicated calibration-study file requested for the (renamed) `utils`
folder. It complements the older eval scripts (`xyr_testing.py`, `xyr_iou.py`):
those measure point accuracy, this one asks whether the model's *predicted
uncertainty* is trustworthy.

Pipeline
--------
Load the trained checkpoint (``../checkpoints/xyr_best.keras``) and run N
MC-dropout forward passes on the Valid and Test splits. Each pass yields
(mu, log_sigma2). Combine per image via the law of total variance:

    mu_pred      = mean_m(mu_m)
    sigma2_alea  = mean_m(exp(log_sigma2_m))     # aleatoric  (data noise)
    sigma2_epi   = var_m(mu_m)                    # epistemic  (model disagreement)
    sigma_total  = sqrt(sigma2_alea + sigma2_epi)

Calibration analysis (Gaussian predictive  y ~ N(mu_pred, sigma_total^2)):
  * reliability curve : nominal central-interval coverage p  vs  observed coverage
  * coverage at +/-{1,2,3} sigma  vs the Gaussian 68.3 / 95.4 / 99.7 %
  * |error| <-> sigma rank correlation (does uncertainty track error?)
  * a single temperature scalar c, FIT ON VALID, applied to TEST
    (so the calibration number is not fit on the set it is reported on)
  * Gaussian NLL before / after the c-scaling

Outputs (under ``../calibration/``):
  * results_test.json   - per-image gt / mu / sigmas / diff
  * metrics.json        - all scalar metrics for valid & test
  * reliability.png     - test reliability curve, raw vs temperature-scaled
  * error_vs_sigma.png  - per-image |error| against predicted sigma_total

Run:  MC_PASSES=30 python utils/calibration_study.py
"""
import os
import sys
import json
import numpy as np

# ---------------------------------------------------------------- paths / config
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)                                   # antibiogo_work/
REPO = os.environ.get("ANTIBIOGO_REPO",
                      "/capstor/scratch/cscs/msayfiddinov/antibiogo")
# Make the repo's xyr package importable so we reuse the exact model definition.
# (xyr_model.py does `from utils import ...`, resolved to xyr/utils.py because we
#  put xyr/ first on sys.path.)
sys.path.insert(0, os.path.join(REPO, "xyr"))

import tensorflow as tf
from xyr_model import model                                    # noqa: E402

CKPT = os.path.join(WORK, "checkpoints", "xyr_best.keras")
OUT = os.path.join(WORK, "calibration")
os.makedirs(OUT, exist_ok=True)

N_MC = int(os.getenv("MC_PASSES", "30"))
BATCH = 32
TEST_DIR = os.path.join(REPO, "data/tf_record_xyr/Test")
VAL_DIR = os.path.join(REPO, "data/tf_record_xyr/Valid")

assert os.path.exists(CKPT), f"missing checkpoint: {CKPT}"
model.load_weights(CKPT)
print(f"loaded checkpoint: {CKPT}")
print(f"MC passes: {N_MC}")


@tf.function(reduce_retracing=True)
def forward(x):
    """One stochastic forward pass. MCDropout stays active at inference (its
    call() forces training=True), so each call is a fresh MC sample. A direct
    call avoids the per-call memory growth of model.predict() when looped tens
    of thousands of times (which OOM-killed the first run)."""
    return model(x, training=False)


# ---------------------------------------------------------------- MC collection
def mc_collect(ds_dir, name):
    """Run N MC-dropout passes over a split; return per-image arrays."""
    ds = tf.data.Dataset.load(ds_dir).batch(BATCH)
    # warm-up so graph trace time is not counted / does not perturb first batch
    for img, _ in ds.take(1):
        for _ in range(2):
            forward(img)

    gts, mus, s2_alea, s2_epi = [], [], [], []
    n = 0
    for bidx, (img, tg) in enumerate(ds):
        samples = np.stack(
            [forward(img).numpy().astype(np.float32) for _ in range(N_MC)],
            axis=0,
        )                                                      # (M, B, 2)
        mu_s = samples[..., 0]                                  # (M, B)
        log_s2 = samples[..., 1]
        mus.extend(mu_s.mean(axis=0).tolist())
        s2_alea.extend(np.exp(log_s2).mean(axis=0).tolist())
        s2_epi.extend(mu_s.var(axis=0).tolist())
        gts.extend(np.asarray(tg, np.float32).ravel().tolist())
        n += int(img.shape[0])
        if (bidx + 1) % 10 == 0:
            print(f"  [{name}] batch {bidx + 1}  n={n}", flush=True)

    gts = np.asarray(gts)
    mus = np.asarray(mus)
    s2_alea = np.asarray(s2_alea)
    s2_epi = np.asarray(s2_epi)
    sigma_total = np.sqrt(s2_alea + s2_epi)
    print(f"  [{name}] done: n={n}")
    return {
        "gt": gts, "mu": mus,
        "sigma_alea": np.sqrt(s2_alea),
        "sigma_epi": np.sqrt(s2_epi),
        "sigma_total": sigma_total,
        "diff": np.abs(gts - mus),
    }


# ---------------------------------------------------------------- calibration math
from scipy.stats import norm, pearsonr, spearmanr            # noqa: E402

# Nominal central coverage levels for the reliability curve.
LEVELS = np.linspace(0.05, 0.99, 20)


def coverage_curve(err, sigma, levels=LEVELS):
    """Observed fraction of points inside the nominal central interval mu +/- z*sigma."""
    obs = []
    for p in levels:
        z = norm.ppf(0.5 + p / 2.0)                            # half-width in sigmas
        obs.append(float(np.mean(err <= z * sigma)))
    return np.asarray(obs)


def calibration_error(err, sigma, levels=LEVELS):
    """Mean abs gap between observed and nominal coverage (lower = better)."""
    return float(np.mean(np.abs(coverage_curve(err, sigma, levels) - levels)))


def gaussian_nll(err, sigma):
    """Mean per-image Gaussian NLL of the true radius under N(mu, sigma^2)."""
    sigma = np.maximum(sigma, 1e-9)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + 0.5 * (err / sigma) ** 2))


def fit_temperature(err, sigma):
    """Variance-calibration scalar c so that scaled z-scores have unit variance:
       mean[(err / (c*sigma))^2] = 1  ->  c = sqrt(mean[(err/sigma)^2])."""
    z2 = (err / np.maximum(sigma, 1e-9)) ** 2
    return float(np.sqrt(np.mean(z2)))


def coverage_at_k(err, sigma):
    return {str(k): float(np.mean(err <= k * sigma) * 100) for k in (1, 2, 3)}


# ---------------------------------------------------------------- run both splits
val = mc_collect(VAL_DIR, "valid")
test = mc_collect(TEST_DIR, "test")

# Temperature fit on VALID, applied to TEST (no fitting on the reported set).
c = fit_temperature(val["diff"], val["sigma_total"])
print(f"\ntemperature scalar c (fit on valid) = {c:.4f}")


def split_metrics(d, c_scale):
    err, sig = d["diff"], d["sigma_total"]
    pr, _ = pearsonr(err, sig)
    sr, _ = spearmanr(err, sig)
    return {
        "n": int(err.size),
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "gt_mean": float(d["gt"].mean()),
        "sigma_total_mean": float(sig.mean()),
        "sigma_alea_mean": float(d["sigma_alea"].mean()),
        "sigma_epi_mean": float(d["sigma_epi"].mean()),
        "var_share_aleatoric_pct": float((d["sigma_alea"] ** 2).mean()
                                         / (sig ** 2).mean() * 100),
        "corr_err_sigma_pearson": float(pr),
        "corr_err_sigma_spearman": float(sr),
        "coverage_pct_raw": coverage_at_k(err, sig),
        "coverage_pct_scaled": coverage_at_k(err, c_scale * sig),
        "calibration_error_raw": calibration_error(err, sig),
        "calibration_error_scaled": calibration_error(err, c_scale * sig),
        "nll_raw": gaussian_nll(err, sig),
        "nll_scaled": gaussian_nll(err, c_scale * sig),
    }


metrics = {
    "n_mc": N_MC,
    "temperature_c_fit_on_valid": c,
    "valid": split_metrics(val, c),
    "test": split_metrics(test, c),
}

# ---------------------------------------------------------------- persist json
with open(os.path.join(OUT, "results_test.json"), "w") as f:
    json.dump({str(i): {k: float(test[k][i]) for k in
                        ("gt", "mu", "sigma_alea", "sigma_epi", "sigma_total", "diff")}
               for i in range(test["gt"].size)}, f)
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------- plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

# 1) reliability curve on TEST: raw vs temperature-scaled
obs_raw = coverage_curve(test["diff"], test["sigma_total"])
obs_scaled = coverage_curve(test["diff"], c * test["sigma_total"])
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
plt.plot(LEVELS, obs_raw, "o-", color="crimson",
         label=f"raw  (cal.err={metrics['test']['calibration_error_raw']:.3f})")
plt.plot(LEVELS, obs_scaled, "s-", color="steelblue",
         label=f"c={c:.2f} scaled  (cal.err={metrics['test']['calibration_error_scaled']:.3f})")
plt.xlabel("nominal central coverage")
plt.ylabel("observed coverage (test)")
plt.title("Reliability curve - xyr radius (Gaussian + MC-dropout)")
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "reliability.png"), dpi=130)
plt.close()

# 2) |error| vs predicted sigma_total (sharpness / usefulness) on TEST
plt.figure(figsize=(6, 6))
plt.scatter(test["sigma_total"], test["diff"], s=8, alpha=0.35, color="seagreen")
lim = max(test["sigma_total"].max(), test["diff"].max())
plt.plot([0, lim], [0, lim], "k--", lw=1, label="|error| = sigma")
plt.xlabel("predicted sigma_total  (px)")
plt.ylabel("|error| = |gt - mu|  (px)")
plt.title(f"Error vs uncertainty (test)  Spearman={metrics['test']['corr_err_sigma_spearman']:+.3f}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "error_vs_sigma.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------- console report
def report(tag, m):
    print(f"\n===== {tag}  (n={m['n']}) =====")
    print(f"  MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  "
          f"sigma_total(mean)={m['sigma_total_mean']:.3f}")
    print(f"  var share: aleatoric={m['var_share_aleatoric_pct']:.1f}%  "
          f"epistemic={100 - m['var_share_aleatoric_pct']:.1f}%")
    print(f"  corr(|err|,sigma): Pearson={m['corr_err_sigma_pearson']:+.3f}  "
          f"Spearman={m['corr_err_sigma_spearman']:+.3f}")
    print(f"  coverage raw    1/2/3 sigma: "
          f"{m['coverage_pct_raw']['1']:.1f} / {m['coverage_pct_raw']['2']:.1f} / "
          f"{m['coverage_pct_raw']['3']:.1f}  (target 68.3/95.4/99.7)")
    print(f"  coverage scaled 1/2/3 sigma: "
          f"{m['coverage_pct_scaled']['1']:.1f} / {m['coverage_pct_scaled']['2']:.1f} / "
          f"{m['coverage_pct_scaled']['3']:.1f}")
    print(f"  calibration error raw={m['calibration_error_raw']:.3f}  "
          f"scaled={m['calibration_error_scaled']:.3f}")
    print(f"  NLL raw={m['nll_raw']:.3f}  scaled={m['nll_scaled']:.3f}")


print("\n========== CALIBRATION STUDY ==========")
print(f"temperature scalar c (fit on valid) = {c:.4f}")
report("VALID", metrics["valid"])
report("TEST", metrics["test"])
print(f"\nsaved: {OUT}/(metrics.json, results_test.json, reliability.png, error_vs_sigma.png)")
