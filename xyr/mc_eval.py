"""
Heteroscedastic + MC-dropout uncertainty evaluation on the full val set.

Each MC pass yields (mu_i, log_sigma2_i). We combine via the law of total variance:
    mu_pred         = mean_i(mu_i)
    sigma2_alea     = mean_i(exp(log_sigma2_i))           # data noise
    sigma2_epi      = var_i(mu_i)                          # model disagreement
    sigma2_total    = sigma2_alea + sigma2_epi
    sigma_total     = sqrt(sigma2_total)

Then we check:
  - Pearson/Spearman correlation between |gt - mu_pred| and sigma_total
  - Empirical coverage at +/-{1,2,3} sigma_total vs Gaussian expectation
  - Quintile binning by sigma_total -> mean |diff|
  - Sparsification curve (random vs uncertainty vs oracle)
  - Aleatoric/epistemic split: how much of the total variance comes from each
"""
import os
import time
import json
import numpy as np

from dataloader import vald_batches
from xyr_model import model
from utils import root_path

N_MC = int(os.getenv("MC_PASSES", "30"))
prefix = os.getenv("FOLDER_PREFIX")
assert prefix, "FOLDER_PREFIX env var is missing"

out_dir = os.path.join(root_path, f"mc_eval_{prefix}")
os.makedirs(out_dir, exist_ok=True)

ckpt = os.path.join(root_path, "ExperimentModels", prefix, "xyr_best.keras")
assert os.path.exists(ckpt), f"missing checkpoint: {ckpt}"
model.load_weights(ckpt)

# Warm-up so XLA compile doesn't pollute timing.
for img, _ in vald_batches.take(1):
    for _ in range(2):
        model.predict(img, verbose=0)

print(f"Running heteroscedastic MC eval N={N_MC} over full val set ...", flush=True)

gts = []
mus_all = []           # (n,)  predictive mean
sigma2_alea_all = []   # (n,)  mean of exp(log_sigma2) across passes
sigma2_epi_all = []    # (n,)  variance of mu across passes

total_predict_time = 0.0
n = 0
t_wall0 = time.perf_counter()

for batch_idx, (img, tg) in enumerate(vald_batches):
    bs = int(img.shape[0])
    pt0 = time.perf_counter()

    # Stack M passes -> (M, B, 2)
    samples = np.stack(
        [np.asarray(model.predict(img, verbose=0), dtype=np.float32) for _ in range(N_MC)],
        axis=0,
    )
    total_predict_time += time.perf_counter() - pt0

    mu_samples = samples[..., 0]               # (M, B)
    log_sigma2_samples = samples[..., 1]       # (M, B)
    sigma2_alea_samples = np.exp(log_sigma2_samples)

    mu_pred = mu_samples.mean(axis=0)          # (B,)
    sigma2_alea = sigma2_alea_samples.mean(axis=0)
    sigma2_epi = mu_samples.var(axis=0)
    gt = np.asarray(tg, dtype=np.float32).ravel()

    gts.extend(gt.tolist())
    mus_all.extend(mu_pred.tolist())
    sigma2_alea_all.extend(sigma2_alea.tolist())
    sigma2_epi_all.extend(sigma2_epi.tolist())
    n += bs
    if (batch_idx + 1) % 10 == 0:
        print(f"  batch {batch_idx+1}: n={n}  cum_predict={total_predict_time:.1f}s", flush=True)

wall = time.perf_counter() - t_wall0

gts = np.asarray(gts)
mus = np.asarray(mus_all)
sigma2_alea = np.asarray(sigma2_alea_all)
sigma2_epi = np.asarray(sigma2_epi_all)
sigma2_total = sigma2_alea + sigma2_epi
sigma_alea = np.sqrt(sigma2_alea)
sigma_epi = np.sqrt(sigma2_epi)
sigma_total = np.sqrt(sigma2_total)
diffs = np.abs(gts - mus)

# Persist per-image results.
records = {
    str(i): {
        "gt": float(gts[i]),
        "mu": float(mus[i]),
        "sigma_alea": float(sigma_alea[i]),
        "sigma_epi": float(sigma_epi[i]),
        "sigma_total": float(sigma_total[i]),
        "diff": float(diffs[i]),
    }
    for i in range(n)
}
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(records, f)

# ----- analysis -----
from scipy.stats import pearsonr, spearmanr

avg_per_estimate_ms = total_predict_time / n * 1000.0

print("\n========== Heteroscedastic + MC dropout uncertainty analysis ==========")
print(f"N_MC = {N_MC}    |    N_val = {n}")
print(f"Wall time: {wall:.1f}s    Pure predict: {total_predict_time:.1f}s")
print(f"Avg per MC estimate: {avg_per_estimate_ms:.1f} ms"
      f"  ({avg_per_estimate_ms / N_MC:.1f} ms per forward pass)")
print()
print(f"GT   range: [{gts.min():.2f}, {gts.max():.2f}]   mean={gts.mean():.2f}")
print(f"MAE        : {diffs.mean():.3f}")
print(f"RMSE       : {np.sqrt((diffs**2).mean()):.3f}")
print(f"Median |diff|: {np.median(diffs):.3f}")
print()
print("Uncertainty components:")
print(f"  sigma_alea  : mean={sigma_alea.mean():.3f}  median={np.median(sigma_alea):.3f}  range=[{sigma_alea.min():.3f}, {sigma_alea.max():.3f}]")
print(f"  sigma_epi   : mean={sigma_epi.mean():.3f}   median={np.median(sigma_epi):.3f}   range=[{sigma_epi.min():.3f}, {sigma_epi.max():.3f}]")
print(f"  sigma_total : mean={sigma_total.mean():.3f} median={np.median(sigma_total):.3f} range=[{sigma_total.min():.3f}, {sigma_total.max():.3f}]")
frac_alea = sigma2_alea.mean() / sigma2_total.mean()
print(f"  variance share: aleatoric={frac_alea*100:5.1f}%   epistemic={(1-frac_alea)*100:5.1f}%")
print()

for label, sig in (("sigma_total", sigma_total), ("sigma_alea", sigma_alea), ("sigma_epi", sigma_epi)):
    pr, _ = pearsonr(diffs, sig)
    sr, _ = spearmanr(diffs, sig)
    print(f"corr(|diff|, {label:12s})  Pearson={pr:+.3f}   Spearman={sr:+.3f}")
print()

print("Empirical coverage of |diff| under +/- k sigma_total (Gaussian expectation in parens):")
for k, expected in zip((1, 2, 3), (68.3, 95.4, 99.7)):
    cov = np.mean(diffs <= k * sigma_total) * 100
    print(f"  |diff| <= {k} sigma : {cov:6.2f} %   (expected ~{expected:.1f}%)")
print()

# Optional one-scalar calibration on the same set (would normally be on a held-out split).
c = float(np.quantile(diffs / np.maximum(sigma_total, 1e-9), 0.683))
print(f"Single-scalar temperature calibration on this set: c = {c:.3f}")
for k, expected in zip((1, 2, 3), (68.3, 95.4, 99.7)):
    cov = np.mean(diffs <= k * c * sigma_total) * 100
    print(f"  |diff| <= {k} (c*sigma) : {cov:6.2f} %   (expected ~{expected:.1f}%)")
print()

qs = np.quantile(sigma_total, np.linspace(0, 1, 6))
bin_idx = np.clip(np.digitize(sigma_total, qs[1:-1]), 0, 4)
print("Bins by sigma_total quintile  ->  mean |diff| / mean sigma / count:")
for i in range(5):
    mask = bin_idx == i
    if mask.any():
        print(f"  Q{i+1}  sigma in [{qs[i]:.3f}, {qs[i+1]:.3f}]"
              f"   n={mask.sum():4d}"
              f"   mean|diff|={diffs[mask].mean():.3f}"
              f"   mean sigma={sigma_total[mask].mean():.3f}")
print()

order_unc = np.argsort(-sigma_total)
order_rand = np.random.default_rng(0).permutation(n)
print("Sparsification (drop top-K%% by sigma_total; oracle drops by |diff|):")
print(f"{'drop %':>7}  {'MAE drop-rand':>13}  {'MAE drop-unc':>13}  {'MAE drop-oracle':>15}")
for pct in (0, 5, 10, 20, 30, 50):
    k = int(n * pct / 100)
    keep_unc = np.sort(order_unc[k:])
    keep_rand = np.sort(order_rand[k:])
    keep_oracle = np.sort(np.argsort(-diffs)[k:])
    mae_unc = diffs[keep_unc].mean()
    mae_rand = diffs[keep_rand].mean() if len(keep_rand) else float("nan")
    mae_oracle = diffs[keep_oracle].mean() if len(keep_oracle) else float("nan")
    print(f"  {pct:>4d}    {mae_rand:>13.3f}  {mae_unc:>13.3f}  {mae_oracle:>15.3f}")
print()
print(f"results saved to: {os.path.join(out_dir, 'results.json')}")
