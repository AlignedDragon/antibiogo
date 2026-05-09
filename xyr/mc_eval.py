"""
Full-val MC-dropout evaluation + uncertainty analysis.

For every validation image: run N MC forward passes, record (gt, mean, std, |diff|).
Then check whether std actually tracks |diff|:
  - Pearson / Spearman correlation between |diff| and std
  - Empirical coverage at +/-1, 2, 3 sigma (Gaussian assumption)
  - Quintile binning by std vs mean error per bin
  - Sparsification curve: drop the K most-uncertain points and see if MAE drops monotonically
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

# Warm-up so XLA compile doesn't pollute timings.
for img, _ in vald_batches.take(1):
    for _ in range(2):
        model.predict(img, verbose=0)

print(f"Running MC eval N={N_MC} over full val set ...", flush=True)
gts, mus, stds = [], [], []
total_predict_time = 0.0
n = 0
t_wall0 = time.perf_counter()

for batch_idx, (img, tg) in enumerate(vald_batches):
    bs = int(img.shape[0])
    pt0 = time.perf_counter()
    samples = np.stack(
        [np.asarray(model.predict(img, verbose=0), dtype=np.float32) for _ in range(N_MC)],
        axis=0,
    )  # (N_MC, B, 1)
    total_predict_time += time.perf_counter() - pt0

    mu = samples.mean(axis=0)[:, 0]
    sd = samples.std(axis=0)[:, 0]
    gt = np.asarray(tg, dtype=np.float32).ravel()

    gts.extend(gt.tolist())
    mus.extend(mu.tolist())
    stds.extend(sd.tolist())
    n += bs
    if (batch_idx + 1) % 10 == 0:
        print(f"  batch {batch_idx+1}: n={n}  cum_predict={total_predict_time:.1f}s", flush=True)

wall = time.perf_counter() - t_wall0
gts = np.asarray(gts)
mus = np.asarray(mus)
stds = np.asarray(stds)
diffs = np.abs(gts - mus)

# Persist per-image results: {idx: [gt, mu, std, diff]}.
results = {str(i): [float(gts[i]), float(mus[i]), float(stds[i]), float(diffs[i])] for i in range(n)}
with open(os.path.join(out_dir, "results.json"), "w") as f:
    json.dump(results, f)

# ----- analysis -----
from scipy.stats import pearsonr, spearmanr

avg_per_estimate_ms = total_predict_time / n * 1000.0

print("\n========== MC dropout uncertainty analysis ==========")
print(f"N_MC = {N_MC}    |    N_val = {n}")
print(f"Wall time: {wall:.1f}s    Pure predict: {total_predict_time:.1f}s")
print(f"Avg per MC estimate: {avg_per_estimate_ms:.1f} ms"
      f"  ({avg_per_estimate_ms / N_MC:.1f} ms per forward pass)")
print()
print(f"GT   range:  [{gts.min():.2f}, {gts.max():.2f}]    mean={gts.mean():.2f}")
print(f"MAE        : {diffs.mean():.3f}")
print(f"RMSE       : {np.sqrt((diffs**2).mean()):.3f}")
print(f"Median |diff|: {np.median(diffs):.3f}")
print(f"Mean std    : {stds.mean():.3f}")
print(f"Median std  : {np.median(stds):.3f}")
print(f"Std range   : [{stds.min():.3f}, {stds.max():.3f}]")
print()

pr, _ = pearsonr(diffs, stds)
sr, _ = spearmanr(diffs, stds)
print(f"Pearson r(|diff|, std)  = {pr:+.3f}")
print(f"Spearman rho(|diff|, std) = {sr:+.3f}")
print("  (positive and >0.3 means std is informative about error)")
print()

# Gaussian coverage check
print("Empirical coverage of |diff| under +/- k sigma (Gaussian expectation in parens):")
for k, expected in zip((1, 2, 3), (68.3, 95.4, 99.7)):
    cov = np.mean(diffs <= k * stds) * 100
    print(f"  |diff| <= {k} sigma : {cov:6.2f} %   (expected ~{expected:.1f}%)")
print()

# Quintile bins by std
qs = np.quantile(stds, np.linspace(0, 1, 6))
bin_idx = np.clip(np.digitize(stds, qs[1:-1]), 0, 4)
print("Bins by std quintile  ->  mean |diff| / mean std / count:")
for i in range(5):
    mask = bin_idx == i
    if mask.any():
        print(f"  Q{i+1}  std in [{qs[i]:.3f}, {qs[i+1]:.3f}]"
              f"   n={mask.sum():4d}"
              f"   mean|diff|={diffs[mask].mean():.3f}"
              f"   mean std={stds[mask].mean():.3f}")
print()

# Sparsification curve: drop K% most-uncertain and recompute MAE.
order_unc = np.argsort(-stds)  # most-uncertain first
order_rand = np.random.default_rng(0).permutation(n)
print("Sparsification (drop top-K%% by std; oracle = drop top-K%% by |diff|):")
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
print("  (drop-unc < drop-rand --> std selects bad cases. drop-oracle is the lower bound.)")
print()

print(f"results saved to: {os.path.join(out_dir, 'results.json')}")
