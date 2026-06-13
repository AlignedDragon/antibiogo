"""
Error-distribution plot for the xyr radius model on the test split.

Reads the per-image predictions saved by calibration_study.py
(``../calibration/results_test.json``) and renders a two-panel figure:

  LEFT  - signed error (pred - truth), in mm: shows bias / symmetry,
          with median & mean lines and the +/-1mm, +/-3mm reference lines
          (same convention as utils/README.md).
  RIGHT - absolute error |pred - truth|, in mm: with MAE, median, RMSE and
          the 90th/95th percentiles marked.

A stats box lists MAE, median, RMSE, bias, and the fraction beyond 1/2/3 mm.
Pixel->mm uses 1 mm = 3.8 px (from tasks.txt).

Output: ../calibration/error_distribution.png
Run:  python utils/error_distribution.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
RES = os.path.join(WORK, "calibration", "results_test.json")
OUT = os.path.join(WORK, "calibration", "error_distribution.png")
PXMM = 3.8                                            # 1 mm = 3.8 px

d = json.load(open(RES))
gt = np.array([v["gt"] for v in d.values()])
mu = np.array([v["mu"] for v in d.values()])
signed = (mu - gt) / PXMM                             # mm, signed
absmm = np.abs(signed)                                # mm, absolute

mae = absmm.mean()
med = np.median(absmm)
rmse = np.sqrt((absmm ** 2).mean())
bias = signed.mean()
p90, p95 = np.percentile(absmm, [90, 95])
n = len(gt)
frac = {mm: 100 * np.mean(absmm > mm) for mm in (1, 2, 3)}

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.5))

# ---- LEFT: signed error -----------------------------------------------------
clip = 6.0                                            # view window (mm)
n_out = int(np.sum(np.abs(signed) > clip))
view = np.clip(signed, -clip, clip)
ax0.hist(view, bins=61, range=(-clip, clip), density=True,
         color="#7fa8d0", edgecolor="white", linewidth=0.3, alpha=0.9)
xs = np.linspace(-clip, clip, 400)
ax0.plot(xs, gaussian_kde(signed)(xs), color="#1f4e79", lw=2, label="KDE")
ax0.axvline(0, color="black", lw=1)
ax0.axvline(np.median(signed), color="crimson", lw=1.8, ls="-",
            label=f"median {np.median(signed):+.2f} mm")
ax0.axvline(bias, color="darkorange", lw=1.8, ls="--",
            label=f"mean(bias) {bias:+.2f} mm")
for s in (-3, -1, 1, 3):
    ax0.axvline(s, color="grey", lw=1, ls=":")
ax0.set_title(f"Signed error (pred - truth)   [{n_out} pts beyond +/-{clip:.0f}mm clipped]")
ax0.set_xlabel("signed error (mm)   dotted = +/-1mm, +/-3mm")
ax0.set_ylabel("density")
ax0.legend(loc="upper right", fontsize=9)
ax0.grid(alpha=0.25)

# ---- RIGHT: absolute error --------------------------------------------------
clipa = 6.0
n_outa = int(np.sum(absmm > clipa))
ax1.hist(np.clip(absmm, 0, clipa), bins=60, range=(0, clipa), density=True,
         color="#8fbf8f", edgecolor="white", linewidth=0.3, alpha=0.9)
xa = np.linspace(0, clipa, 400)
ax1.plot(xa, gaussian_kde(absmm)(xa), color="#1f5e1f", lw=2, label="KDE")
ax1.axvline(med, color="crimson", lw=1.8, label=f"median {med:.2f} mm")
ax1.axvline(mae, color="darkorange", lw=1.8, ls="--", label=f"MAE {mae:.2f} mm")
ax1.axvline(rmse, color="purple", lw=1.8, ls="-.", label=f"RMSE {rmse:.2f} mm")
ax1.axvline(p95, color="grey", lw=1.5, ls=":", label=f"P95 {p95:.2f} mm")
ax1.set_title(f"Absolute error |pred - truth|   [{n_outa} pts beyond {clipa:.0f}mm clipped]")
ax1.set_xlabel("absolute error (mm)")
ax1.set_ylabel("density")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

stats = (f"n = {n}\n"
         f"MAE    = {mae:.2f} mm ({mae*PXMM:.2f} px)\n"
         f"median = {med:.2f} mm ({med*PXMM:.2f} px)\n"
         f"RMSE   = {rmse:.2f} mm ({rmse*PXMM:.2f} px)\n"
         f"bias   = {bias:+.2f} mm\n"
         f"P90 / P95 = {p90:.2f} / {p95:.2f} mm\n"
         f">1mm {frac[1]:.1f}%  >2mm {frac[2]:.1f}%  >3mm {frac[3]:.1f}%")
ax1.text(0.97, 0.50, stats, transform=ax1.transAxes, ha="right", va="top",
         fontsize=9, family="monospace",
         bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9))

fig.suptitle("xyr radius prediction - error distribution (test split, n=2092)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=130)
print(f"saved {OUT}")
print(stats)
