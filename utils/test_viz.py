"""
Test-split prediction visualization for the `xyr` radius model.

Each output is a TWO-PANEL figure for one test image:

  * LEFT  - the original image, untouched (no overlay).
  * RIGHT - the same image with the prediction overlay:
        GROUND TRUTH - green  circle  (true inhibition-zone radius, centered)
        PREDICTION   - red    circle  (mu = MC-mean predicted radius)
        DEVIATION    - orange dashed +/-1 sigma_total band around the prediction
    and, in a dedicated text bar BELOW the right panel (so the text can never
    spill off the image), the numeric ground truth, prediction +/- sigma, the
    aleatoric / epistemic split, and the signed deviation.

Uncertainty is from N MC-dropout passes combined with the law of total variance,
identical to calibration_study.py.

Outputs JPGs under ``../test_viz/``.
Run:  MC_PASSES=20 python utils/test_viz.py
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)
REPO = os.environ.get("ANTIBIOGO_REPO",
                      "/capstor/scratch/cscs/msayfiddinov/antibiogo")
sys.path.insert(0, os.path.join(REPO, "xyr"))

import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from PIL import Image, ImageDraw, ImageFont
from xyr_model import model
from utils import drawer                                       # xyr/utils.py drawer

CKPT = os.path.join(WORK, "checkpoints", "xyr_best.keras")
OUT = os.path.join(WORK, "test_viz")
os.makedirs(OUT, exist_ok=True)
N_MC = int(os.getenv("MC_PASSES", "20"))
BATCH = 32
TEST_DIR = os.path.join(REPO, "data/tf_record_xyr/Test")

# ---- layout constants -------------------------------------------------------
PAD = 8                 # outer / inter-panel padding
GAP = 10                # gap between the two panels
TEXTBAR = 92            # height of the text strip under each panel
LINE_H = 17             # text line height
BG = (18, 18, 18)       # dark canvas background
DIV = (90, 90, 90)      # divider colour

try:
    FONT = ImageFont.load_default(size=14)          # Pillow >= 10 supports size
    FONT_B = ImageFont.load_default(size=14)
except TypeError:
    FONT = ImageFont.load_default()
    FONT_B = FONT

assert os.path.exists(CKPT), f"missing checkpoint: {CKPT}"
model.load_weights(CKPT)
print(f"loaded {CKPT}  | MC passes={N_MC}")


@tf.function(reduce_retracing=True)
def forward(x):
    """One stochastic MC-dropout forward pass (direct call avoids the
    model.predict() memory growth seen when looped many thousands of times)."""
    return model(x, training=False)


def dashed_circle(draw, cx, cy, r, color, n=48, width=2):
    """Draw a dashed circle of radius r centered at (cx, cy)."""
    if r <= 0:
        return
    for k in range(0, n, 2):
        a0 = 2 * np.pi * k / n
        a1 = 2 * np.pi * (k + 1) / n
        draw.line([(cx + r * np.cos(a0), cy + r * np.sin(a0)),
                   (cx + r * np.cos(a1), cy + r * np.sin(a1))],
                  fill=color, width=width)


def panel(img_rgb, title_lines):
    """Wrap a square image into a panel with a text bar underneath.
    The text lives in its own strip below the image, so it never overlaps or
    spills past the picture."""
    w, h = img_rgb.size
    p = Image.new("RGB", (w, h + TEXTBAR), BG)
    p.paste(img_rgb, (0, 0))
    d = ImageDraw.Draw(p)
    y = h + 5
    for ln, color in title_lines:
        d.text((6, y), ln, fill=color, font=FONT)
        y += LINE_H
    return p


def compose(left_panel, right_panel):
    """Place the two same-height panels side by side with a divider."""
    h = left_panel.height
    lw, rw = left_panel.width, right_panel.width
    canvas = Image.new("RGB", (PAD + lw + GAP + rw + PAD, PAD + h + PAD), BG)
    canvas.paste(left_panel, (PAD, PAD))
    canvas.paste(right_panel, (PAD + lw + GAP, PAD))
    # vertical divider line in the gap
    x_div = PAD + lw + GAP // 2
    ImageDraw.Draw(canvas).line([(x_div, PAD), (x_div, PAD + h)], fill=DIV, width=1)
    return canvas


test = tf.data.Dataset.load(TEST_DIR).batch(BATCH)
# warm-up
for img, _ in test.take(1):
    for _ in range(2):
        forward(img)

n = 0
for bidx, (image_batch, target_batch) in enumerate(test):
    samples = np.stack(
        [forward(image_batch).numpy().astype(np.float32) for _ in range(N_MC)],
        axis=0,
    )                                                          # (M, B, 2)
    mu = samples[..., 0].mean(axis=0)
    sigma_alea = np.sqrt(np.exp(samples[..., 1]).mean(axis=0))
    sigma_epi = samples[..., 0].std(axis=0)
    sigma_total = np.sqrt(sigma_alea ** 2 + sigma_epi ** 2)

    bs = int(image_batch.shape[0])
    for i in range(bs):
        gt = float(target_batch[i])
        m = float(mu[i])
        st = float(sigma_total[i])
        sa = float(sigma_alea[i])
        se = float(sigma_epi[i])
        dev = m - gt                                           # signed deviation

        original = array_to_img(image_batch[i]).convert("RGB")

        # RIGHT panel image: GT (green) + prediction (red) circles + dashed band.
        overlay = drawer(original.copy(), [gt, m]).convert("RGB")
        od = ImageDraw.Draw(overlay)
        cx, cy = overlay.width // 2, overlay.height // 2
        dashed_circle(od, cx, cy, m + st, (255, 165, 0))
        dashed_circle(od, cx, cy, max(m - st, 0.0), (255, 165, 0))

        left = panel(original, [("Original", (200, 200, 200))])
        right = panel(overlay, [
            (f"GT {gt:.2f}", (0, 230, 0)),
            (f"Pred {m:.2f} +/- {st:.2f}", (255, 80, 80)),
            (f"  alea {sa:.2f}   epi {se:.2f}", (255, 165, 0)),
            (f"Dev {dev:+.2f}  (|{abs(dev):.2f}|)", (235, 235, 235)),
        ])
        compose(left, right).save(os.path.join(OUT, f"{bidx}_{i}.jpg"))
        n += 1
    if (bidx + 1) % 10 == 0:
        print(f"  batch {bidx + 1}  saved n={n}", flush=True)

print("\nlayout: left = original image, right = overlay "
      "(green=ground truth, red=prediction mu, orange dashed = +/-1 sigma_total)")
print(f"saved {n} two-panel test visualizations to: {OUT}")
