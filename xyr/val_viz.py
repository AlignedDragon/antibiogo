"""
Heteroscedastic + MC-dropout visualization on val_batches.
Each image: green = GT, red = MC-mean prediction.
Overlay: Pred = mu +/- sigma_total  with aleatoric vs epistemic split.
"""
from dataloader import vald_batches
from utils import root_path, drawer
import os
import time
from xyr_model import model
from tensorflow.keras.utils import array_to_img
from PIL import ImageDraw
from tqdm import tqdm
import numpy as np

N_MC = int(os.getenv("MC_PASSES", "30"))

prefix = os.getenv("FOLDER_PREFIX")
viz_path = os.path.join(root_path, f"xyr_viz_{prefix}/")
os.makedirs(viz_path, exist_ok=True)

model_path = os.path.join(root_path, "ExperimentModels", prefix, "xyr_best.keras")
assert os.path.exists(model_path), f"There is no saved model in {model_path}"
model.load_weights(model_path)

total_mc_time = 0.0
total_estimates = 0

for batch_idx, (image_batch, target_batch) in tqdm(enumerate(vald_batches), desc="MC visualizations"):
    bs = int(image_batch.shape[0])

    t0 = time.perf_counter()
    samples = np.stack(
        [np.asarray(model.predict(image_batch, verbose=0), dtype=np.float32) for _ in range(N_MC)],
        axis=0,
    )  # (N_MC, B, 2)
    total_mc_time += time.perf_counter() - t0
    total_estimates += bs

    mu_samples = samples[..., 0]
    sigma2_alea_samples = np.exp(samples[..., 1])

    mu = mu_samples.mean(axis=0)                             # (B,)
    sigma_alea = np.sqrt(sigma2_alea_samples.mean(axis=0))   # (B,)
    sigma_epi = mu_samples.std(axis=0)                       # (B,)
    sigma_total = np.sqrt(sigma_alea**2 + sigma_epi**2)

    for img_idx in range(bs):
        gt = float(target_batch[img_idx])
        m = float(mu[img_idx])
        st = float(sigma_total[img_idx])
        sa = float(sigma_alea[img_idx])
        se = float(sigma_epi[img_idx])
        diff = abs(gt - m)

        text = (f"Base: {gt:.2f}\n"
                f"Pred: {m:.2f} +/- {st:.2f}\n"
                f"  alea: {sa:.2f}  epi: {se:.2f}\n"
                f"Diff: {diff:.2f}")

        img_obj = drawer(array_to_img(image_batch[img_idx]), [gt, m]).convert("RGB")
        draw = ImageDraw.Draw(img_obj)

        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x_pos = img_obj.width - text_w - 10
        y_pos = img_obj.height - text_h - 10
        draw.text((x_pos, y_pos), text, fill="white", stroke_fill="black", stroke_width=1)

        img_obj.save(os.path.join(viz_path, f"{batch_idx}_{img_idx}.jpg"))

avg_per_estimate_ms = (total_mc_time / total_estimates) * 1000.0 if total_estimates else 0.0
print(f"\nMC dropout (heteroscedastic): N={N_MC} forward passes per estimate")
print(f"Total estimates: {total_estimates}")
print(f"Total MC compute time: {total_mc_time:.2f} s")
print(f"Average time per MC estimate: {avg_per_estimate_ms:.1f} ms "
      f"({avg_per_estimate_ms / N_MC:.1f} ms per forward pass)")
print(f"Visualizations saved to: {viz_path}")
