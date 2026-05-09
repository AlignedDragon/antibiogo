"""
Visualizes the MC-dropout model's predictions on val_batches.
Each image shows the base truth (green), the MC-mean prediction (red),
and a text overlay with mean +/- std and the abs difference.
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

# Number of MC-dropout forward passes per image. 30 is a common balance:
# Var(mean) ~ 1/N and Var(std) ~ 1/(2(N-1)) — diminishing returns past ~30.
# Override via env: MC_PASSES=...
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
    )  # (N_MC, B, 1)
    total_mc_time += time.perf_counter() - t0
    total_estimates += bs

    mean = samples.mean(axis=0)  # (B, 1)
    std = samples.std(axis=0)    # (B, 1)

    for img_idx in range(bs):
        gt = float(target_batch[img_idx])
        mu = float(mean[img_idx, 0])
        sigma = float(std[img_idx, 0])
        diff = abs(gt - mu)

        text = (f"Base: {gt:.2f}\n"
                f"Pred: {mu:.2f} +/- {sigma:.2f}\n"
                f"Diff: {diff:.2f}")

        img_obj = drawer(array_to_img(image_batch[img_idx]), [gt, mu]).convert("RGB")
        draw = ImageDraw.Draw(img_obj)

        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x_pos = img_obj.width - text_w - 10
        y_pos = img_obj.height - text_h - 10
        draw.text((x_pos, y_pos), text, fill="white", stroke_fill="black", stroke_width=1)

        img_obj.save(os.path.join(viz_path, f"{batch_idx}_{img_idx}.jpg"))

avg_per_estimate_ms = (total_mc_time / total_estimates) * 1000.0 if total_estimates else 0.0
print(f"\nMC dropout: N={N_MC} forward passes per estimate")
print(f"Total estimates: {total_estimates}")
print(f"Total MC compute time: {total_mc_time:.2f} s")
print(f"Average time per MC estimate: {avg_per_estimate_ms:.1f} ms "
      f"({avg_per_estimate_ms / N_MC:.1f} ms per forward pass)")
print(f"Visualizations have been saved to: {viz_path}")
