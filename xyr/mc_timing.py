"""Quick MC-dropout timing: N forward passes per image, no IO."""
from dataloader import vald_batches
from utils import root_path
import os, time
from xyr_model import model
import numpy as np

N_MC = int(os.getenv("MC_PASSES", "30"))
N_BATCHES = int(os.getenv("N_BATCHES", "10"))  # cap for quick measurement

prefix = os.getenv("FOLDER_PREFIX")
model_path = os.path.join(root_path, "ExperimentModels", prefix, "xyr_best.keras")
model.load_weights(model_path)

# Warm-up: trigger XLA / autograph compilation so the first batch isn't measured.
for image_batch, _ in vald_batches.take(1):
    for _ in range(2):
        model.predict(image_batch, verbose=0)

total_mc_time = 0.0
total_estimates = 0
batches_done = 0

for image_batch, _ in vald_batches.take(N_BATCHES):
    bs = int(image_batch.shape[0])
    t0 = time.perf_counter()
    samples = np.stack(
        [np.asarray(model.predict(image_batch, verbose=0), dtype=np.float32) for _ in range(N_MC)],
        axis=0,
    )
    total_mc_time += time.perf_counter() - t0
    total_estimates += bs
    batches_done += 1
    _ = samples.mean(0), samples.std(0)

avg_ms = (total_mc_time / total_estimates) * 1000.0
print(f"\nMC dropout: N={N_MC} forward passes per estimate")
print(f"Batches measured: {batches_done}  |  Estimates: {total_estimates}")
print(f"Total MC compute time: {total_mc_time:.2f} s")
print(f"Average time per MC estimate: {avg_ms:.1f} ms "
      f"({avg_ms / N_MC:.1f} ms per forward pass)")
