"""
This file visualizes a model's predictions for val_batches.
It draws base_truth in green and the model's mean_predictions with 
standard_deviation in red. In the bottom of the pictures, there are
the base_truth, mean_prediction, standard_deviation, and the abs_difference
between base_truth and mean_prediction
"""
from dataloader import vald_batches
from utils import root_path, drawer
import os
from xyr_model import model
import tensorflow as tf
from tensorflow.keras.utils import array_to_img

prefix = os.getenv("FOLDER_PREFIX")
viz_path = os.path.join(root_path, f"xyr_viz_{prefix}/")
os.makedirs(viz_path, exist_ok=True)

model_path = os.path.join(*[root_path, "ExperimentModels", prefix, "xyr_best.keras"])
assert os.path.exists(model_path), f"Thre is no saved model in {model_path}"
model.load_weights(model_path)

stream = vald_batches.take(5)

for batch_idx, (image_batch, target_batch) in enumerate(stream):
    batch_predictions = tf.cast(model.predict(image_batch, verbose=0), dtype=tf.float32)

    for img_idx, (sample_image, sample_target, pred) in enumerate(zip(image_batch, target_batch, batch_predictions)):
        output_path = os.path.join(viz_path, f"{batch_idx}_{img_idx}.jpg")
        drawer(array_to_img(sample_image), [sample_target, pred]).convert("RGB").save(output_path)


print("===========================================")
print(f"Visualizations have beend saved to: {viz_path}")
print("===========================================")
