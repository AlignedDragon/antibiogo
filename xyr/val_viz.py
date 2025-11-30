"""
This file visualizes a model's predictions for val_batches.
It draws base_truth in green and the model's mean_predictions with 
standard_deviation in red. In the bottom of the pictures, there are
the base_truth, mean_prediction, standard_deviation, and the abs_difference
between base_truth and mean_prediction
"""
from dataloader import vald_batches
from utils import root_path, drawer, unnormalize
import os
from xyr_model import model
import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from PIL import ImageDraw
from tqdm import tqdm

prefix = os.getenv("FOLDER_PREFIX")
viz_path = os.path.join(root_path, f"xyr_viz_{prefix}/")
os.makedirs(viz_path, exist_ok=True)

model_path = os.path.join(*[root_path, "ExperimentModels", prefix, "xyr_best.keras"])
assert os.path.exists(model_path), f"Thre is no saved model in {model_path}"
model.load_weights(model_path)

# stream = vald_batches.take(5)

for batch_idx, (image_batch, target_batch) in tqdm(enumerate(vald_batches), desc="Drawing visualizations"):
    batch_predictions = tf.cast(model.predict(image_batch, verbose=0), dtype=tf.float32)
    for img_idx, (sample_image, sample_target, pred) in enumerate(zip(image_batch, target_batch, batch_predictions)):
        # 1. Extract scalars for display
        gt, mu, std = float(unnormalize(sample_target)), float(unnormalize(pred[0])), float(unnormalize(pred[1], sigma=True))
        diff = abs(gt - mu)

        # 2. Format the text
        text = (f"Base: {gt:.2f}\n"
                f"Mean: {mu:.2f}\n"
                f"Std:  {std:.2f}\n"
                f"Diff: {diff:.2f}")
        
        output_path = os.path.join(viz_path, f"{batch_idx}_{img_idx}.jpg")
        img_obj = drawer(array_to_img(sample_image), [sample_target, pred]).convert("RGB")
        draw = ImageDraw.Draw(img_obj)

        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x_pos = img_obj.width - text_w - 10
        y_pos = img_obj.height - text_h - 10
        
        # Draw text (White with generic positioning)
        draw.text((x_pos, y_pos), text, fill="white", stroke_fill="black", stroke_width=1)

        filename=f"{batch_idx}_{img_idx}.jpg"
        output_path = os.path.join(viz_path, filename)
        img_obj.save(output_path)

print(f"Visualizations have beend saved to: {viz_path}")
