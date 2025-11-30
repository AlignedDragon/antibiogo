"""
This file visualizes a model's predictions for val_batches.
It draws base_truth in green and the model's mean_predictions with 
standard_deviation in red. In the bottom of the pictures, there are
the base_truth, mean_prediction, standard_deviation, and the abs_difference
between base_truth and mean_prediction
"""
from dataloader import vald_batches
from utils import root_path
import os
from detector import yolo as model
import tensorflow as tf
from PIL import ImageDraw, Image
from tqdm import tqdm
import numpy as np


prefix = os.getenv("FOLDER_PREFIX")
viz_path = os.path.join(root_path, f"yolo_viz_{prefix}/")
os.makedirs(viz_path, exist_ok=True)

model_path = os.path.join(*[root_path, "ExperimentModels", prefix, "yolo_best.keras"])
assert os.path.exists(model_path), f"Thre is no saved model in {model_path}"
model.load_weights(model_path)

def draw_boxes(image, boxes, color="green"):
    # Convert numpy array to PIL Image if it's not already
    if isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Denormalize if the image is in float format
            image = (image * 255).astype(np.uint8)
        
        # Ensure the image is in RGB mode
        image = Image.fromarray(image)
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    for box in boxes:
        x_center, y_center, width, height = box

        if np.array_equal(box, [-1, -1, -1, -1]) or width > 100 or height > 100:  # Skip invalid boxes
            continue
        
        x_min = int(x_center - width/2)
        y_min = int(y_center - height/2)
        x_max = int(x_center + width/2)
        y_max = int(y_center + height/2)
        
        # Draw rectangle
        if color=="red":
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(255, 0, 0), width=2)
        else:
            draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)
    
    return image

for batch_idx, batch in tqdm(enumerate(vald_batches), desc="Drawing visualizations"):
    image_batch, box_batch = np.array(batch[0]), np.array(batch[1]['boxes'])

    batch_predictions = np.array(model.predict(image_batch, verbose=0)["boxes"])
    
    for img_idx, (sample_image, sample_boxes, pred) in enumerate(zip(image_batch, box_batch, batch_predictions)):
        img_with_boxes = draw_boxes(sample_image, sample_boxes)
        img_with_boxes = draw_boxes(img_with_boxes, pred, color="red")

        filename=f"{batch_idx}_{img_idx}.jpg"
        output_path = os.path.join(viz_path, filename)
        img_with_boxes.save(output_path)
    if batch_idx > 2:
        break

print(f"Visualizations have beend saved to: {viz_path}")
