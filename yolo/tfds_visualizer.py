from PIL import Image, ImageDraw
import numpy as np
import os
from dataloader import orig_train_batches

output_dir = '/mloscratch/sayfiddi/yolo/base_data/processed'

def draw_boxes(image, boxes):
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
        if np.array_equal(box, [-1, -1, -1, -1]):  # Skip invalid boxes
            continue
        
        # Convert center-based format to corner-based format
        x_center, y_center, width, height = box
        x_min = int(x_center - width/2)
        y_min = int(y_center - height/2)
        x_max = int(x_center + width/2)
        y_max = int(y_center + height/2)
        
        # Draw rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=(0, 255, 0), width=2)
    
    return image

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

i = 0
# Iterate through batches
for batch_idx, batch in enumerate(orig_train_batches):
    images = np.array(batch[0])  # Shape: (32, 256, 256, 3)
    boxes = np.array(batch[1]['boxes'])  # Shape: (32, 16, 4)

    for img_idx, (image, img_boxes) in enumerate(zip(images, boxes)):
        # Draw bounding boxes
        img_with_boxes = draw_boxes(image, img_boxes)

        # Save the image
        output_path = os.path.join(output_dir, f"batch_{batch_idx}_img_{img_idx}.jpg")
        img_with_boxes.save(output_path)
        
        i += 1
        if i % 100 == 0:
            print(f'{i} images done')

print(f"Visualizations saved to {output_dir}")