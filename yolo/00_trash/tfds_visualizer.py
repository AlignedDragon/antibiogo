import cv2
import numpy as np
import os


output_dir = '/home/muhammad-ali/working/detector/base_data/processed'

def draw_boxes(image, boxes):
    for box in boxes:
        if np.array_equal(box, [-1, -1, -1, -1]):  # Skip invalid boxes
            continue
        x_min, y_min, x_max, y_max = int(box[0] -box[3]/2), int(box[1] -box[2]/2), int(box[0] +box[3]/2), int(box[1] +box[2]/2)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image
i = 0
# Iterate through batches
for batch_idx, batch in enumerate(orig_train_batches):
    images = np.array(batch[0])  # Shape: (32, 256, 256, 3)
    boxes = np.array(batch[1]['boxes'])  # Shape: (32, 16, 4)

    for img_idx, (image, img_boxes) in enumerate(zip(images, boxes)):
        # img_height, img_width = image.shape[:2]
        # img = (image * 255).astype(np.uint8)  # Denormalize if necessary
        # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Convert and draw bounding boxes
        img_with_boxes = draw_boxes(image, img_boxes)

        # Save the image
        output_path = os.path.join(output_dir, f"batch_{batch_idx}_img_{img_idx}.jpg")
        cv2.imwrite(output_path, img_with_boxes)
        i+=1
        if i%100==0:
            print(f'{i} images done')

print(f"Visualizations saved to {output_dir}")