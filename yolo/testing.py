import numpy as np
import os
from dataloader import orig_train_batches

output_dir = '/mloscratch/sayfiddi/detector/base_data/processed'

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
    print(images.shape)
    print(boxes.shape)
