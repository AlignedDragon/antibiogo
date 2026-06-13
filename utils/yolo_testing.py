from yolo_dataloader import test_batches
from yolo_utils import drawer, targetize
import numpy as np
import os
import cv2
from yolo import yolo
import tensorflow as tf
from tensorflow.keras.utils import array_to_img

output_dir = '/home/muhammad-ali/working/check/yolo_processed'
model_path = '/home/muhammad-ali/working/check/saved_models/YOLOxsS512.h5'
yolo.load_weights(model_path)
i = 0
# Iterate through batches
import time
length = []
for batch_idx, batch in enumerate(test_batches):
    start = time.time()
    images = batch[0]  # Shape: (32, 256, 256, 3)
    boxes = np.array(batch[1]['boxes'])    # Shape: (32, 3)

    for img_idx, (image, img_boxes) in enumerate(zip(images, boxes)):
        # img_height, img_width = image.shape[:2]
        # img = image * 255  # Denormalize if necessary  # Convert to BGR for OpenCV
        start = time.time()
        yolo.predict(image[tf.newaxis, ...])
        end = time.time()
        length += [end - start]


        # img_with_boxes = drawer(array_to_img(image), [img_boxes, yolo.predict(image[tf.newaxis, ...])['boxes'][0]])

        # img_with_boxes = cv2.cvtColor(np.array(img_with_boxes), cv2.COLOR_RGB2BGR)
        # output_path = os.path.join(output_dir, f"batch_{batch_idx}_img_{img_idx}.jpg")
        # cv2.imwrite(output_path, img_with_boxes)
        # i += 1
        # if i % 100 == 0:
        #     print(f'{i} images done')


print(length)
print(np.mean(length))
print(f"Visualizations saved to {output_dir}")
# drawer(array_to_img(sample_image), [sample_target, targetize(self.model.predict(sample_image[tf.newaxis, ...]))]