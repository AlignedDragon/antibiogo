from xyr_dataloader import test_batches
from xyr_utils import drawer, targetize
import numpy as np
import os
import cv2
from xyr_model import model
import tensorflow as tf
from tensorflow.keras.utils import array_to_img

output_dir = '/home/muhammad-ali/working/check/xyr_processed'
model_path = '/home/muhammad-ali/working/check/saved_models/xyrV3SmallS256.h5'
model.load_weights(model_path)
i = 0
# Iterate through batches
import time
length = []
for batch_idx, batch in enumerate(test_batches):
    images = batch[0]  # Shape: (32, 256, 256, 3)
    true_circles = np.array(batch[1])  # Shape: (32, 3)

    for img_idx, (image, img_circles) in enumerate(zip(images, true_circles)):
        # img_height, img_width = image.shape[:2]
        # img = image * 255  # Denormalize if necessary
        # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        # start = time.time()
        model.predict(image[tf.newaxis, ...])
        # end = time.time()
        # length += [end - start]
        # Convert and draw bounding boxes
        # img_with_circles = drawer(array_to_img(img), [img_circles, targetize(model.predict(image[tf.newaxis, ...]))])
        # img_with_circles = cv2.cvtColor(np.array(img_with_circles), cv2.COLOR_RGB2BGR)
        # Save the image
        # output_path = os.path.join(output_dir, f"batch_{batch_idx}_img_{img_idx}.jpg")

        # cv2.imwrite(output_path, img_with_circles)
        # i += 1
        # if i % 100 == 0:
        #     print(f'{i} images done')
# 387, 162
print(np.mean(length))
# print(f"Visualizations saved to {output_dir}")
# drawer(array_to_img(sample_image), [sample_target, targetize(self.model.predict(sample_image[tf.newaxis, ...]))]