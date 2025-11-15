from xyr_dataloader import test_batches
from xyr_utils import drawer, targetize
import numpy as np
import os
import cv2 as cv
from xyr_model import model
import tensorflow as tf
import json

model_path = '/home/muhammad-ali/working/check/saved_models/xyrV3SmallS256.h5'
model.load_weights(model_path)
i = 0
# Iterate through batches
def calculate_iou(pred_xyr: list, truth_xyr: list, size: tuple):
    h, w = size
    pred = np.zeros([h, w])
    truth = np.zeros([h, w])
    x, y, r = pred_xyr
    cv.circle(pred, (int(x), int(y)), int(r), 1, -1)

    x, y, r = truth_xyr
    cv.circle(truth, (int(x), int(y)), int(r), 1, -1)

    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()

    if union == 0:
        return 1 if intersection == 0 else 0  # Perfect match if both are empty, else 0
    return intersection / union

results = []
for batch_idx, batch in enumerate(test_batches):
    images = batch[0]  # Shape: (32, 256, 256, 3)
    true_circles = np.array(batch[1])  # Shape: (32, 3)

    for img_idx, (image, img_circles) in enumerate(zip(images, true_circles)):
        results += [calculate_iou(list(targetize(model.predict(image[tf.newaxis, ...]))), list(img_circles), (256,256))]
        i += 1
        if i % 100 == 0:
            print(f'{i} images done')
print(np.mean(results))
print(results)
with open('iou.json', 'w') as f:
    json.dump(results, f)