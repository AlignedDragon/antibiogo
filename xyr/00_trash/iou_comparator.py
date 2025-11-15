import json
import cv2 as cv
import os
import numpy as np
import cv2 as cv

def calculate_iou(pred_xyr: list, truth_xyr: list, size: tuple):
    h, w = size
    pred = np.zeros([h, w])
    truth = np.zeros([h, w])
    for circ in pred_xyr:
        x, y, r = circ
        cv.circle(pred, (int(x), int(y)), int(r), 1, -1)
    for circ in truth_xyr:
        x, y, r = circ
        cv.circle(truth, (int(x), int(y)), int(r), 1, -1)

    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()

    if union == 0:
        return 1 if intersection == 0 else 0  # Perfect match if both are empty, else 0
    return intersection / union


improc = json.load(open('/home/muhammad-ali/Downloads/improc.json'))
base = json.load(open('/hybrid_detector/base_data/img_circles.json'))
folder = '/home/muhammad-ali/working/hybrid_detector/base_data/images'

c = 0
iou = {}
for f in os.listdir(folder):
    img = cv.imread(os.path.join(folder, f))
    size = img.shape[:2]
    iou[f] = calculate_iou(improc[f], base[f], size)
    c+= 1
    if c%100 == 0:
        print(f'{c} images done')
with open('/home/muhammad-ali/Downloads/eval.json', 'w') as f:
    json.dump(iou, f)
# for key, value in corr:

