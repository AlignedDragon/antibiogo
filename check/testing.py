import json
import os

import numpy as np
import cv2 as cv


def match_points(truth, pred, max_dist):
    """
    Match truth points with prediction points based on Euclidean distance.

    Parameters:
    - truth: numpy array of truth points (N x 3 array)
    - pred: numpy array of predicted points (M x 3 array)
    - max_dist: maximum allowed Euclidean distance for matching

    Returns:
    - matches: Array of matched point indices, with 0 for unmatched points
      [5,2,4] truth - with index 0 matches pred with index 5
    """
    truth = np.array(truth)
    pred = np.array(pred)
    # If either array is empty, return array of zeros
    if len(truth) == 0 or len(pred) == 0:
        return np.zeros(len(truth), dtype=int)

    # Initialize matches array with zeros
    matches = np.zeros(len(truth), dtype=int)

    # Create copy of pred to track used predictions
    used_pred = np.zeros(len(pred), dtype=bool)

    # Iterate through truth points
    for t_idx, truth_point in enumerate(truth[:, :2]):
        # Find minimum distance prediction
        min_dist = float('inf')
        best_pred_idx = -1

        # Check against all unused predictions
        for p_idx, pred_point in enumerate(pred[:, :2]):
            if not used_pred[p_idx]:
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((truth_point - pred_point) ** 2))

                # Update if closer and within max_dist
                if dist < min_dist and dist <= max_dist:
                    min_dist = dist
                    best_pred_idx = p_idx

        # If a match is found, update matches array
        # Indices are 1-based to distinguish from 0 (unmatched)
        if best_pred_idx != -1:
            matches[t_idx] = best_pred_idx + 1
            used_pred[best_pred_idx] = True

    return matches

def via_plant(data, via):
    for key, value in data.items():
        for i in range(len(value)):
            via['metadata'][f'{key.split('.')[0]}_{i}'] = {}
            via['metadata'][f'{key.split('.')[0]}_{i}']['vid'] = f'{key.split('.')[0]}'
            via['metadata'][f'{key.split('.')[0]}_{i}']['flg'] = 0
            via['metadata'][f'{key.split('.')[0]}_{i}']['z'] = []
            via['metadata'][f'{key.split('.')[0]}_{i}']['xy'] = [3, value[i][0], value[i][1], pellets[key]]
            via['metadata'][f'{key.split('.')[0]}_{i}']['av'] = {}
    # with open('/home/muhammad-ali/Downloads/via_project_05Dec2024_11h09m30s.json', 'w') as f:
    #     json.dump(via, f)
    """for rectangles improc has x,y,w,h format where xy are top-left corners"""

def read_csv():
    with open("/home/muhammad-ali/Downloads/Output1.csv", "r") as file:
        lines = file.readlines()
        data = [line.strip().split(",") for line in lines]

def get_from_via():
    aim = {}
    for i in range(1, 1629):
        aim[f'{i}.jpg'] = []
    for key, value in via['metadata'].items():
        aim[f'{value['vid']}.jpg'] += [value['xy'][1:]]


