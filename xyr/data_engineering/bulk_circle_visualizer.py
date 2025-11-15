import json
import cv2 as cv
import os
import numpy as np


def vid_gen(fily: dict) -> dict:
    """generates a dict where {key: value} is {vid: filename}"""
    vid = {}
    for key, value in fily['file'].items():
        vid[key] = value['fname']
    return vid

def vid_tr(vid: dict) -> dict:
    """generates a dict where {key: value} is {filename: vid}"""
    vid_tran = {}
    for key, value in vid.items():
        vid_tran[value] = key
    return vid_tran

def label_gen(vid: dict, fily: dict)-> dict:
    """generates a dict where {key: value} is {vid: [list of label names]}"""
    labels = {}
    for key, value in vid.items():
        labels[key] = []
    for key, value in fily['metadata'].items():
        if len(value['xy']) != 0:
            labels[value['vid']] += [key]
    return labels

fily = json.load(open('/home/muhammad-ali/working/OD+prediction/base_data/images.json'))
folder_path = '/home/muhammad-ali/working/OD+prediction/base_data/images'
save = '/home/muhammad-ali/working/OD+prediction/base_data/processed'

vid = vid_gen(fily)
labels = label_gen(vid, fily)

circles = {}
for v, ls in labels.items():
    circles[f'{v}.jpg'] = []
    for l in ls:
        circles[f'{v}.jpg'] += [fily['metadata'][l]['xy'][1:]]

i = 0
for img_name, circles in circles.items():
    img_path = os.path.join(folder_path, img_name)
    img = cv.imread(img_path)
    
    # Calculate a scaling factor based on image size
    scale_factor = min(img.shape[1], img.shape[0]) / 500  # 500 is an arbitrary base size
    font_scale = max(0.5 * scale_factor, 0.3)//2  # Avoid too small font
    thickness = max(int(2 * scale_factor), 1)//2  # Ensure thickness is at least 1

    # Find min, max radius and count of circles
    radii = [circle[2] for circle in circles]
    max_radius = max(radii)
    min_radius = min(radii)
    circle_count = len(circles)

    # Draw each circle and annotate

    for x, y, r in circles:
        # Draw the circle
        x,y,r = int(x), int(y), int(r)
        cv.circle(img, (x, y), r, (0, 255, 0), thickness)
        # Write the coordinates and radius at the center of each circle
        cv.putText(img, f"({x},{y}), r={r}", (x, y), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Write max, min radius and circle count at the bottom left corner
    bottom_left_text = f"Max Radius: {max_radius}, Min Radius: {min_radius}, Count: {circle_count}"
    cv.putText(img, bottom_left_text, (10, img.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    # Save or display the annotated image
    output_path = os.path.join(save, img_name)
    cv.imwrite(output_path, img)
    i+= 1
    if i%100 ==0:
        print(f'{i} images done')