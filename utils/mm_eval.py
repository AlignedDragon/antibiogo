import numpy as np
import json
from PIL.Image import open as imread
import os
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns

def match_points(truth, pred, max_dist):
    """
    Match truth points with prediction points based on Euclidean distance.

    Parameters:
    - truth: numpy array of truth points (N x 3 array)
    - pred: numpy array of predicted points (M x 3 array)
    - max_dist: maximum allowed Euclidean distance for matching

    Returns:
    - matches: Array of matched point indices, with 0 for unmatched points
    """
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



draw_dir = '/home/muhammad-ali/working/hungarian/base_data/ready'
truth = json.load(open('circles_H1024.json'))
improc = json.load(open('improc_1024.json'))
pellet = json.load(open('pellet_annot.json'))

improc_err = json.load(open('improc_mmErrorMiss.json'))
xyr_err = json.load(open('xyr_mmError.json'))


## plotting for error
# print(np.mean((np.abs(improc_err) < 2.5) & (np.abs(improc_err) > 0.5)
# ))
# sns.violinplot(x=improc_err, color='red', fill=False, label='Improc Error')
# sns.violinplot(x=xyr_err, color='blue', fill=False, label='Model Error')
#
# # Add a vertical line at x = 1
# plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
# plt.axvline(x=-1, color='black', linestyle='--', linewidth=1)
#
# # Add a legend
# plt.legend()
#
# # Show the plot
# plt.title("Violin Plots with Labels and Reference Line")
# plt.show()


# diff = []
# for key, value in truth.items():
#     true = np.array(truth[key])
#     pred = np.array(improc[key])
#     k = pellet[key][0][2]/6 # pixel per mm
#     print(pellet[key][0][2], 'pxl diameter')
#     print(pellet[key][0][2]/6, 'pxl/mm')
#     matches = match_points(true, pred, max_dist= 16)
#     for i in range(len(matches)):
#         if matches[i] !=0:
#             delta = (pred[matches[i]-1][2]-true[i][2])/k
#             diff.append(delta)
#             print(pred[matches[i]-1][2]-true[i][2], 'pxl diff')
#             print((pred[matches[i]-1][2]-true[i][2])/k, 'mm diff')
        # else:
        #     delta = -true[i][2]/k
        #     diff.append(delta)

# with open('improc_mmErrorNoMiss.json', 'w') as f:
#     json.dump(diff, f)

# # improc to 1024
# img_dir = '/home/muhammad-ali/working/hungarian/base_data/images'
# bi = 0
# new_improc = {}
# for i in os.listdir(img_dir):
#     img = cv.imread(os.path.join(img_dir, i))
#     # draw = cv.imread(os.path.join(draw_dir,i))
#     h,w,_ = img.shape
#     dim = [h,w]
#     if np.argmax(dim)==0:
#         x_0 = (h-w)/2
#         y_0 = 0
#     if np.argmax(dim)==1:
#         x_0 = 0
#         y_0 = (w-h)/2
#     s = (np.max(dim), np.max(dim))
#     k = float(1024/s[0])
#     new_labels = []
#     for label in improc[i]:
#         new_labels += [[k*(float(label[0])+x_0),k*(float(label[1])+y_0),k*(float(label[2]))]]
#         # x,y,r = k*(float(label[0])+x_0),k*(float(label[1])+y_0),k*(float(label[2]))
#         # cv.circle(draw, (int(x),int(y)),int(r), (255,0,0), 2)
#     # print(new_labels)
#     # cv.imwrite(os.path.join('/home/muhammad-ali/working/check/processed', i), draw)
#     bi += 1
#     if bi%100 == 0:
#         print(f"{bi} images done")
#     new_improc[i] = new_labels
# with open('improc_1024.json', 'w') as f:
#     json.dump(new_improc, f)
# bi = 0
# new_improc = json.load(open('improc_1024.json'))
# for i in os.listdir(draw_dir):
#     draw = cv.imread(os.path.join(draw_dir, i))
#     for label in new_improc[i]:
#         x,y,r = label[0], label[1], label[2]
#         cv.circle(draw, (int(x),int(y)),int(r), (255,0,0), 2)
#     cv.imwrite(os.path.join('/home/muhammad-ali/working/check/processed', i), draw)
#     bi += 1
#     if bi%100 == 0:
#         print(f"{bi} images done")
#         break

# improc = json.load(open('improc_1024.json'))
# for k, v in improc.items():
#     if len(v) > 16:
#         print('ishkal')
#         for i in range(len(v)):
#             if v[i] == [228.4119833669355, 670.0846774193549, 24.093406012038145]:
#                 print(v[i])
#                 v.pop(i)
#                 break
#         improc[k] = v
# with open('improc_1024.json', 'w') as f:
#     json.dump(improc, f)

# img = cv.imread(os.path.join(draw_dir, '703.jpg'))
# print(img.shape)
# print((img.shape[0]/2, img.shape[1]/2))
# # img = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
# print(img)
# cv.imshow('image',img)
# cv.waitKey(0)
