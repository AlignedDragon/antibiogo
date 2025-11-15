import json
import cv2 as cv
import os
import numpy as np
from data_engineering.viaToDs import vid_gen, vid_tr, label_gen, padding

def framer(fily: dict, data: str, frame_dir: str, size: int):
    """accesses images -> pads and resizes to squares with sizexsize, simultaneously creating the corresponding masks with 'fill' being
       the values to be filled inside the circles"""
    bi = 0
    r_max = 0
    correspondance = {}
    for i in os.listdir(data):
        i_dir = os.path.join(data, i)
        img = cv.imread(i_dir, cv.IMREAD_COLOR)
        h,w,_ = img.shape
        dim = [h,w]
        if np.argmax(dim)==0:
            x_0 = (h-w)/2
            y_0 = 0
        if np.argmax(dim)==1:
            x_0 = 0
            y_0 = (w-h)/2
        s = (np.max(dim), np.max(dim))
        img = padding(img, s)
        k = float(size/s[0])
        img = cv.resize(img, (size, size), interpolation = cv.INTER_LINEAR)
        for label in labels[vid_tran[i]]:
            if len(fily['metadata'][label]['xy']) != 0:
                mask = np.zeros((256, 256))
                f_dir = os.path.join(frame_dir, f'images/{label}_{i}')
                m_dir = os.path.join(frame_dir, f'masks/{label}_{i}')
                x = k*(float(fily['metadata'][label]['xy'][1])+x_0)
                y = k*(float(fily['metadata'][label]['xy'][2])+y_0)
                r = k*(float(fily['metadata'][label]['xy'][3]))
                img = padding(img, (1280,1280))
                x = x + 128
                y = y + 128
                x_min, x_max = int(x-128), int(x+128)
                y_min, y_max = int(y-128), int(y+128)

                delta_x = np.random.randint(-10, 11)
                delta_y = np.random.randint(-10, 11)
                x_min += delta_x
                x_max += delta_x
                y_min += delta_y
                y_max += delta_y

                frame = img[y_min:y_max, x_min:x_max]
                cv.imwrite(f_dir, frame)
                correspondance[f'{label}_{i}'] = [x-x_min,y-y_min,r]
                cv.circle(mask, (int(x-x_min), int(y-y_min)), int(r), 255, -1)
                cv.imwrite(m_dir, mask)
        bi += 1
        if bi%100 == 0:
            print(f"{bi} images done")
            if bi//3== 100:
                print(r_max)
                break
    return correspondance

fily = json.load(open('../base_data/images.json'))
data = '../base_data/images'
frame_dir = '../xyr_data'

size = 1024
vid = vid_gen(fily)
vid_tran = vid_tr(vid)
labels = label_gen(vid, fily)
correspond = framer(fily, data, frame_dir, size)
with open('../frame_corr.json', 'w') as f:
    json.dump(correspond, f)