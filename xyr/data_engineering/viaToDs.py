import json
import os
import cv2 as cv
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

def padding(img, desired_size):
    """generates an image with padded with (0,0,0) to the desired_size"""
    old_image_height, old_image_width, channels = img.shape
    new_image_height, new_image_width = desired_size
    # create new image of desired size and color (black) for padding
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img

    return result

def pad_resize(fily: dict, data: str, mask_dir: str, fill: int, size: int):
    """accesses images -> pads and resizes to squares with sizexsize, simultaneously creating the corresponding masks with 'fill' being
       the values to be filled inside the circles"""
    bi = 0
    for i in os.listdir(data):
        i_dir = os.path.join(data, i)
        m_dir = os.path.join(mask_dir, i)
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
        mask = np.zeros((size, size))
        for label in labels[vid_tran[i]]:
            if len(fily['metadata'][label]['xy']) != 0:
                x = k*(float(fily['metadata'][label]['xy'][1])+x_0)
                y = k*(float(fily['metadata'][label]['xy'][2])+y_0)
                r = k*(float(fily['metadata'][label]['xy'][3]))
                cv.circle(mask, (int(x),int(y)), int(r), fill, -1)
        cv.imwrite(m_dir, mask)
        cv.imwrite(i_dir, img)
        bi += 1
        if bi%100 == 0:
            print(f"{bi} images done")
    return

if __name__ == '__main__':
    """fily: json file containing x,y,r
       data: corresponding images
       mask_dir: where it needs to be stored
       fill: value to be filled inside the circle
       size: the size of the square image"""

    fily = json.load(open('/home/muhammad-ali/working/base_data/images.json'))
    data = '/home/muhammad-ali/working/base_data/images'
    mask_dir = '/home/muhammad-ali/working/base_data/masks_01'
    fill = 255
    size = 1024

    vid = vid_gen(fily)
    vid_tran = vid_tr(vid)
    labels = label_gen(vid, fily)
    pad_resize(fily, data, mask_dir, fill, size)
