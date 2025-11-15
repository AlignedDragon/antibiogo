"""
This file takes data and masks and produces bulk images with masks applied. Conclusion taken was that the data is not clean - masks are
misapplied (with boxed circles).
"""

import numpy as np
import os
from PIL.Image import open
import PIL.Image as Image


archive = '/home/muhammad-ali/working/OD+prediction/xyr_data'
save = '/home/muhammad-ali/working/OD+prediction/xyr_data/processed'
col_trans = (132, 112, 255, 255)

img_dir = os.path.join(archive, 'images')
mask_dir = os.path.join(archive, 'masks')

img_list = sorted(os.listdir(img_dir))
mask_list = sorted(os.listdir(mask_dir))

img_dirs = [os.path.join(img_dir, x) for x in img_list]
mask_dirs = [os.path.join(mask_dir, x) for x in mask_list]


for i in range(len(img_dirs)):
    image = open(img_dirs[i]).convert("RGBA")
    mask = open(mask_dirs[i]).resize(image.size)
    mask =  1 - np.array(mask)//255
    mask = 190 + 60*mask

    shape = (image.size[0], image.size[0], 4)
    alpha = np.full(shape, col_trans, dtype=np.uint8)

    mask = Image.fromarray(np.uint8(mask)).convert('L')
    alpha = Image.fromarray(alpha)

    alpha.paste(image, (0, 0), mask)
    alpha = alpha.convert('RGB')

    alpha.save(str(save + f"/{img_list[i]}"), 'JPEG')
    if i%100 == 0:
        print(f"{i} images done")