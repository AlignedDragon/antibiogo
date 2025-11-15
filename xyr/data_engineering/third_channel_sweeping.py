import numpy as np
import os
import cv2 as cv

archive = '/home/muhammad-ali/working/base_data'
save = '/home/muhammad-ali/working/base_data/cleaned'
mask_dir = os.path.join(archive, 'masks_01')

mask_list = sorted(os.listdir(mask_dir))
mask_dirs = [os.path.join(mask_dir, x) for x in mask_list]

for i in range(len(mask_dirs)):
    mask = cv.imread(mask_dirs[i])
    mask = (2 * mask > 1).astype(np.int8)
    cv.imwrite((str(save) + f"/{mask_list[i]}"), mask)

    if i % 100 == 0:
        print(f"{i} images done")