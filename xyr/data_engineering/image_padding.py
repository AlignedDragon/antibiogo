import os
import PIL
from PIL.Image import open


def padding(img_mask, expected_size):
    img, mask = img_mask
    desired_size = expected_size
    delta_width = desired_size[0] - img.size[0]
    delta_height = desired_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return PIL.ImageOps.expand(img, padding), PIL.ImageOps.expand(mask, padding)

archive = '/home/muhammad-ali/working/base_data'

img_dir = os.path.join(archive, 'images')
mask_dir = os.path.join(archive, 'masks_01')
padded_im = '/home/muhammad-ali/working/base_data/padded_images'
padded_ms = '/home/muhammad-ali/working/base_data/padded_masks'

img_list = sorted(list(os.listdir(img_dir)))
mask_list = sorted(list(os.listdir(mask_dir)))

img_dirs = [os.path.join(img_dir, x) for x in img_list]
mask_dirs = [os.path.join(mask_dir, x) for x in mask_list]

for i in range(len(img_dirs)):
    maxim = max(image.size)
    image = PIL.Image.open(img_dirs[i])
    mask = PIL.Image.open(mask_dirs[i])
    image, mask = padding((image, mask), (maxim, maxim))
    image.save(str(padded_im + f"/{img_list[i]}"), 'JPEG')
    mask.save(str(padded_ms + f"/{mask_list[i]}"), 'JPEG')

