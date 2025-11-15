"""
This file contains code to correct the orientation of images. Masks didn't have this problem

Problem -- the dataset received had images not matching the orientation of masks

"""

from PIL import Image, ExifTags
import os


def correct_orientation(image):
    """
    Function to correct the orientation of images
    """
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image don't have getexif
        pass
    return image

# directories to load from and save to
archive = '/home/muhammad-ali/working/base_data'
save_dir = '/home/muhammad-ali/working/clean_data'

# images are contained in '/images' subdirectory of archive
image_dir = os.path.join(archive, 'images')
img_list = sorted(list(os.listdir(image_dir)))
img_dirs = [os.path.join(image_dir, x) for x in img_list]

# opening images
images = [Image.open(x) for x in img_dirs]


# correcting the orientation
for i in range(len(images)):
    images[i] = correct_orientation(images[i])

# saving corrected images
for i in range(len(images)):
   images[i].save(str(save_dir)+f"/{img_list[i]}", 'JPEG')