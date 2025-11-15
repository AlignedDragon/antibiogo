import os
from numpy.random import seed as seednp
import numpy as np
import tensorflow as tf

import json
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir, annot_path, classes_path

tf.random.set_seed(tf_global_seed)
seednp(np_seed)
tf.config.run_functions_eagerly(True)


vald_ratio = 0.2
test_ratio = 0.1

image_paths = []
bbox = []
classes = []

annot = json.load(open(annot_path))
classDict = json.load(open(classes_path))
data_count = len(annot)

for key, value in annot.items():
    image_paths.append(os.path.join(img_pth, key))
    bbox.append(annot[key])
    classes.append(classDict[key])

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
print(image_paths)

def loader(input_img, cls, bbox):
    img = tf.io.read_file(input_img)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE),
                          method=tf.image.ResizeMethod.BILINEAR,
                          antialias=False)
    return tf.cast(img, tf.float32), {'boxes': tf.cast(bbox,tf.float32) , 'classes': tf.cast(cls, tf.float32)}


ready_ds = data.map(loader, num_parallel_calls= AUTOTUNE).shuffle(BUFFER_SIZE, seed=shuffle_data_seed)
#
# # Split into train/val/test
# test_size = int(data_count * test_ratio)
# test_ds = ready_ds.take(test_size)
# val_size = int(data_count * vald_ratio)
# val_ds = ready_ds.skip(test_size).take(val_size)
# train_size = data_count - val_size - test_size
# train_ds = ready_ds.skip(test_size + val_size).take(train_size)
#
# for ds, dir_path in [(train_ds, orig_train_dir), (val_ds, val_dir), (test_ds, test_dir)]:
#
#     os.makedirs(dir_path, exist_ok=True)
#     dir_files = os.listdir(dir_path)
#     if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
#     if len(dir_files) > 0:
#         raise ValueError(f"The directory {dir_path} exists and is not empty.")
#     ds.save(dir_path)
