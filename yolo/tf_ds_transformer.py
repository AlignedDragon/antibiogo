import os
from numpy.random import seed as seednp
from keras_cv import bounding_box
import tensorflow as tf
import json
import keras_cv
from tensorflow import keras

from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir, annot_path, classes_path

tf.random.set_seed(tf_global_seed)
seednp(np_seed)
# tf.config.run_functions_eagerly(True)


vald_ratio = 0.2
test_ratio = 0.1


annot = json.load(open(annot_path))
classDict = json.load(open(classes_path))
data_count = len(annot)

bbox = []
classes = []
image_paths = []

for key, value in annot.items():
    image_paths.append(os.path.join(img_pth, key))
    classes.append(classDict[key])
    bbox.append(value)

bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)

data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE),method=tf.image.ResizeMethod.BILINEAR, antialias=False)
    image = tf.cast(image, tf.float32)/255.
    return image


def load_dataset(image_path, classes, bbox):

    image = load_image(image_path)
    bounding_boxes = {
        "classes": tf.cast(classes, dtype=tf.float32),
        "boxes": tf.cast(bbox/(1024/IMG_SIZE), dtype = tf.float32),
    }
    return image, bounding_box.to_dense(
        bounding_boxes, max_boxes=16
    )

ready_ds = data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE,seed=shuffle_data_seed)


test_size = int(data_count * test_ratio)
test_ds = ready_ds.take(test_size)
val_size = int(data_count * vald_ratio)
val_ds = ready_ds.skip(test_size).take(val_size)
train_size = data_count - val_size - test_size
train_ds = ready_ds.skip(test_size + val_size).take(train_size)
orig_train = train_ds


# augmenters = [
#         keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="center_xywh"),
#         keras_cv.layers.JitteredResize(
#             target_size=(IMG_SIZE, IMG_SIZE), scale_factor=(0.75, 1.3), bounding_box_format="center_xywh"
#         ),
#     ]
# def create_augmenter_fn(augmenters):
#     def augmenter_fn(image, bboxes):
#         for augmenter in augmenters:
#             # Handle image and bounding boxes separately
#             if isinstance(augmenter, keras_cv.layers.RandomFlip):
#                 image = augmenter(image)
#                 # For flip, you might need to adjust bounding boxes
#             elif isinstance(augmenter, keras_cv.layers.JitteredResize):
#                 image, bboxes = augmenter(image, bounding_boxes=bboxes)
#         return image, bboxes
#
#     return augmenter_fn

# augmenter_fn = create_augmenter_fn(augmenters)
#
# train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)

for ds, dir_path in [(orig_train, orig_train_dir), (val_ds, val_dir), (test_ds, test_dir)]:

    os.makedirs(dir_path, exist_ok=True)
    dir_files = os.listdir(dir_path)
    if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
    if len(dir_files) > 0:
        raise ValueError(f"The directory {dir_path} exists and is not empty.")
    ds.save(dir_path)
