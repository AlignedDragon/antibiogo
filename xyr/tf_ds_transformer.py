import os

from numpy.random import seed as seednp
import tensorflow as tf
from typing import Tuple
from pathlib import Path
from os import listdir
import json
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir, corDictDir

tf.random.set_seed(tf_global_seed)
seednp(np_seed)
# tf.config.run_functions_eagerly(True)

vald_ratio = 0.2
test_ratio = 0.1

img_dir = [os.path.join(img_pth, x) for x in os.listdir(img_pth)]

img_list_ds = tf.data.Dataset.list_files(img_dir, shuffle=False)

data_count = tf.data.experimental.cardinality(img_list_ds).numpy()


def normalize(input_img: tf.Tensor) -> tf.Tensor:
    input_image = -1 + tf.cast(input_img, tf.float32) / 127.5
    return input_image


def load_image(input_img: tf.string, corr:dict) -> tuple[tf.Tensor, tf.Tensor]:
    if isinstance(input_img, tf.Tensor):
        input_img = input_img.numpy().decode('utf-8')
    filename = os.path.basename(input_img)
    target = tf.convert_to_tensor(corr[filename]/(256/IMG_SIZE), dtype=tf.float32)

    img = tf.io.read_file(input_img)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method= tf.image.ResizeMethod.BILINEAR, antialias=False)
    img = normalize(img)

    return img, target

@tf.py_function(Tout= (tf.float32, tf.float32))
def process_path(image_path: str) -> Tuple:
    corr = json.load(open(corDictDir))
    img, target = load_image(image_path, corr)
    return img, target



ready_ds = img_list_ds.take(data_count).shuffle(BUFFER_SIZE, seed=shuffle_data_seed).map(process_path,
                                                                                         num_parallel_calls=AUTOTUNE)

test_size = int(data_count * test_ratio)
test_ds = ready_ds.take(test_size)
val_size = int(data_count * vald_ratio)
val_ds = ready_ds.skip(test_size).take(val_size)
train_size = data_count - val_size - test_size
train_ds_original = ready_ds.skip(test_size + val_size).take(train_size)

Path(orig_train_dir).mkdir(parents=True)
dir_files = listdir(orig_train_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

train_ds_original.save(orig_train_dir)

Path(val_dir).mkdir(parents=True)
dir_files = listdir(val_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

val_ds.save(val_dir)

Path(test_dir).mkdir(parents=True)
dir_files = listdir(test_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

test_ds.save(test_dir)
