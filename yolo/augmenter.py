from numpy.random import seed as seednp
import tensorflow as tf
import tensorflow.keras as keras
import keras_cv
from pathlib import Path
from os import listdir
from utils import IMG_SIZE,AUTOTUNE, BATCH_SIZE, tf_global_seed,np_seed,train_dir,orig_train_dir


# The global random seed.
tf.random.set_seed(tf_global_seed)
seednp(np_seed)

train_ds = tf.data.Dataset.load(orig_train_dir)

augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(mode="horizontal_and_vertical", bounding_box_format="center_xywh"),
        keras_cv.layers.RandomShear(x_factor=0.2, y_factor=0.2, bounding_box_format="center_xywh"),
        keras_cv.layers.JitteredResize(
            target_size=(IMG_SIZE, IMG_SIZE), 
            scale_factor=(0.75, 1.3), 
            bounding_box_format="center_xywh"
        ),
    ]
)

def augment_fn(x, training=True):
    return augmenter(x) if training else x

train_ds = train_ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

# Save the Training dataset with the augmented one.
Path(train_dir).mkdir(parents=True)
dir_files = listdir(train_dir)
if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
if len(dir_files) > 0: raise ValueError("The directory exists and is not empty.")

train_ds.save(train_dir)
