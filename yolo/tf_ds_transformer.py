import os, json
from numpy.random import seed as seednp
import tensorflow as tf
import keras_cv
from tqdm import tqdm

from utils import BUFFER_SIZE, shuffle_data_seed, train_dir, \
    val_dir, test_dir, orig_train_dir,root_path, load_dataset

# assuming that database has only train, val and test folders
# inside the database each folder must have images, annot.json, and classes.json
database_dir = os.path.join(root_path, "yolo_database")
data_dict = {}

for split in tqdm(os.listdir(database_dir)):
    img_pth = os.path.join(*[database_dir, split, "images"])
    annot_path = os.path.join(*[database_dir, split, "annot.json"])
    classes_path = os.path.join(*[database_dir, split, "classes.json"])
    
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

    bbox = tf.constant(bbox)
    classes = tf.constant(classes)
    image_paths = tf.constant(image_paths) 

    data_dict[split] = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))
    data_dict[split] = data_dict[f].map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE,seed=shuffle_data_seed)

train = data_dict["train"]
val = data_dict["val"]
test = data_dict["test"]

for ds, dir_path in [(train, orig_train_dir), (val, val_dir), (test, test_dir)]:

    os.makedirs(dir_path, exist_ok=True)
    dir_files = os.listdir(dir_path)
    if ".DS_Store" in dir_files: dir_files.remove(".DS_Store")
    if len(dir_files) > 0:
        raise ValueError(f"The directory {dir_path} exists and is not empty.")
    ds.save(dir_path)
