import os, json
import tensorflow as tf
from tqdm import tqdm
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, root_path, \
    train_dir, val_dir, test_dir


def normalize(img):
    img = -1 + tf.cast(img, tf.float32) / 127.5
    return img

def get_lookup_table(all_keys, all_values):
    """Creates a static lookup table inside the TF graph."""
    keys_tensor = tf.constant(all_keys)
    vals_tensor = tf.constant(all_values)
    initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
    # default_value -1.0 handles missing keys safely
    return tf.lookup.StaticHashTable(initializer, default_value=-1.0) 

# 2. Pre-load all metadata into Python lists first
database = os.path.join(root_path, "xyr_database")
global_keys = []
global_values = []

# We iterate first just to build the global dictionary
print("Building Lookup Table...")
for split in ["train", "val", "test"]:
    radii_path = os.path.join(database, split, "radii.json")
    with open(radii_path) as f:
        data = json.load(f)
        for k, v in data.items():
            global_keys.append(k)
            global_values.append(v / (256 / IMG_SIZE))

# 3. Create the Graph Table
lookup_table = get_lookup_table(global_keys, global_values)

def load_and_process(file_path):
    parts = tf.strings.split(file_path, os.sep)
    filename = parts[-1] 
    target = lookup_table.lookup(filename)
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method='bilinear')
    img = normalize(img)
    
    return img, target

# 4. Build and Save Datasets
data_dict = {}
for split in tqdm(os.listdir(database)):
    img_path_pattern = os.path.join(database, split, "images", "*")
    ds = tf.data.Dataset.list_files(img_path_pattern, shuffle=False)
    data_dict[split] = ds.map(load_and_process, num_parallel_calls=AUTOTUNE)

train = data_dict["train"]
val = data_dict["val"]
test = data_dict["test"]
    
for ds, dir_path in tqdm([(train, train_dir), (val, val_dir), (test, test_dir)]):
    os.makedirs(dir_path, exist_ok=True)
    dir_files = os.listdir(dir_path)
    if len(dir_files) > 0:
        raise ValueError(f"The directory {dir_path} exists and is not empty.")
    ds.save(dir_path)