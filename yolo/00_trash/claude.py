import os
from numpy.random import seed as seednp
from keras_cv import bounding_box
import tensorflow as tf
import json
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir, annot_path, classes_path

# Set seeds for reproducibility
tf.random.set_seed(tf_global_seed)
seednp(np_seed)

# Dataset split ratios
val_ratio = 0.2
test_ratio = 0.1

# Load annotations and classes
annot = json.load(open(annot_path))
classDict = json.load(open(classes_path))
data_count = len(annot)

# Prepare data lists
bbox = []
classes = []
image_paths = []

for key, value in annot.items():
    image_paths.append(os.path.join(img_pth, key))
    classes.append(classDict[key])
    bbox.append(value)

# Convert to ragged tensors
bbox = tf.ragged.constant(bbox)
classes = tf.ragged.constant(classes)
image_paths = tf.ragged.constant(image_paths)


def load_image(image_path):
    """Load and preprocess single image"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False
    )
    return tf.cast(image, tf.float32)


def prepare_bounding_boxes(bbox, classes):
    """Prepare bounding boxes in correct format"""
    # Convert to float and normalize coordinates
    boxes = tf.cast(bbox, dtype=tf.float32) / 4.0
    classes = tf.cast(classes, dtype=tf.float32)

    # Create bounding box dictionary
    bounding_boxes = {
        "boxes": boxes,
        "classes": classes,
    }

    # Convert to dense format with padding
    return bounding_box.to_dense(
        bounding_boxes,
        max_boxes=16,  # Adjust max_boxes as needed
    )


def load_dataset(image_path, classes, bbox):
    """Load and prepare single example"""
    image = load_image(image_path)
    boxes = prepare_bounding_boxes(bbox, classes)
    return image, boxes


# Create initial dataset
data = tf.data.Dataset.from_tensor_slices((image_paths, classes, bbox))

# Apply preprocessing
processed_ds = data.map(
    load_dataset,
    num_parallel_calls=AUTOTUNE
).shuffle(
    BUFFER_SIZE,
    seed=shuffle_data_seed,
    reshuffle_each_iteration=True
)

# Calculate split sizes
test_size = int(data_count * test_ratio)
val_size = int(data_count * val_ratio)
train_size = data_count - val_size - test_size

# Split dataset
test_ds = processed_ds.take(test_size)
remaining_ds = processed_ds.skip(test_size)
val_ds = remaining_ds.take(val_size)
train_ds = remaining_ds.skip(val_size)

# Configure datasets for training
BATCH_SIZE = 16  # Adjust as needed


def configure_dataset(ds, training=False):
    """Configure dataset with batching and prefetching"""
    # Add batching with padding
    ds = ds.padded_batch(
        BATCH_SIZE,
        padded_shapes=(
            (IMG_SIZE, IMG_SIZE, 3),
            {
                "boxes": (None, 4),
                "classes": (None,)
            }
        ),
        padding_values=(
            0.0,
            {
                "boxes": 0.0,
                "classes": 0.0
            }
        )
    )

    # Add prefetching
    ds = ds.prefetch(AUTOTUNE)

    # Add augmentation if training
    if training:
        ds = ds.map(
            augment_data,
            num_parallel_calls=AUTOTUNE
        )

    return ds


def augment_data(image, boxes):
    """Add data augmentation"""
    # Add your augmentation here if needed
    # Example:
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, boxes


# Configure final datasets
train_ds = configure_dataset(train_ds, training=True)
val_ds = configure_dataset(val_ds, training=False)
test_ds = configure_dataset(test_ds, training=False)

# Save datasets if needed
for ds, dir_path in [
    (train_ds, train_dir),
    (val_ds, val_dir),
    (test_ds, test_dir)
]:
    os.makedirs(dir_path, exist_ok=True)
    dir_files = os.listdir(dir_path)
    if ".DS_Store" in dir_files:
        dir_files.remove(".DS_Store")
    if len(dir_files) > 0:
        raise ValueError(f"The directory {dir_path} exists and is not empty.")
    ds.save(dir_path)



#
# # Train model
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=100,
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint(
#             'yolo_model_{epoch:02d}.h5',
#             monitor='val_mean_average_precision',
#             mode='max',
#             save_best_only=True
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_mean_average_precision',
#             mode='max',
#             patience=10,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_mean_average_precision',
#             mode='max',
#             factor=0.1,
#             patience=5,
#             min_lr=1e-6
#         )
#     ]
# )