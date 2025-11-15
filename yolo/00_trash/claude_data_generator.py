from typing import Dict, Tuple
import tensorflow as tf
import keras_cv
from utils import class_mapping
# from dataloader import originl_train_dataset
from utils import LEARNING_RATE, GLOBAL_CLIPNORM, IMG_SIZE
import os
from numpy.random import seed as seednp
from keras_cv import bounding_box
import tensorflow as tf
import json
from utils import IMG_SIZE, BUFFER_SIZE, AUTOTUNE, shuffle_data_seed, tf_global_seed, np_seed, img_pth, train_dir, \
    val_dir, test_dir, orig_train_dir, annot_path, classes_path



class YOLODataLoader:
    def __init__(
            self,
            img_size: int,
            buffer_size: int,
            autotune: int = tf.data.AUTOTUNE,
            max_boxes: int = 16
    ):
        self.img_size = img_size
        self.buffer_size = buffer_size
        self.autotune = autotune
        self.max_boxes = max_boxes

    def load_image(self, image_path: str) -> tf.Tensor:
        """Load and preprocess image"""
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(
            image,
            (self.img_size, self.img_size),
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False
        )
        return tf.cast(image, tf.float32)

    def prepare_bbox_format(
            self,
            bbox: tf.RaggedTensor,
            classes: tf.RaggedTensor
    ) -> Dict[str, tf.Tensor]:
        """Convert bounding boxes to dense format"""
        # Ensure boxes are in correct format
        boxes = tf.cast(bbox, dtype=tf.float32) / 4.0  # Normalize coordinates

        # Convert classes to float32
        classes = tf.cast(classes, dtype=tf.float32)

        # Create bounding box dictionary with dense tensors
        bounding_boxes = {
            "boxes": boxes,
            "classes": classes,
        }

        # Convert to dense format with padding
        dense_boxes = bounding_box.to_dense(
            bounding_boxes,
            max_boxes=self.max_boxes
        )

        return dense_boxes

    def load_dataset(
            self,
            image_path: tf.Tensor,
            classes: tf.Tensor,
            bbox: tf.Tensor
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Load and prepare single example"""
        image = self.load_image(image_path)
        boxes = self.prepare_bbox_format(bbox, classes)
        return image, boxes

    def create_dataset(
            self,
            image_paths: list,
            classes: list,
            bbox: list,
            batch_size: int
    ) -> tf.data.Dataset:
        """Create complete dataset"""
        # Convert to ragged tensors
        bbox = tf.ragged.constant(bbox)
        classes = tf.ragged.constant(classes)
        image_paths = tf.ragged.constant(image_paths)

        # Create initial dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            image_paths, classes, bbox
        ))

        # Apply transformations
        dataset = dataset.map(
            self.load_dataset,
            num_parallel_calls=self.autotune
        )

        # Shuffle and batch
        dataset = dataset.shuffle(
            self.buffer_size,
            reshuffle_each_iteration=True
        )

        # Use padded batch for variable number of boxes
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(
                (self.img_size, self.img_size, 3),
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

        return dataset.prefetch(self.autotune)


# Usage example
def prepare_datasets(
        annot_path: str,
        classes_path: str,
        img_path: str,
        img_size: int,
        batch_size: int,
        buffer_size: int,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    # Load annotations and classes
    annot = json.load(open(annot_path))
    class_dict = json.load(open(classes_path))

    # Prepare data lists
    bbox = []
    classes = []
    image_paths = []

    for key, value in annot.items():
        image_paths.append(os.path.join(img_path, key))
        classes.append(class_dict[key])
        bbox.append(value)

    # Create data loader
    data_loader = YOLODataLoader(
        img_size=img_size,
        buffer_size=buffer_size
    )

    # Calculate splits
    data_count = len(annot)
    test_size = int(data_count * test_ratio)
    val_size = int(data_count * val_ratio)
    train_size = data_count - val_size - test_size

    # Split data
    indices = tf.range(data_count)
    indices = tf.random.shuffle(indices)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    # Create datasets
    train_ds = data_loader.create_dataset(
        [image_paths[i] for i in train_indices],
        [classes[i] for i in train_indices],
        [bbox[i] for i in train_indices],
        batch_size
    )

    val_ds = data_loader.create_dataset(
        [image_paths[i] for i in val_indices],
        [classes[i] for i in val_indices],
        [bbox[i] for i in val_indices],
        batch_size
    )

    test_ds = data_loader.create_dataset(
        [image_paths[i] for i in test_indices],
        [classes[i] for i in test_indices],
        [bbox[i] for i in test_indices],
        batch_size
    )

    return train_ds, val_ds, test_ds


# Create datasets
BATCH_SIZE = 16  # Adjust as needed

train_ds, val_ds, test_ds = prepare_datasets(
    annot_path=annot_path,
    classes_path=classes_path,
    img_path=img_pth,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    val_ratio=0.2,
    test_ratio=0.1
)

# Your existing model setup remains the same
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone_coco")
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="center_xywh",
    backbone=backbone,
    fpn_depth=1,
)

# Compile model
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer,
    classification_loss="binary_crossentropy",
    box_loss="ciou"
)



import wandb
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
wandb.init(
        project="yolo",
        name= f"{'test':<20}|{date.today()}",
        config={
            "epoch": 100
        },
    )
config = wandb.config
    # # Add WandbMetricsLogger to log metrics
wandb_callbacks =WandbMetricsLogger()

# Train model
history = yolo.fit(
    train_ds.take(10),
    validation_data=val_ds,
    epochs=100,
    callbacks=[wandb_callbacks,
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        )
    ]
)