"""
* dataloader.py for loading and preparing the data during Training, Validating, and Testing.

"""
import tensorflow as tf
from utils import AUTOTUNE, IMG_SIZE, BUFFER_SIZE,BATCH_SIZE, train_dir,val_dir,test_dir,orig_train_dir


def remap_classes(image, target):
    classes = tf.cast(target["classes"], tf.float32)
    target["classes"] = tf.where(classes > 0.0, classes - 1.0, -1.0)
    return image, target

originl_train_dataset = tf.data.Dataset.load(orig_train_dir).map(remap_classes, num_parallel_calls=AUTOTUNE)
# train_dataset = tf.data.Dataset.load(train_dir)
val_ds = tf.data.Dataset.load(val_dir).map(remap_classes, num_parallel_calls=AUTOTUNE)
test_ds = tf.data.Dataset.load(test_dir).map(remap_classes, num_parallel_calls=AUTOTUNE)

single_batch = (
    originl_train_dataset
    .shuffle(BUFFER_SIZE)
    .take(1)
    .cache()
    .padded_batch(
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
                    "classes": -1.0
                }
            )
        )
    .prefetch(buffer_size=AUTOTUNE))
        
orig_train_batches = (
    originl_train_dataset
    .cache()
    .shuffle(BUFFER_SIZE,reshuffle_each_iteration=True)
    .padded_batch(
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
    .prefetch(buffer_size=AUTOTUNE))


vald_batches = val_ds.padded_batch(
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

test_batches = test_ds.padded_batch(
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
