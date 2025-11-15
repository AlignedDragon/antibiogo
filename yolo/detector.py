import tensorflow as tf
import keras_cv
from utils import class_mapping
# from dataloader import originl_train_dataset
from utils import LEARNING_RATE, GLOBAL_CLIPNORM, IMG_SIZE
tf.config.run_functions_eagerly(True)

backbone = keras_cv.models.YOLOV8Backbone.from_preset(
    "yolo_v8_xs_backbone_coco"  # We will use yolov8 small backbone with coco weights
)
inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
yolo = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="center_xywh",
    backbone=backbone,
    fpn_depth=1,
)
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    global_clipnorm=GLOBAL_CLIPNORM,
)

yolo.compile(
    optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
)
# print(yolo.input_shape)
# print(yolo.inputs)
# print(yolo.summary())