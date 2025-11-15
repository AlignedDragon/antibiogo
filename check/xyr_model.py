from ssl import _create_default_https_context, _create_unverified_context
from xyr_utils import IMG_SIZE, OUTPUT_CLASSES, initial_bias
import tensorflow as tf
from xyr_modelclass import CustomModel


def xyr_model():
    # _create_default_https_context = _create_unverified_context
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    model = tf.keras.applications.MobileNetV3Small(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)
    x = model(inputs)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    outputs = tf.keras.layers.Dense(3, name='output')(x)
    return CustomModel(inputs=inputs, outputs=outputs)


model = xyr_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003))
