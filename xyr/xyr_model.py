from utils import IMG_SIZE, INITIAL_BIAS
import tensorflow as tf
from modelclass import CustomModel
from utils import LEARNING_RATE, lr_schedule
from MobileNets import MobileNetV3Small_MCDropout

def weight_init():
    original = tf.keras.applications.MobileNetV3Small(input_shape=[IMG_SIZE, IMG_SIZE, 3], weights="imagenet", include_top=False, include_preprocessing=False)
    modified = MobileNetV3Small_MCDropout(input_shape=[IMG_SIZE, IMG_SIZE, 3],weights=None, include_top=False, include_preprocessing=False)
    for layer in modified.layers:
        if not layer.get_weights():
            continue  # no-parameter layers (activations, dropout, rescaling, input) — nothing to copy
        try:
            weights = original.get_layer(layer.name).get_weights()
            layer.set_weights(weights)
        except (ValueError, KeyError):
            print(f"Skipping (no match in pretrained): {layer.name}")
    return modified


def xyr_model():
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    model = weight_init()
    x = model(inputs)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    outputs = tf.keras.layers.Dense(1, name='output', bias_initializer=tf.keras.initializers.Constant(INITIAL_BIAS))(x)  
    return CustomModel(inputs = inputs, outputs = outputs)


model = xyr_model()
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule))

# print(model.summary())