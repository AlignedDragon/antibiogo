from utils import IMG_SIZE, INITIAL_BIAS, INITIAL_LOG_SIGMA2
import tensorflow as tf
from modelclass import CustomModel
from utils import lr_schedule
from MobileNets import MobileNetV3Small_MCDropout

def weight_init():
    original = tf.keras.applications.MobileNetV3Small(input_shape=[IMG_SIZE, IMG_SIZE, 3], weights="imagenet", include_top=False, include_preprocessing=False)
    modified = MobileNetV3Small_MCDropout(input_shape=[IMG_SIZE, IMG_SIZE, 3],weights=None, include_top=False, include_preprocessing=False)
    for layer in modified.layers:
        if not layer.get_weights():
            continue
        try:
            weights = original.get_layer(layer.name).get_weights()
            layer.set_weights(weights)
        except (ValueError, KeyError):
            print(f"Skipping (no match in pretrained): {layer.name}")
    return modified


def xyr_model():
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    backbone = weight_init()
    feats = backbone(inputs)
    feats = tf.keras.layers.GlobalMaxPooling2D()(feats)
    # Zero kernel init keeps every sample at exactly the bias at step 0 -
    # no per-sample spread on log_sigma^2 (which would otherwise blow up the
    # NLL through exp(-log_sigma^2)).
    mu = tf.keras.layers.Dense(
        1, name="mu",
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(INITIAL_BIAS),
    )(feats)
    log_sigma2 = tf.keras.layers.Dense(
        1, name="log_sigma2",
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(INITIAL_LOG_SIGMA2),
    )(feats)
    outputs = tf.keras.layers.Concatenate(axis=-1, name="mu_logvar")([mu, log_sigma2])
    return CustomModel(inputs=inputs, outputs=outputs)


model = xyr_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, global_clipnorm=1.0))
