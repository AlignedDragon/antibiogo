# why ssl contexts are needed?

from ssl import _create_default_https_context, _create_unverified_context
from utils import IMG_SIZE, initial_bias
import tensorflow as tf
from modelclass import CustomModel

def xyr_model():
    _create_default_https_context = _create_unverified_context
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

    model = tf.keras.applications.EfficientNetV2B0(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False)
    x = model(inputs)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    # 0th output is mu, 1st output is sigma
    outputs = tf.keras.layers.Dense(2, name='output', bias_initializer=tf.keras.initializers.Constant(90))(x)  
    return CustomModel(inputs = inputs, outputs = outputs)


model = xyr_model()
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0003))

print(model.summary())