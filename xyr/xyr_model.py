from utils import IMG_SIZE, initial_bias
import tensorflow as tf
from modelclass import CustomModel
from utils import LEARNING_RATE

# def xyr_model():
#     """
#     Total params: 45,530 (177.85 KB)   (alpha=0.10)
#     Total params: 132,434 (517.32 KB)  (alpha=0.30)
#     """
#     inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])

#     model = tf.keras.applications.MobileNetV3Small(input_shape=[IMG_SIZE, IMG_SIZE, 3], include_top=False, alpha=0.1, weights=None)
#     x = model(inputs)
#     x = tf.keras.layers.GlobalMaxPooling2D()(x)
#     # 0th output is mu, 1st output is sigma
#     outputs = tf.keras.layers.Dense(2, name='output', bias_initializer=tf.keras.initializers.Constant(initial_bias))(x)  
#     return CustomModel(inputs = inputs, outputs = outputs)

def xyr_model():
    """
    Total params: 23,714 (92.63 KB)
    """
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
    
    # Simple feature extraction without the bloat
    x = tf.keras.layers.Conv2D(16, (3, 3), strides=2, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    outputs = tf.keras.layers.Dense(2, name='output', bias_initializer=tf.keras.initializers.Constant(initial_bias))(x)
    return CustomModel(inputs=inputs, outputs=outputs)


model = xyr_model()
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))

# print(model.summary())