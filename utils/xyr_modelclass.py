"""
* modelclass.py contains The modified training and testing steps of tf.keras.Model class.
* we are only interested on specific metrics, and we focused mainly on them.

"""
import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.experimental import numpy as tnp
import keras_cv
from tensorflow.experimental import numpy as tnp
sqrtpi = tf.sqrt(tnp.pi)

def area(targets):
    x,y,r = targets[..., 0], targets[..., 1], targets[..., 2]
    return tf.stack([x, y,sqrtpi*r], axis=-1)

def converter(targets):
    x,y,r = targets[..., 0], targets[..., 1], targets[..., 2]
    return tf.stack([x, y,2 * r,2 * r], axis=-1)


class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

  def train_step(self, data):
    image, target = data
    with tf.GradientTape() as tape:

        predictions = self(image, training=True)
        # IOU Loss
        # loss_value = keras_cv.losses.IoULoss(bounding_box_format = 'center_xywh')(y_true = converter(target), y_pred = converter(predictions))
        # Improved MSE loss
        loss_value = tf.keras.losses.MeanSquaredError()(y_true=area(target), y_pred=area(predictions))

    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))

    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(target, predictions)
    self.mse_metric.update_state(target, predictions)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = tf.keras.losses.MeanSquaredError()(y_true=target, y_pred=predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(target, predictions)
    self.mse_metric.update_state(target, predictions)
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  @property
  def metrics(self):
      return [self.loss_tracker, self.mae_metric, self.mse_metric]

