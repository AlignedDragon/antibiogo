import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.experimental import numpy as tnp

# sqrpi = tf.sqrt(tnp.pi)

# def area(targets):
#   x,y,r = targets[..., 0], targets[..., 1], targets[..., 2] 
#   r = sqrpi*r
#   return tf.stack([x, y,2 * r,2 * r], axis=-1)

class GaussianNLL(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mu, sigma = y_pred[..., 0], y_pred[..., 1]
        sigma = tf.maximum(sigma, 1e-6)  # avoid log(0)
        return 0.5 * tf.math.log(2 * tnp.pi * sigma**2) + ((y_true - mu)**2) / (2 * sigma**2)

class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

  def train_step(self, data):
    # print(data)
    image, target = data
    # print("Train step - Image shape:", tf.shape(image))
    # print("Train step - Target shape:", tf.shape(target))
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = self(image, training=True)
        # print("Predictions shape:", predictions.shape)
        # Compute the loss.
        # loss_value = tf.keras.losses.MeanSquaredError()(y_true=area(target), y_pred=area(predictions))
        loss_value = GaussianNLL(y_true = target, y_pred = predictions)
    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
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
