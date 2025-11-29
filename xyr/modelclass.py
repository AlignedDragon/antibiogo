import tensorflow as tf
import numpy as np
from utils import unnormalize


class GaussianNLL(tf.keras.losses.Loss):
    def __init__(self, name="Gaussian_NLL", **kwargs):
       super().__init__(name=name, **kwargs)
       self.half_log2pi = 0.5 * tf.math.log(2 * np.pi)

    def call(self, y_true, y_pred):
        # mu, logsigma are in the range [-1,1]
        mu, logsigma = y_pred[..., 0], y_pred[..., 1]
        """ 
        y_ture is normalized to the range [-1,1] from [0, sqrt(2)*(IMG_SIZE-1)/2]
        logsigma is outputted in [-1,1] for stability of training.
        Useful sigma range is[0.16, sqrt(2)*(IMG_SIZE-1)/2] where 0.16 will give 
        99% certainity of a single pixel neighborhood. Actually, because y_true is scaled 
        with dividing by sqrt(2)*(IMG_SIZE-1)/4, sigma (not logsigma) must be also divided.
        Thus scaled_sigma is [0.16/(sqrt(2)*(IMG_SIZE-1)/4), 2]. So the model should be outputting 
        logsigma in the range [-6.398,0.693] or [-7,1] with IMG_SIZE=256. Because we want stability, we make the model output
        [-1,1], which we scale back to [-7,1] for gaussian calculations
        """
        # scaling values are assumed for IMG_SIZE=256 here and in utils.output_unnormalize
        # logsigma = tf.clip_by_value((logsigma*4.0)-3.0, -7.0, 1.0) # for numerical stability
        # unnormalizing for better results
        y_true, mu, logsigma = unnormalize(y_true), unnormalize(mu), unnormalize(logsigma, sigma=True)
        sigma = tf.math.exp(logsigma)  # avoids sigma=0
        return  self.half_log2pi + logsigma + 0.5*tf.math.square((y_true - mu)/sigma)

    
class CustomModel(tf.keras.Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.nll = GaussianNLL()
    self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
    self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")

  def train_step(self, data):
    image, target = data
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = self(image, training=True)
        loss_value = self.nll(target, predictions)
        
    # Compute gradients and update weights
    trainable_vars = self.trainable_variables
    grads = tape.gradient(loss_value, trainable_vars)
    self.optimizer.apply_gradients(zip(grads, trainable_vars))
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(unnormalize(target), unnormalize(predictions[...,0]))
    self.mse_metric.update_state(unnormalize(target), unnormalize(predictions[...,0]))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = loss_value = self.nll(target, predictions)
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(unnormalize(target), unnormalize(predictions[...,0]))
    self.mse_metric.update_state(unnormalize(target), unnormalize(predictions[...,0]))
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  @property
  def metrics(self):
      return [self.loss_tracker, self.mae_metric, self.mse_metric]
