import tensorflow as tf
tf.config.run_functions_eagerly(True)
import numpy as np


# class gaussian_nll(tf.keras.losses.Loss):
#     """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
#     This implementation implies diagonal covariance matrix.
    
#     Parameters
#     ----------
#     ytrue: tf.tensor of shape [n_samples, n_dims]
#         ground truth values
#     ypreds: tf.tensor of shape [n_samples, n_dims*2]
#         predicted mu and logsigma values (e.g. by your neural network)
        
#     Returns
#     -------
#     neg_log_likelihood: float
#         negative loglikelihood averaged over samples
        
#     This loss can then be used as a target loss for any keras model, e.g.:
#         model.compile(loss=gaussian_nll, optimizer='Adam') 
    
#     """
#     def call(self, ytrue, ypreds):
#       n_dims = int(int(ypreds.shape[1])/2)
#       mu = ypreds[:, 0:n_dims]
#       logsigma = ypreds[:, n_dims:]
      
#       mse = -0.5*tf.math.reduce_sum(tf.math.square((ytrue-mu)/tf.math.exp(logsigma)),axis=1)
#       sigma_trace = -tf.math.reduce_sum(logsigma, axis=1)
#       log2pi = -0.5*n_dims*np.log(2*np.pi)
      
#       log_likelihood = mse+sigma_trace+log2pi

#       return tf.math.reduce_mean(-log_likelihood)


class GaussianNLL(tf.keras.losses.Loss):
    def __init__(self, name="Gaussian_NLL", **kwargs):
       super().__init__(name=name, **kwargs)
       self.half_log2pi = 0.5 * tf.math.log(2 * np.pi)

    def call(self, y_true, y_pred):
        mu, logsigma = y_pred[..., 0], y_pred[..., 1]
        logsigma = tf.clip_by_value(logsigma, -2.0, 5.0) # for numerical stability
        sigma = tf.math.exp(logsigma)  # avoid log(0)
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
    self.mae_metric.update_state(target, predictions[...,0])
    self.mse_metric.update_state(target, predictions[...,0])
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  def test_step(self, data):
    # Unpack the data.
    image, target = data
    # Compute predictions
    predictions = self(image, training=False)
    # Compute the loss.
    loss_value = tf.keras.losses.MeanSquaredError()(y_true=target, y_pred=predictions[...,0])
    # Update metrics
    self.loss_tracker.update_state(loss_value)
    self.mae_metric.update_state(target, predictions[...,0])
    self.mse_metric.update_state(target, predictions[...,0])
    # Return a dict mapping metric names to current value
    return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mse": self.mse_metric.result()}

  @property
  def metrics(self):
      return [self.loss_tracker, self.mae_metric, self.mse_metric]
