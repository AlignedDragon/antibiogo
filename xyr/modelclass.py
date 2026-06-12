import tensorflow as tf

# Lower bound on log_sigma^2 only — prevents inv_var from blowing up if the
# model becomes overconfident. We do NOT upper-clip: tf.clip_by_value zeroes
# the gradient outside its range, which would freeze any sample that ever
# crosses the upper bound and stall learning of mu (since mu's gradient is
# weighted by exp(-log_sigma^2)).
LOG_SIGMA2_MIN = -8.0


def gaussian_nll(target, mu, log_sigma2):
    """Mean Gaussian negative log-likelihood (constant 0.5*log(2*pi) dropped)."""
    log_sigma2 = tf.maximum(log_sigma2, LOG_SIGMA2_MIN)
    inv_var = tf.exp(-log_sigma2)
    sq_err = tf.square(target - mu)
    return tf.reduce_mean(0.5 * inv_var * sq_err + 0.5 * log_sigma2)


def split_pred(predictions):
    """Split (B, 2) -> (mu (B, 1), log_sigma2 (B, 1))."""
    mu = predictions[..., 0:1]
    log_sigma2 = predictions[..., 1:2]
    return mu, log_sigma2


class CustomModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.sigma_tracker = tf.keras.metrics.Mean(name="sigma_alea")

    def _step(self, image, target, training):
        target = tf.reshape(tf.cast(target, tf.float32), (-1, 1))
        predictions = self(image, training=training)
        mu, log_sigma2 = split_pred(predictions)
        loss_value = gaussian_nll(target, mu, log_sigma2)
        self.loss_tracker.update_state(loss_value)
        self.mae_metric.update_state(target, mu)
        self.mse_metric.update_state(target, mu)
        # Track average aleatoric sigma the model is predicting.
        sigma = tf.exp(0.5 * tf.maximum(log_sigma2, LOG_SIGMA2_MIN))
        self.sigma_tracker.update_state(sigma)
        return loss_value

    def train_step(self, data):
        image, target = data
        with tf.GradientTape() as tape:
            loss_value = self._step(image, target, training=True)
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
            "mse": self.mse_metric.result(),
            "sigma_alea": self.sigma_tracker.result(),
        }

    def test_step(self, data):
        image, target = data
        self._step(image, target, training=False)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
            "mse": self.mse_metric.result(),
            "sigma_alea": self.sigma_tracker.result(),
        }

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric, self.mse_metric, self.sigma_tracker]
