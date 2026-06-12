"""
* callback.py contains callsbacks that are needed for saving the models and predictions during training.

"""
import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from dataloader import single_batch
import wandb
from utils import drawer, targetize


for images, targets in single_batch:
    sample_image, sample_target = images[0], targets[0]

class DisplayCallback(tf.keras.callbacks.Callback):
  def _mu(self):
      pred = self.model.predict(sample_image[tf.newaxis, ...], verbose=0)
      return float(pred[0, 0])  # mu head; pred[0, 1] is log sigma^2

  def on_train_begin(self, logs=None):
      mu = self._mu()
      wandb.log({"Prediction": [wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), 0]), caption="Base truth"),
                                wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), mu]), caption="Compare"),
                                wandb.Image(drawer(array_to_img(sample_image), [0, mu]), caption="Prediction start")]})

  def on_epoch_end(self, epoch, logs=None):
      mu = self._mu()
      wandb.log({"Prediction": [wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), 0]), caption="Base truth"),
                                wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), mu]), caption="Compare"),
                                wandb.Image(drawer(array_to_img(sample_image), [0, mu]), caption=f"Prediction epoch - {epoch}")]})



          