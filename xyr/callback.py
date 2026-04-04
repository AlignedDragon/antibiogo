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

# Callbacks 
#EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(patience=3)
checkpoint_filepath = '/users/msayfiddinov/scratch/antibiogo_data/SavedModels/checkpoint.model.keras'
savemodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                        monitor='val_loss',
                                                        save_best_only=True,
                                                        mode='min'
                                                        )


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs=None):
      wandb.log({"Prediction": [wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), 0]), caption="Base truth"),
                                wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), self.model.predict(sample_image[tf.newaxis, ...])[0,0]]),caption="Compare"),
                                wandb.Image(drawer(array_to_img(sample_image), [0, self.model.predict(sample_image[tf.newaxis, ...])[0,0]]),caption=f"Prediction start")            ]})

  def on_epoch_end(self, epoch, logs=None):
      wandb.log({"Prediction": [wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), 0]), caption="Base truth"),
                                wandb.Image(drawer(array_to_img(sample_image), [targetize(sample_target), self.model.predict(sample_image[tf.newaxis, ...])[0,0]]),caption="Compare"),
                                wandb.Image(drawer(array_to_img(sample_image), [0, self.model.predict(sample_image[tf.newaxis, ...])[0,0]]),caption=f"Prediction epoch - {epoch}")  ]})



          