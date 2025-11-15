"""
* callback.py contains callsbacks that are needed for saving the models and predictions during training.

"""
import tensorflow as tf
from tensorflow.keras.utils import array_to_img
from dataloader import single_batch
import wandb
from utils import drawer, targetize


for images, target in single_batch:
    sample_image, sample_target = images[0], target['boxes'][0]


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()  # Initialize model attribute

    # def set_model(self, model):
    #     super().set_model(model)  # Call parent's set_model method
    #     self.model = model  # Store the model instance

    def on_train_begin(self, logs=None):
        if self.model is not None:
            wandb.log({"Prediction": [
                wandb.Image(drawer(array_to_img(sample_image), [sample_target]), caption="Base truth"),
                wandb.Image(drawer(array_to_img(sample_image), [sample_target, self.model.predict(sample_image[tf.newaxis, ...])['boxes'][0]]), caption="Compare"),
                wandb.Image(drawer(array_to_img(sample_image), [[[0,0,0,0]], self.model.predict(sample_image[tf.newaxis, ...])['boxes'][0]]), caption="Prediction start")
            ]})

    def on_epoch_end(self, epoch, logs=None):
        if self.model is not None:
            wandb.log({"Prediction": [
                wandb.Image(drawer(array_to_img(sample_image), [sample_target]), caption="Base truth"),
                wandb.Image(drawer(array_to_img(sample_image), [sample_target, self.model.predict(sample_image[tf.newaxis, ...])['boxes'][0]]), caption="Compare"),
                wandb.Image(drawer(array_to_img(sample_image), [[[0,0,0,0]], self.model.predict(sample_image[tf.newaxis, ...])['boxes'][0]]), caption=f"Prediction epoch - {epoch}")
            ]})      