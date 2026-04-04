import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np
import os
import math

root_path = '/users/msayfiddinov/scratch/antibiogo'

train_dir = os.path.join(root_path,"tf_record_xyr/Train")
val_dir = os.path.join(root_path,"tf_record_xyr/Valid")
test_dir = os.path.join(root_path,"tf_record_xyr/Test")

# Hyper-parameters
LEARNING_RATE = 0.0003
EPOCHS = 600
INITIAL_BIAS = 0
MAX_LR=0.003
START_LR=0.0003
END_LR=0.00001
MCDROPOUT_RATE=0.1

# Data-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 128
BATCH_SIZE = 32
IMG_SIZE = 256
DATA_SIZE = 14000
SCALE=tf.math.sqrt(2.0)*(IMG_SIZE-1.0)/4.0

steps_per_epoch = math.ceil(DATA_SIZE / BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = int(total_steps * 0.3) 

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=MAX_LR,   # The Peak LR (target of warmup)
    decay_steps=total_steps,        # Total duration of the schedule
    alpha=END_LR / MAX_LR,          # Minimum LR (fraction of Peak)
    warmup_target=None,             # Defaults to initial_learning_rate
    warmup_steps=warmup_steps       # Enables the Linear Ramp-up
)

def drawer(image: list, tars: list):
  """Given image and targets with base_truth and predictions,
     draw base_truth with green and prediction with red"""
  colors = [(0,255,0), (255, 0, 0)]
  image = image.convert("RGBA")

  true_R = tars[0].numpy()
  w, h = image.size
  x, y = w // 2, h // 2

  # draw the base truth
  r = float(true_R)
  top_left = (x - r, y - r)
  bottom_right = (x + r, y + r)

  # Draw ellipse
  draw = ImageDraw.Draw(image)
  draw.ellipse([top_left, bottom_right], outline=colors[0], width=3)

  if len(tars)> 1:
    r = float(tars[1])
    if r>0.0:
      # Draw ellipse
      top_left = (x - r, y - r)
      bottom_right = (x + r, y + r)
      draw = ImageDraw.Draw(image)
      draw.ellipse([top_left, bottom_right], outline=colors[1], width=3)
  return image

def targetize(pred_target):
  pred_target = pred_target[0]
  pred_target = pred_target.tolist()
  return pred_target
  
