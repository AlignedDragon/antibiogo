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

def unnormalize(sample, sigma=False):
  """Get the mean from [-1,1] range back to [0, sqrt(2)*(IMG_SIZE-1)/2]
     and the logsigma from [-1,1] to [-7,1] to log[0.16, sqrt(2)*(IMG_SIZE-1)/2]"""
  if sigma==True:
    # unnormalizing logsigma
    sample = tf.clip_by_value((sample*4.0)-3.0, -7.0, 1.0) + tf.math.log(SCALE)
    return sample
  sample = (sample+1.0)*SCALE
  return sample

def drawer(image: list, tars: list):
  """Given image and targets with base_truth and predictions,
     draw base_truth with green and prediction with red"""
  colors = [(0,255,0), (255, 0, 0)]
  image = image.convert("RGBA")

  true_R = tars[0].numpy()
  w, h = image.size
  x, y = w // 2, h // 2

  # draw the base truth
  r = float(unnormalize(true_R))
  top_left = (x - r, y - r)
  bottom_right = (x + r, y + r)

  # Draw ellipse
  draw = ImageDraw.Draw(image)
  draw.ellipse([top_left, bottom_right], outline=colors[0], width=3)

  if len(tars)> 1:
    # tars is [true_R, (mean_R, std_R)]
    mean_R, logsigma = float(unnormalize(tars[1][0])), float(unnormalize(tars[1][1], sigma=True))
    std_R = np.exp(logsigma)
    if mean_R>0 and std_R>0:
      r = mean_R

      # drawing stadard deviation
      overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
      od = ImageDraw.Draw(overlay)

      outer_r = r + std_R
      inner_r = max(0, r - std_R)  # avoid negative

      bbox_outer = (x - outer_r, y - outer_r, x + outer_r, y + outer_r)
      bbox_inner = (x - inner_r, y - inner_r, x + inner_r, y + inner_r)

      # draw filled outer ellipse with alpha, then punch hole by drawing transparent inner ellipse
      od.ellipse(bbox_outer, fill=(255, 0, 0, 50))   # red with alpha (0-255)
      od.ellipse(bbox_inner, fill=(0, 0, 0, 0))       # makes the hole transparent

      # composite overlay onto original image
      image = Image.alpha_composite(image, overlay)

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
  
