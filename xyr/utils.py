from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np


root_path = '/users/msayfiddinov/scratch/antibiogo'
img_pth = path.join(root_path,"patches")
radii = path.join(root_path, "radii.json")


train_dir = path.join(root_path,"tf_record_xyr/Train")
val_dir = path.join(root_path,"tf_record_xyr/Valid")
test_dir = path.join(root_path,"tf_record_xyr/Test")
orig_train_dir = path.join(root_path,"tf_record_xyr/Original_Train")


tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
initial_bias = 90


# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 128
BATCH_SIZE = 2
LEARNING_RATE = 0.003
# The required image size.


IMG_SIZE = 256
EXPR_BATCHES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 32, 64, 128, 256, 512]
EXPR_FILTERS = [8, 16, 32, 64]
EXPR_WEIGHTS = [0.1, 0.01, 0.001, 0.0001]


def display(display_list:List)->None:
  """
  [true_image,true_mask,predicted_mask] -> display
  """
  plt.figure(figsize=(15, 15))
  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for idx in range(len(display_list)):
    plt.subplot(1, len(display_list), idx+1)
    plt.title(title[idx])
    plt.imshow(array_to_img(display_list[idx]))
    plt.axis('off')
  plt.show()


def drawer(image: list, tars: list):
  # tars is [true_R, (mean_R, std_R)]
  colors = [(0,255,0), (255, 0, 0)]
  image = image.convert("RGBA")

  true_R = tars[0].numpy()
  x, y = 128, 128

  # draw the base truth
  r = true_R
  top_left = (x - r, y - r)
  bottom_right = (x + r, y + r)

  # Draw ellipse
  draw = ImageDraw.Draw(image)
  draw.ellipse([top_left, bottom_right], outline=colors[0], width=3)

  if len(tars)> 1:
    mean_R, std_R = tars[1][0], tars[1][1]
    if mean_R>1 and std_R>1:
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
  
  
# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

