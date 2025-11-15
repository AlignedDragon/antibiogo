
from typing import List
import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
from os import path,getenv
import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np


# root_path = getenv("ROOT_DIR")
root_path = '/mloscratch/sayfiddi/xyr'
img_pth = path.join(root_path,"xyr_data")
train_dir = path.join(root_path,"tf_record/Train")
val_dir = path.join(root_path,"tf_record/Valid")
test_dir = path.join(root_path,"tf_record/Test")
orig_train_dir = path.join(root_path,"tf_record/Original_Train")
corDictDir = '/mloscratch/sayfiddi/xyr/xyr_data/frame_corr.json'

tf_global_seed = 1234
np_seed = 1234
shuffle_data_seed = 12345
initial_bias = 90


# Hyper-parameters
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
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
  # Assuming target is [x, y, r] from model prediction
  colors = [(0,255,0), (255, 0, 0)]
  for i in range(len(tars)):
    # print(tars)
    circle = tars[i]
    # print(circle)
    x, y, r = circle  # center (x, y) and radius (r)

    # Convert to top-left and bottom-right coordinates
    if x>1 and y>1:
      top_left = (x - r, y - r)
      bottom_right = (x + r, y + r)

      # Ensure the coordinates are in the correct order
      top_left = (min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1]))
      bottom_right = (max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1]))

      # Draw ellipse
      draw = ImageDraw.Draw(image)
      draw.ellipse([top_left, bottom_right], outline=colors[i], width=3)
  return image


def targetize(pred_target):
  pred_target = pred_target[0]
  pred_target = pred_target.tolist()
  return pred_target
  


  
# Instantiate an optimizer.
optimAdam = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

