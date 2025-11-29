import tensorflow as tf

train_data = tf.data.Dataset.load("$HOME/scratch/antibiogo/tf_record_xyr_target_unnormalized/Train")

for i, d in enumerate(train_data):
    print(d)
    if i>1:
        break

# 