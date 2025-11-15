from dataloader import orig_train_batches, vald_batches, single_batch
from detector import yolo as model
import keras_cv
from callback import DisplayCallback  # ,EarlyStopping_callback, savemodel_callback
import wandb
# from modelclass import model
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
import tensorflow as tf
tf.config.run_functions_eagerly(True)

if __name__=="__main__":
    EPOCHS = 500
    # Launch an experiment
    wandb.init(
        project="yolo",
        name= f"{'test':<10}|{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    # # Add WandbMetricsLogger to log metrics
    wandb_callbacks =WandbMetricsLogger()
    # model.load_weights('SavedModels/test.h5')
    model_history = model.fit(orig_train_batches,
                              epochs=config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[wandb_callbacks, DisplayCallback(), keras_cv.callbacks.PyCOCOCallback(vald_batches, bounding_box_format="center_xywh")]#,savemodel_callback,DisplayCallback()]  #+EarlyStopping_callback
                              )
    model.save('SavedModels/test.h5')
    # Mark the run as finished
    wandb.finish()


