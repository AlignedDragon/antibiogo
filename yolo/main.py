from dataloader import orig_train_batches, vald_batches, single_batch
from detector import yolo as model
import keras_cv
from callback import DisplayCallback  # ,EarlyStopping_callback, savemodel_callback
import wandb, os
from utils import root_path
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
import tensorflow as tf
tf.config.run_functions_eagerly(True)


if __name__=="__main__":
    dump_path = f"{root_path}/ExperimentModels"
    prefix = os.getenv("FOLDER_PREFIX")
    assert prefix is not None, "FOLDER_PREFIX env var is missing"
    os.makedirs(f"{dump_path}/{prefix}", exist_ok=True)

    EPOCHS = 200
    wandb.init(
        dir=f"{root_path}/logs/wandb",
        project="yolo",
        name= f"{prefix}_yolo |{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    wandb_callbacks =WandbMetricsLogger()
    ckpt_path = f'{dump_path}/{prefix}/yolo_best.keras'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,     # Only overwrite if the model improves
        monitor='val_loss',      # Watch validation loss
        mode='min'               # We want loss to be minimized
    )

    # model.load_weights('SavedModels/test.keras')
    model_history = model.fit(orig_train_batches,
                              epochs=config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[wandb_callbacks, DisplayCallback(), checkpoint_cb,
                                         keras_cv.callbacks.PyCOCOCallback(vald_batches, bounding_box_format="center_xywh")]) #,savemodel_callback,DisplayCallback()]  #+EarlyStopping_callback
    
    model.save(f'{dump_path}/{prefix}/yolo.keras')
    wandb.finish()


