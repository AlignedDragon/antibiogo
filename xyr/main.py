from dataloader import train_batches, vald_batches, single_batch
from xyr_model import model
from callback import DisplayCallback # ,EarlyStopping_callback, savemodel_callback
import wandb, os
# Use wandb-core
wandb.require("core")
from wandb.integration.keras import WandbMetricsLogger
from datetime import date
from utils import EXPR_BATCHES, EXPR_FILTERS, EXPR_WEIGHTS, root_path
import tensorflow as tf



if __name__=="__main__":
    dump_path = f"{root_path}/ExperimentModels"
    prefix = os.getenv("FOLDER_PREFIX")
    assert prefix is not None, "FOLDER_PREFIX env var is missing"
    os.makedirs(f"{dump_path}/{prefix}", exist_ok=True)

    EPOCHS = 600
    wandb.init(
        dir=f"{root_path}/logs/wandb",
        project="yolo",
        name= f"{prefix}_xyr |{date.today()}",
        config={
            "epoch": EPOCHS
        },
    )
    config = wandb.config
    wandb_callbacks =WandbMetricsLogger()
    ckpt_path = f'{dump_path}/{prefix}/xyr_best.keras'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,     # Only overwrite if the model improves
        monitor='val_loss',      # Watch validation loss
        mode='min'               # We want loss to be minimized
    )

    try:
        model.load_weights(f'/users/msayfiddinov/scratch/antibiogo/ExperimentModels/{prefix}/xyr_best.keras')
        print("SUCCESS: Weights loaded.")
    except Exception as e:
        print(f"FAILURE: Could not load weights. Error: {e}")
    model_history = model.fit(train_batches,
                              epochs= config.epoch,
                              verbose = 0,
                              validation_data=vald_batches,
                              callbacks=[DisplayCallback(),wandb_callbacks, checkpoint_cb]  # , +EarlyStopping_callback
                              )
    # ! WARNING model names may overlap thus loosing trained paramaters
    model.save(f'{dump_path}/{prefix}/xyr.keras')
    wandb.finish()
