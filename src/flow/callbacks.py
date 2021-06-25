"""
Callbacks for training keras models 
TODO:
- a lot of work to be done in here
- go through all the callbacks one more time throughly
- make this dict like config editable on run.ipynb
- standardize format for callbacks for tensorfloe and for torch
"""
import keras
import tensorflow_addons as tfa
from wandb.keras import WandbCallback

# Important & Input
MAX_TRAIN_HOURS = 8

# Common Config
MONITOR = 'val_acc'
MODE = 'max'
VERBOSE = 1 # can be 2

# Enter yourselves
CHECKPOINT_PATH = ''
EARLY_STOP_PATIENCE = 5
USE_WANDB = True 
TENSORBOARD_LOG_DIR = ''

common_kwargs = {
    'monitor': MONITOR, 
    'mode': MODE, 
    'verbose': VERBOSE, 
}


def get_callbacks(
    model, 
    max_train_hours=MAX_TRAIN_HOURS, 
    use_wandb=USE_WANDB, 
    tensorboard_log_dir=TENSORBOARD_LOG_DIR, 
    checkpoint_path=CHECKPOINT_PATH, 
    early_stop_patience=EARLY_STOP_PATIENCE,  
):
    """
    Model Checkpoint 
    something about that 
    """
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        save_weights_only=False, 
        **common_kwargs,   
    )

    early_stopping = keras.callbacks.EarlyStopping(
        patience=early_stop_patience,
        restore_best_weights=True,
        **common_kwargs, 
    )
            
    reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
        factor=0.2,
        patience=5,
        min_delta=0.0001,
        min_lr=0,
        **common_kwargs, 
    )
    
    time_stopping = tfa.callbacks.TimeStopping(seconds=max_train_hours*3600)
    
    tqdm_bar = tfa.callbacks.TQDMProgressBar()
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(tensorboard_log_dir))
    
    callbacks = [model_checkpoint, early_stopping, terminate_on_nan, reduce_lr_on_plateau, time_stopping, tqdm_bar, tensorboard_callback]
        
    if use_wandb: 
        wandb_callback = WandbCallback()
        callbacks.append(wandb_callback)

    return keras.callbacks.CallbackList(
        callbacks, 
        add_progbar = True, 
        model = model,
        add_history=True, 
    )
        
        
if __name__ == '__main__':
    model = keras.Sequential([keras.layers.Dense(4)])
    get_callbacks(model)    