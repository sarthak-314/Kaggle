import pytorch_lightning as pl 
from omegaconf import OmegaConf
import hydra
import torch 

def lr_find(trainer, model, num_steps):
    """
    Find a good learning rate for the model
    num_steps: number of steps to run between min lr (1e-8), max lr(1)

    """
    lr_finder = trainer.tuner.lr_find(model, num_training=num_steps) # run learning rate finder
    print(lr_finder.results) # see the results
    lr_finder.plot(suggest=True).show() # plot the results
    new_lr = lr_finder.suggestion() # get a suggestion
    print('new lr:', new_lr)
    return new_lr    


def get_loggers(loggers_config): 
    loggers = []
    for _, logger_conf in loggers_config.items():
        print(f'Initializing logger <{logger_conf._target_}>')
        loggers.append(hydra.utils.instantiate(logger_conf))
    return loggers
    
def fast_trainer(fast_dev_run=1, overfit_batches=0.01, max_time={'minutes': '5'}, limit_train_batches=1): 
    """
    Build a trainer for debugging

    Args:
        fast_dev_run (bool / int): run n number of train, valid and test batches to find any bugs (unit test)
        overfit_batches (float / int): Use 1% of training data or 10 batches
        max_time (dict): maximum amount of time to give to this
        limit_train_batches (float / int): run the epoch quicky. Useful for something that happens at the end of epoch
    """
    return pl.Trainer(fast_dev_run=fast_dev_run, overfit_batches=overfit_batches, max_time=max_time, limit_train_batches=limit_train_batches)


def get_callbacks(callbacks_config): 
    callbacks = []
    for _, cb_conf in callbacks_config.items():
        print(f"Instantiating callback <{cb_conf._target_}>")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks



def build_trainer(trainer_config, callbacks, loggers, resume_from_checkpoint=None): 
    trainer_kwargs = {
        **trainer_config, 
        'resume_from_checkpoint': resume_from_checkpoint,  
        'callbacks': callbacks, 
        'logger': loggers,
    }
    
    if torch.cuda.is_available():
        trainer_kwargs['gpus'] = -1 # train on all gpus
        trainer_kwargs['precision'] = 16 # 2 x speed up
    
    return pl.Trainer(**trainer_kwargs)


if __name__ == '__main__':
    print('working?')

"""
TODO: Standarize the function naming for reading yalm config file
"""