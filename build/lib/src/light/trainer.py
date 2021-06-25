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


def trainer(trainer_config, callbacks, ):
    default_root_dir = 'epic'
    fast_dev_run=True # run one batch 

    mode = 'debug', 'overfit', 'finetune', 'train'    


def loggers(): 
    
    # default logger used by trainer
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        version=1,
        name='lightning_logs'
    )
    Trainer(logger=logger)