"""
⚒ Improvements, Ideas & Threads ⚒
----------------------------------
-> go through callbacks docs fully once again
-> how to seprate important callbacks config (like freeze_backbone_at from stupid config like verbose)
# 1. Go through backbonefinetuning docs once again
    - why is backbone lr > model in docs
# 2. Look at basefinetuning callback to implement your algos
# 3. 

-> go through callbacks and trainer arguments
"""
import pytorch_lightning as pl

def get_trainer_callbacks(config): 
    callbacks = []
    backbone_finetuning = pl.callbacks.BackboneFinetuning(
        unfreeze_backbone_at_epoch=config['unfreeze_backbone_at_epoch'], 
        verbose=True, 
    )
    
    early_stopping = pl.callbacks.EarlyStopping(
        monitor='val/acc', 
        patience=
    )
    
    
    if config['use_backbone_finetuning']: 
        callbacks.append(backbone_finetuning)
    





print('next version of callbacks coming in next cycle soon')



    log.section('Model Callbacks', level=3)
    log.info(f'Using backbone finetuning, unfreezing it at {hp.backbone.unfreeze_at_epoch} epoch with initial backbone lr: lr*{hp.backbone.initial_ratio_lr} (hopefully). Will divide the lr by {hp.backbone.initial_denom_lr} when unfreezing')

    early_stopping = pl.callbacks.EarlyStopping(
        monitor=hp.monitor, 
        patience=hp.early_stop_patience, 
        verbose=True, 
        mode=hp.mode, 
    )
    log.info(f'Using early stopping callback monitoring {hp.monitor} ({hp.mode} mode) with {hp.early_stop_patience} patience')
    log.imp_info('Pro Tip: Use gradient accumulation with very large image / data')

    logging_interval = 'epoch'
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval=logging_interval)
    log.info(f'Monitoring learning rate every {logging_interval}')


    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath = hp.model_folder_name, 
        filename = hp.checkpoint_path, 
        verbose = True, 
        save_last = True, 
        mode = hp.mode, 
    )
    log.info(f'Saving model checkpoints at {hp.checkpoint_path} in {hp.model_folder_name} folder')
    log.info('Pro Tip: Look for last.ckpt file in the folder to resume training')
    log.info('Pro Tip: (Skipped) Use model pruning to make the model smaller and faster. Maybe try it in ensembling')

    log.info('(Skipped) Quantization allows speeding up inference and decreasing memory requirements by performing computations and storing tensors at lower bitwidths (such as INT8 or FLOAT16) than floating point precision')

    print_table = pl_bolts.callbacks.printing.PrintTableMetricsCallback()
    log.info('PrintTableMetric prints a table with the metrics in columns on every epoch end')
    verification = pl_bolts.callbacks.BatchGradientVerificationCallback()


    log.red('Check out TrainingDataMonitor and ModuleDataMonitor')

    log.info('Check out https://lightning-bolts.readthedocs.io/en/latest/self_supervised_callbacks.html for self supervised callbacks')

    callbacks = [backbone_finetuning, early_stopping, lr_monitor, model_checkpoint, print_table, verification]
    return callbacks