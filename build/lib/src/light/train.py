from typing import List, Optional
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.utils import utils


log = utils.get_logger(__name__)
def train(config): 
    """
    Contains the training pipeline for torch 
    Instantiate all the PyTorch Lightning objects from config
    :param config (DictConfig): configuration composed by Hydra.
    :returns Optional[float]: validation metric score for hyperparameter optimization.
    """
    # init datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # init Lightning model 
    log.info(f"Instantiating model <{config.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # init Lightning callbacks
    callbacks: List[pl.Callback] = []
    for _, cb_conf in config["callbacks"].items():
        if "_target_" in cb_conf:
            log.info(f"Using callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))


def train(config: DictConfig) -> Optional[float]:
    """
    Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule

    # Init Lightning model

    
    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set after training
    if not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]
