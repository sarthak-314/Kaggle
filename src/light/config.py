from pathlib import Path
import src.light
from omegaconf import OmegaConf

LIGHT_PACKAGE_PATH = Path(src.light.__path__[0])
LIGHT_CONFIG_PATH = LIGHT_PACKAGE_PATH / 'config'

CALLBACKS_CONFIG = OmegaConf.load(LIGHT_CONFIG_PATH / 'callbacks.yaml')
LOGGERS_CONFIG = OmegaConf.load(LIGHT_CONFIG_PATH / 'loggers.yaml')
TRAINER_CONFIG = OmegaConf.load(LIGHT_CONFIG_PATH / 'trainer.yaml')


if __name__=='__main__':
    print('CALLBACKS_CONFIG: ', CALLBACKS_CONFIG)
    print('LOGGERS_CONFIG: ', LOGGERS_CONFIG)
    print('TRAINER_CONFIG: ', TRAINER_CONFIG)
