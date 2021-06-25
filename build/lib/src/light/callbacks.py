from omegaconf import OmegaConf
import src.comp.utils
import hydra

def get_callbacks(callbacks_config): 
    callbacks = []
    for _, cb_conf in callbacks_config.items():
        print(f"Instantiating callback <{cb_conf._target_}>")
        callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks

callbacks_config_path = src.comp.utils.SRC_PACKAGE_PATH / 'light' / 'config' / 'callbacks.yaml'
CALLBACKS_CONFIG = OmegaConf.load(callbacks_config_path)

if __name__ == '__main__':
    get_callbacks(CALLBACKS_CONFIG)
