from omegaconf import OmegaConf
from pathlib import Path
import torch
import os


def hardware_hookup(): 
    hardware = {}
    if 'TPU_NAME' in os.environ: 
        processor = 'TPU'
        hardware['tpu_cores'] = 8
        print('Using TPU')
    elif torch.cuda.is_available(): 
        processor = 'GPU'
        gpu_name = torch.cuda.get_device_name()
        hardware['gpus'] = 1
        hardware['gpu_name'] = gpu_name
    else: 
        processor = 'CPU'
    hardware['processor'] = processor
    return hardware


# paths dict
paths = {}
paths['work'] = Path(os.getcwd())
paths['data'] = paths['work'] / 'data'

# comp config dict
comp = OmegaConf.load('src/config/comp.yaml')

# hardware 
hardware = hardware_hookup()

if __name__ == '__main__':
    import src
    print(dir(src))
    print('paths: ', paths)
    print('comp:', comp)
    print('hardware: ', hardware)



