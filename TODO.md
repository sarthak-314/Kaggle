function lazygit() {
    git add .
    git commit -a -m "$1"
    git push
}

conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
https://github.com/albumentations-team/autoalbument-benchmarks/tree/main/conf for a good hydra architecture

just installed pytorch for now coz it's slow

- go though hydra, lightning callbacks & trainer very carefully 
- use OmegaConf and access config through dot notation OmegaConf.load('config.yaml')
- load the aug config and run config in the jupyter notebook
- https://github.com/rafaelpadilla/Object-Detection-Metrics

CREATIVE MODE
- think about the architecture

MODERATELY ZONED IN MODE
- go through hydra docs, lightning callbacks & trainer
- auto augment paper and related


What can be changed

LIGHTNING
datamodule 
    - features, transformations 

model 
    - backbone, final_layer, metric, optimizer, scheduler

run 
    - trainer, overfitter, freeze and shot



FLOW OF DATA 
running an experiment needs trainer and a model

To breakup the dataflow, make small modules and control them with jupyter notebook
Try this strategy - if it can be broken up - then break it up

QUESTIONS 
- which augmentations to apply and how to apply 
  - augmentations for class imbalance
  - aug mix, cut mix stuff
- best way to structure and run experiments
- how to scale cheating 



IDEAS 
- class for bounding box?
- 


"""
----------------------------------
⚒ Improvements, Ideas & Threads ⚒
----------------------------------

Project Level 
-------------
- Integrate a robust config system in the project 
    - master hydra and take a look at other configutation systems

Torch 
-----
- Master PyTorch Lightning and integrate all the goodness
    - Take a look at flash, bolts and similar repos

-----------------
🚴🏻‍♀️ Next Cycle 🚴🏻‍♀️
-----------------
🚀 Changes to make in the next cycle of the project
- Integrate tensorflow


----------
🔮 Future 
----------
- Master fast ai and integrate it
- How to integrate external data and external models 
- see black python formatting package
- Look at https://github.com/TezRomacH/python-package-template to level up python structure
"""

TODOS: 
- Find how to best structure a python pakcage and a project and jupyternotebooks with i