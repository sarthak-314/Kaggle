"""
USE: 
run the code in the kaggle notebook to get results fast


NOTES: 
- uses faster autoaugment
- GAN: generator applies augmentationa and discriminator must tell if the image is augmented 
You want to produce images similar to the original images 
- internally uses torch, but you can use your model too

STEPS: 
- define basic autoaugment using only dataset on Kaggle GPUs
- input your model after you built a model to search for policies


TODO:
- read the faster autoaugment paper 
"""

