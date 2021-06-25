"""
Main point of contact with other modules

- Start with no augmentation for baseline 

⚒ Improvement & Ideas ⚒
------------------------
- Try UDA (unsupervised data augmentation) by google 
"""


def train_transform_comp(img):
    return img

def valid_transform_comp(img): 
    return img

comp_transforms = {
    'train': train_transform_comp, 
    'valid': valid_transform_comp, 
}