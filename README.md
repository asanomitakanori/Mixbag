# Mixbag: Bag-Level Data Augmentation for Learning from Label Proportions

## Overview of our method
![Illustration](./image/overview.png)
Takanori Asanomi, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise
> Learning from label proportions (LLP) is a promising weakly supervised learning problem. In LLP, a set of instances (bag) has label proportions, but no instance-level labels are given. LLP aims to train an instance-level classifier by using the label proportions of the bag. In this paper, we propose a bag-level data augmentation method for LLP called MixBag, based on the key observation from our preliminary experiments; that the instance-level classification accuracy improves as the number of labeled bags increases even though the total number of instances is fixed. We also propose a confidence interval loss designed based on statistical theory to use the augmented bags effectively.


## Requirements
* PyTorch 
* cuda && cudnn

We strongly recommend using a virtual environment like Anaconda or Docker.
The following is how to build the virtual environment for this code using anaconda.
```
$ pip install -r requirements.py
```

## Dataset

```
./dataset
    ├── train
    │   └── train_imgs                       
    │        ├── sequence001                # Each sequence has 300 images 
    │        ├── sequence002
    │        ├── :
    │        └── sequenceN                       
    ├── val  
    │   └─ val_imgs                          # Same structure of train_imgs
    │        ├── sequence011                # Each sequence has 12 images 
    │        ├── sequence015
    │        ├── :
    │        └── sequenceM  
    └── test
         └── test_imgs                       # Same structure of train_imgs.
              ├── sequence011                # Each sequence has 300 images 
              ├── sequence015
              ├── :
              └── sequenceM  
```