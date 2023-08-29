# Mixbag

## Overview of our method
![Illustration](./image/overview.png)
> Overview of our method. 


## Requirements
* PyTorch 
* cuda && cudnn

We strongly recommend using a virtual environment like Anaconda or Docker.
The following is how to build the virtual environment for this code using anaconda.
```
$ pip install -r env/requirements.py
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