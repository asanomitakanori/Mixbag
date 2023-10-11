# [Mixbag: Bag-Level Data Augmentation for Learning from Label Proportions](https://openaccess.thecvf.com/content/ICCV2023/papers/Asanomi_MixBag_Bag-Level_Data_Augmentation_for_Learning_from_Label_Proportions_ICCV_2023_paper.pdf)

## Overview of our method
![Illustration](./image/overview.png)
Takanori Asanomi, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise
> Learning from label proportions (LLP) is a promising weakly supervised learning problem. In LLP, a set of instances (bag) has label proportions, but no instance-level labels are given. LLP aims to train an instance-level classifier by using the label proportions of the bag. In this paper, we propose a bag-level data augmentation method for LLP called MixBag, based on the key observation from our preliminary experiments; that the instance-level classification accuracy improves as the number of labeled bags increases even though the total number of instances is fixed. We also propose a confidence interval loss designed based on statistical theory to use the augmented bags effectively.


## Requirements
* python >= 3.9
* cuda && cudnn

We strongly recommend using a virtual environment like Anaconda or Docker.
The following is how to build the virtual environment for this code using anaconda.
```
# pytorch install
$ pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.py
```

## Dataset
```
./data
     └── cifar10
          ├── 0
          │   ├── train_bags.npy          # train data (512, 10, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
          │   ├── train_labels.npy        # train class label of each data (512, 10) = (the number of bags, bag size)
          │   ├── train_original_lps.npy  # train label proportions (512, 10) = (the number of bags, class label proportions)
          │   ├── val_bags.npy            # val data (10, 64, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
          │   ├── val_labels.npy          # val class label of each data (10, 64) = (the number of bags, bag size)
          │   └── val_original_lps.npy    # val label proportions(10, 10) = (the number of bags, class label proportions)
          │                
          ├── :
          │
          ├── 4
          │   ├── train_bags.npy          # train data (512, 10, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
          │   ├── train_labels.npy        # train class label of each data (512, 10) = (the number of bags, bag size)
          │   ├── train_original_lps.npy  # train label proportions (512, 10) = (the number of bags, class label proportions)
          │   ├── val_bags.npy            # val data (10, 64, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
          │   ├── val_labels.npy          # val class label of each data (10, 64) = (the number of bags, bag size)
          │   └── val_original_lps.npy    # val label proportions(10, 10) = (the number of bags, class label proportions)
          │
          ├── test_data.npy               # (10000, 32, 32, 3) = (the number of data, height, width, channel)
          │
          └── test_label.npy              # (10000,) = (the number of data labels)
               
```

## Training & Test
If you want to train network, please run following command.
5 fold cross-validation is implemented and Test is automatically done in our code.
```
$ python run.py
```

## Arguments
You can set up any parameters at arguments.py

## Citation
If you find MixBag useful in your work, please cite our paper:
```none
@inproceedings{asanomi2023mixbag,
  title={MixBag: Bag-Level Data Augmentation for Learning from Label Proportions},
  author={Asanomi, Takanori and Matsuo, Shinnosuke and Suehiro, Daiki and Bise, Ryoma},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={16570--16579},
  year={2023}
}
```

## Author
👤 **Takanori Asanomi**
* Github: [@takanoriasanomi](https://github.com/asanomitakanori)
* Contact: asataka1998@gmail.com