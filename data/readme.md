## Dataset
The `pathmnist` dataset is already set as exmaple.

You can create dataset by running following code. Dataset is stored in `./data` directory.
```
$ python create_dataset.py
```

The Data structure of `./data` directory is written as belows, after running `python create_dataset.py`.
```
./data
     └── cifar10  # dataset name
     │    ├── 0  # 5-fold 
     │    │   ├── train_bags.npy          # train data (512, 10, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
     │    │   ├── train_labels.npy        # train class label of each data (512, 10) = (the number of bags, bag size)
     │    │   ├── train_original_lps.npy  # train label proportions (512, 10) = (the number of bags, class label proportions)
     │    │   ├── val_bags.npy            # val data (10, 64, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
     │    │   ├── val_labels.npy          # val class label of each data (10, 64) = (the number of bags, bag size)
     │    │   └── val_original_lps.npy    # val label proportions(10, 10) = (the number of bags, class label proportions)
     │    │                
     │    ├── :
     │    │
     │    ├── 4  # 5-fold 
     │    │   ├── train_bags.npy          # train data (512, 10, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
     │    │   ├── train_labels.npy        # train class label of each data (512, 10) = (the number of bags, bag size)
     │    │   ├── train_original_lps.npy  # train label proportions (512, 10) = (the number of bags, class label proportions)
     │    │   ├── val_bags.npy            # val data (10, 64, 32, 32, 3) = (the number of bags, bag size, height, width, channel)
     │    │   ├── val_labels.npy          # val class label of each data (10, 64) = (the number of bags, bag size)
     │    │   └── val_original_lps.npy    # val label proportions(10, 10) = (the number of bags, class label proportions)
     │    │
     │    ├── test_data.npy               # (10000, 32, 32, 3) = (the number of data, height, width, channel)
     │    │
     │    └── test_label.npy              # (10000,) = (the number of data labels)
     │
     ├──  svhn  # dataset name
     │    ├── 0  # 5-fold
     :    :
```