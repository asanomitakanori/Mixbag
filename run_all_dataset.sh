for dataset in "cifar10" "svhn" "bloodmnist" "octmnist" "organamnist" "organcmnist" "organsmnist" "pathmnist" "pathmnist"
do
    python run.py --dataset=$dataset
done