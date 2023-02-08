# for dataset in mnist cifar10 svhn
for dataset in mnist svhn
do
    for w_MIL in 0 0.01 1 100
    do
        # python3.9 code/main-MIL.py --dataset=$dataset --w_MIL=$w_MIL
        python3.9 code/vis.py --dataset=$dataset --w_MIL=$w_MIL
    done
done