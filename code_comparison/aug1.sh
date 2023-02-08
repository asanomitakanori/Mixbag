for dataset in cifar10 svhn
do
    for bags in 512 256 128
    do
    if [ $bags -eq 512 ]; then
        ins=10
    elif [ $bags -eq 256 ]; then
        ins=20
    elif [ $bags -eq 128 ]; then
        ins=40
    fi
        for aug in 'gaussblur'
        do
            python code_comparison/PL_instance.py --dataset=$dataset --ins=$ins --bags=$bags --classes=10 --device='cuda:2' --augmentation=$aug
        done
    done
done