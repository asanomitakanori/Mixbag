for dataset in "organsmnist" "pathmnist"
do
if [ $dataset = "bloodmnist" ]; then
    classes=8
    channels=3
elif [ $dataset = "octmnist" ]; then
    classes=4
    channels=1
elif [ $dataset = "organamnist" ]; then
    classes=11
    channels=1
elif [ $dataset = "organcmnist" ]; then
    classes=11
    channels=1
elif [ $dataset = "organsmnist" ]; then
    classes=11
    channels=1
elif [ $dataset = "pathmnist" ]; then
    classes=9
    channels=3
fi 
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
            python code_comparison/PL_instance.py --dataset=$dataset --ins=$ins --bags=$bags --classes=$classes --device='cuda:2' --augmentation=$aug --channels=$channels
        done
    done
done