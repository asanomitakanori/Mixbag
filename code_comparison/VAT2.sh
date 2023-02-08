for dataset in "octmnist"
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
    for bags in 512
    do
        if [ $bags -eq 512 ]; then
            ins=10
        elif [ $bags -eq 256 ]; then
            ins=20
        elif [ $bags -eq 128 ]; then
            ins=40
    fi
        for x in  0 1
        do
            # for cons in vat pi
            for cons in pi
            do
                if [ $x -eq 1 ]; then
                    static=True
                    for standard in 0.005
                    do 
                        for choice in uniform
                        do
                            python code_comparison/PL_VAT.py --dataset=$dataset --standard_normal_value=$standard --choice=$choice --ins=$ins --bags=$bags --classes=$classes --consistency=$cons --statistic=$static  --channels=$channels 
                        done
                    done
                elif [  $x -eq 0 ]; then
                    python code_comparison/PL_VAT.py --dataset=$dataset --standard_normal_value=0.005 --choice=uniform --ins=$ins --bags=$bags --classes=$classes --consistency=$cons --channels=$channels
                fi
            done
        done
    done
done
