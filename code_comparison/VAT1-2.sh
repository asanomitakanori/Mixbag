for dataset in svhn
do
    for bags in 512
    do
    if [ $bags -eq 512 ]; then
        ins=10
    elif [ $bags -eq 256 ]; then
        ins=20
    elif [ $bags -eq 128 ]; then
        ins=40
    fi
        for x in  1
        do
            for cons in vat pi
            do
                if [ $x -eq 1 ]; then
                    static=True
                    for standard in 0.005
                    do 
                        for choice in uniform
                        do
                            python code_comparison/PL_VAT.py --dataset=$dataset --standard_normal_value=$standard --choice=$choice --ins=$ins --bags=$bags --classes=10 --consistency=$cons  --statistic  --channels=3
                        done
                    done
                elif [  $x -eq 0 ]; then
                    python code_comparison/PL_VAT.py --dataset=$dataset --standard_normal_value=0.005 --choice=uniform --ins=$ins --bags=$bags --classes=10 --consistency=$cons --channels=3
                fi
            done
        done
    done
done
