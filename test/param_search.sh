learning_rate=(1e-3)
batch_size=(64 128)
weight_decay=(0.0009)
num_of_walk=(10 13 16 19)
for lr in ${learning_rate[@]}
    do
    for numofwalk in ${num_of_walk[@]}
        do
        for wd in ${weight_decay[@]}
            do
            python -u run.py \
            -d drug -p ../data/DDdataset \
            -cuda cuda:0 \
            -epochs 300 \
            -n ${numofwalk} \
            -lr ${lr} \
            -wd ${wd} \
            -train >> param_search_nosimfeature.log
        done
    done
done

