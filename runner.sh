#!/bin/bash
for n_trackers in 1
do
    counter=1
    while [ $counter -le 100 ]
    do
        echo $counter
        python3 run_modular.py $1 $n_trackers $2
        ((counter++))
    done
done

