#!/bin/bash
for n_trackers in 1 2 5 7
do
    counter=1
    while [ $counter -le 20 ]
    do
        echo $counter
        python3 run_modular.py $1 $n_trackers $2
        ((counter++))
    done
done

