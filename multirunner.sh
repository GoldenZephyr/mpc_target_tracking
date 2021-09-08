#!/bin/bash
for i in 1 2
do
    for j in {1..5}
    do
        ./run_persistent_monitoring.py --n_targets $1 --n_trackers $2 --hlp_type $3 --env_type $4 --n_steps 5000 --plot_mode none --log &
    done
    wait
done
