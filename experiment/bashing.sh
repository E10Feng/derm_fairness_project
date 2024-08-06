#!/bin/bash

# modify or duplicate this file as necessary to run trials

runs=10 #number of runs per experiment

script='/PATH/TO/MAIN_FILE'

args="ARGS"

for ((i=1;i<=runs;i++))
do
    echo "Run #$i with arguments: $args"
    /home/path/to/python/interpreter $script $args $i
done

echo "All runs completed."


# example of terminal command: nohup ./bashing2.sh > dev_2_regular.out &
# job id: 