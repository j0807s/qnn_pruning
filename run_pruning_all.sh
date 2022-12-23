#!/bin/bash
echo "Run original PQCs"

for i in 1 3 5 9
do
    for j in {1..10}
    do
        python run_pqcs.py -nl $j -pqc $i -p 0.2 >> output_pruning.txt 
        echo "$i"
        echo "running"
        grep "Training" output.txt >> result_pruning.txt
        grep "Best" output.txt >> result_pruning.txt
        rm output_pruning.txt
    done
done