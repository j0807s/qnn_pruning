#!/bin/bash
echo "Run original PQCs"

for i in 1 3 5 9
do
    for j in {1..10}
    do
        python run_pqcs.py -nl $j -pqc $i >> output_org.txt 
        echo "$i"
        echo "running"
        grep "Training" output.txt >> result_org.txt
        grep "Best" output.txt >> result_org.txt
        rm output_org.txt
    done
done