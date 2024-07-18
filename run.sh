#!/bin/bash
batch_size=(10 100 1000 10000)
sizes=(1024 2048 4096)

for i in "${batch_size[@]}" ; do
    for j in "${sizes[@]}" ; do
            echo "batch_size: $i, size: $j"
            ./build/cufft_bench --batch $i --size $j
    done
done