# cufft-benchmarks

This repository contains a set of benchmarks for the cuFFT library. 

I am trying to see the different between using FP16, FP32 and FP64 for the cuFFT library.

# Usage


```bash
cmake -S . -B build
cmake --build build
./build/cufft_bench --batch 10 --size 1024
```

Or run all tests

```bash
bash run_all.sh
```