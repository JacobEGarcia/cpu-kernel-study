#!/bin/sh
# Reproduce everything: builds kernels.c, runs it, then the numpy comparison.
set -e
gcc -O2 -march=x86-64 -o kernels kernels.c -lm
{ ./kernels
  OPENBLAS_NUM_THREADS=1 python3 bench_numpy.py
  python3 bench_numpy.py; } | tee results.txt
