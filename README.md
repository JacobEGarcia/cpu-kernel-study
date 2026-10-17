# CPU kernel study: tiled and SIMD matmul, vectorized reduction

Small C project: fp32 matrix multiply (1024x1024) and a 64M-element sum, each timed against a plain baseline and,
for matmul, against numpy/OpenBLAS. Written with AI assistance under Jacob Garcia's direction; compiled, run and
checked for real. CPU only. No GPU or CUDA.

## Run
    ./run_all.sh        # needs gcc and python3 with numpy

## Results (one run; shared Intel Xeon 2.6 GHz VM, 2 vCPUs, gcc -O2, best of several reps)
| Kernel | Time | GFLOP/s | vs naive |
|---|---|---|---|
| naive ijk (C) | 3.118 s | 0.69 | 1x |
| tiled 64, scalar (C) | 0.632 s | 3.40 | 4.9x |
| tiled 64 + AVX2 FMA (C) | 0.114 s | 18.8 | 27x |
| numpy/OpenBLAS, 1 thread | 0.0186 s | 115.5 | 168x |
| numpy/OpenBLAS, 2 threads | 0.0093 s | 231.3 | 335x |

Naive timing varied between 2.7 s and 3.1 s across runs (shared VM), so the 25-27x figure is approximate.

**My best kernel is about 6x slower than OpenBLAS on one thread.** That is expected: OpenBLAS uses hand-tuned
register-blocked micro-kernels, packing, and AVX-512 on this CPU. This project uses only AVX2, a single tile level
and no packing. It is a learning-scale kernel, not a BLAS replacement.

Reduction, 64M floats: scalar 3.2 GB/s, 8 accumulators 1.5x, AVX2 4 vectors x 8 lanes 3.5x (11 GB/s).
Max matmul difference vs naive: 4.8e-6 (different summation order). Naive fp32 sum was 48% off a double reference
at this size; vector partial sums were 0.01% off.

## Why these choices help
**Loop order (i, k, j).** In C, matrices are row-major, so B[k][j] and C[i][j] are contiguous along j. The naive
i-j-k order walks B down a column, jumping 4 KB per step, so nearly every load misses cache. Putting j innermost makes
every inner-loop access sequential, which also lets the compiler or intrinsics load 8 floats at once.

**Tile size 64.** A 64x64 fp32 tile is 16 KB. Three tiles (A, B, C) are 48 KB, which fits in typical L1/L2 cache
(L1 is often 32-48 KB, L2 1 MB+). Each tile is reused 64 times before being evicted, so data comes from cache, not
RAM. Too small wastes loop overhead; too large spills out of cache. I did not sweep tile sizes, so 64 is a reasoned
choice, not a measured optimum.

**AVX2 FMA.** One instruction does 8 multiply-adds, so the same loop does up to 8x the arithmetic per cycle.

**Reduction accumulators.** A single running sum forces every add to wait for the previous one (latency-bound).
Several independent accumulators let the CPU overlap adds, and also shrink rounding error because each sums fewer terms.

## Limits
- CPU only, single thread for the C kernels. No GPU, CUDA, or multi-core scaling for the C code.
- One shared VM; absolute numbers will differ elsewhere. Not a controlled benchmark.
- The tile size was not tuned by sweep. AVX-512 is available on this CPU but not used.
