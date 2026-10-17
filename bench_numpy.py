"""numpy (OpenBLAS) sgemm reference for the same 1024x1024 fp32 problem. Run with
OPENBLAS_NUM_THREADS=1 for a fair single-thread comparison against kernels.c."""
import numpy as np, time, os
n = 1024
rng = np.random.default_rng(1)
A = rng.random((n, n), dtype=np.float32) - .5
B = rng.random((n, n), dtype=np.float32) - .5
A @ B  # warm up
best = 1e9
for _ in range(10):
    t = time.perf_counter(); C = A @ B; best = min(best, time.perf_counter() - t)
print(f"numpy/OpenBLAS sgemm n={n} threads={os.environ.get('OPENBLAS_NUM_THREADS','default')}: "
      f"{best:.4f} s  {2*n**3/best/1e9:.2f} GFLOP/s")
