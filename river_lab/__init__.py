"""Reproducible River experiments for lab steps 1–5."""
"""Project River experiments use a declared single-thread numerical profile."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
