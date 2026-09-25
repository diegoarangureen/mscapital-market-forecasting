"""Optional pilot: independent models on XLA devices; no gradient all-reduce."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')


from audit.launcher import worker


if __name__=='__main__':
    import sys
    import torch_xla
    torch_xla.launch(worker,args=(sys.argv[1:],))
