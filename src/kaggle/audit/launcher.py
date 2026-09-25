"""Importable worker target also works inside a generated standalone kernel."""


def worker(index, argv):
    import torch_xla.runtime as xr
    from train_audited import parser, run
    args = parser().parse_args(argv)
    args.device = 'xla'
    args.worker_id = xr.global_ordinal()
    args.workers = xr.world_size()
    run(args)
