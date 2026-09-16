# aligned multi-column streaming over single-batch compressed feather files
import sys, numpy as np, io, os
sys.path.insert(0, '/tmp/work')
from streamcol import stream_column
from arrowparse import parse_ipc_file

def names_idx(path, names):
    fd, fields, blocks = parse_ipc_file(path)
    lut = {n: i for i, (n, t, nu) in enumerate(fields)}
    os.close(fd)
    return [lut[n] for n in names]

def _exact(gen, nbytes_target):
    """rebuffer a byte-chunk generator into exact-size byte chunks (except final)"""
    buf = b''
    for b in gen:
        buf += b
        while len(buf) >= nbytes_target:
            yield buf[:nbytes_target]; buf = buf[nbytes_target:]
    if buf:
        yield buf

def ziter(path, names, elems=1<<20):
    """yield aligned numpy chunk lists, exactly `elems` rows per yield (last may be shorter)"""
    idxs = names_idx(path, names)
    gens = []
    for i, n in zip(idxs, names):
        # discover dtype itemsize via one probe generator
        g = stream_column(path, i, chunk=1<<20)
        first = next(g)
        item = first.dtype.itemsize
        target = elems * item
        def mk(first=first, g=g, item=item, target=target):
            def bytegen():
                yield first.tobytes()
                for c in g:
                    yield c.tobytes()
            for bb in _exact(bytegen(), target):
                yield np.frombuffer(bb[:len(bb)//item*item], dtype=first.dtype)
        gens.append(mk())
    while True:
        out = []
        for g in gens:
            try: out.append(next(g))
            except StopIteration: return
        L = len(out[0])
        assert all(len(o) == L for o in out), [len(o) for o in out]
        yield out