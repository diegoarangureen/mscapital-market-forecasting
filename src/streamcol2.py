# streamcol2: stream a feather (Arrow IPC) file in chunks, yielding numpy arrays per requested column.
import numpy as np, pyarrow as pa

def ziter(path, cols, elems=1<<19):
    reader = pa.ipc.open_file(path)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        arrays = [b.column(c).to_numpy(zero_copy_only=False) for c in cols]
        n = len(arrays[0])
        for s in range(0, n, elems):
            yield tuple(a[s:s+elems] for a in arrays)
