import struct, os, io, numpy as np
import lz4.frame, zstandard as zstd
import sys
sys.path.insert(0, '/tmp/work')
from arrowparse import parse_ipc_file, parse_message

class RangeReader(io.RawIOBase):
    def __init__(self, fd, off, n):
        self.fd=fd; self.off=off; self.n=n; self.pos=0
    def readable(self): return True
    def read(self, m=-1):
        if self.pos>=self.n: return b''
        if m<0 or m>self.n-self.pos: m=self.n-self.pos
        b=os.pread(self.fd, m, self.off+self.pos); self.pos+=len(b); return b
    def readinto(self, b):
        d=self.read(len(b)); b[:len(d)]=d; return len(d)

def stream_column(path, col_idx, chunk=1<<23):
    """yield numpy chunks for a primitive column of a single-batch compressed feather/IPC file"""
    fd, fields, blocks = parse_ipc_file(path)
    off, metalen, bodylen = blocks[0]
    info = parse_message(fd, off, metalen)
    bodyoff = info['bodyoff']
    bufs = info['buffers']
    data_off, data_len = bufs[col_idx*2 + 1]
    name, ttype, nullable = fields[col_idx]
    dtype = {2: np.int32, 3: np.float32}[ttype] if name!='side' and ttype!=2 else None
    # type 2 = Int (need bitwidth; here all int32 except side int8, month int16 only in label)
    # handle by name for our known schemas
    dtmap = {'sample_id':np.int32,'seconds_before_predict':np.float32,'price':np.float32,
             'volume':np.int32,'side':np.int8,'order_action':np.int8,
             'transaction_avgprice':np.float32,'transaction_volume':np.int32,'transaction_count':np.int32,
             'ask_price_1':np.float32,'ask_price_2':np.float32,'bid_price_1':np.float32,'bid_price_2':np.float32,
             'ask_volume_1':np.int32,'ask_volume_2':np.int32,'bid_volume_1':np.int32,'bid_volume_2':np.int32,
             'month':np.int16,'target':np.float32}
    dt = np.dtype(dtmap[name])
    abs_off = bodyoff + data_off
    hdr = os.pread(fd, 8, abs_off)
    usize = struct.unpack('<q', hdr)[0]
    nrows = info['nrows']
    raw = usize == -1
    if not raw:
        assert usize == nrows * dt.itemsize, (name, usize, nrows*dt.itemsize)
    item = dt.itemsize
    if raw:
        pos = abs_off; remaining = data_len
        while remaining > 0:
            m = min(chunk - (chunk % item), remaining)
            b = os.pread(fd, m, pos); pos += m; remaining -= m
            yield np.frombuffer(b, dtype=dt)
    else:
        rr = RangeReader(fd, abs_off+8, data_len-8)
        codec = info['codec']
        if codec == 0:
            f = lz4.frame.open(rr, 'rb')
        else:
            f = zstd.ZstdDecompressor().stream_reader(rr)
        while True:
            b = f.read(chunk - (chunk % item))
            if not b: break
            if len(b) % item: b = b[:len(b)//item*item]
            yield np.frombuffer(b, dtype=dt)

def n_rows(path):
    fd, fields, blocks = parse_ipc_file(path)
    info = parse_message(fd, blocks[0][0], blocks[0][1])
    return info['nrows']