import struct, os

class FB:
    """generic flatbuffers table accessor over a bytes-like buffer"""
    def __init__(self, buf, pos):
        self.buf = buf; self.pos = pos
        so = struct.unpack_from('<i', buf, pos)[0]
        self.vt = pos - so
        self.vlen = struct.unpack_from('<H', buf, self.vt)[0]
    def _off(self, i):
        if 4 + 2*i + 2 <= self.vlen:
            o = struct.unpack_from('<H', self.buf, self.vt + 4 + 2*i)[0]
            return o if o != 0 else None
        return None
    def scalar(self, i, fmt, default=0):
        o = self._off(i)
        return struct.unpack_from(fmt, self.buf, self.pos+o)[0] if o is not None else default
    def table(self, i):
        o = self._off(i)
        if o is None: return None
        p = self.pos + o
        uoff = struct.unpack_from('<I', self.buf, p)[0]
        return FB(self.buf, p + uoff)
    def vector(self, i):
        o = self._off(i)
        if o is None: return None
        p = self.pos + o
        uoff = struct.unpack_from('<I', self.buf, p)[0]
        vp = p + uoff
        n = struct.unpack_from('<I', self.buf, vp)[0]
        return vp + 4, n

def read_at(fd, off, n):
    return os.pread(fd, n, off)

def parse_ipc_file(path):
    """returns (schema fields list, record batch blocks, message info for batch 0)"""
    fd = os.open(path, os.O_RDONLY)
    size = os.fstat(fd).st_size
    tail = read_at(fd, size-10, 10)
    flen = struct.unpack_from('<i', tail, 0)[0]
    assert tail[4:] == b'ARROW1', tail
    footer = read_at(fd, size-10-flen, flen)
    fb = FB(footer, struct.unpack_from('<I', footer, 0)[0])
    # footer: version(0) schema(1) dictionaries(2) recordBatches(3)
    vp, n = fb.vector(3)
    blocks = []
    for i in range(n):
        off, metalen, bodylen = struct.unpack_from('<qqq', footer, vp + i*24)
        blocks.append((off, metalen, bodylen))
    # schema
    sch = fb.table(1)
    fvp, nf = sch.vector(1)  # fields vector (slot 1, not 2)
    fields = []
    for i in range(nf):
        fo = struct.unpack_from('<I', footer, fvp + 4*i)[0]
        ft = FB(footer, fvp + 4*i + fo)
        nvp, nn = ft.vector(0)  # name
        name = bytes(footer[nvp:nvp+nn]).decode()
        nullable = ft.scalar(1, '<b', 0)
        ttype = ft.scalar(2, '<b', 0)  # Type union: 0-16ish
        fields.append((name, ttype, nullable))
    return fd, fields, blocks

def parse_message(fd, off, metalen):
    mraw = read_at(fd, off, metalen)
    pos = 0
    if struct.unpack_from('<I', mraw, 0)[0] == 0xFFFFFFFF:
        pos = 4
    mlen = struct.unpack_from('<I', mraw, pos)[0]
    msg = FB(mraw, pos + 4 + struct.unpack_from('<I', mraw, pos+4)[0])
    htype = msg.scalar(1, '<b', 0)
    bodylen = msg.scalar(3, '<q', 0)
    hdr = msg.table(2)
    info = {'bodylen': bodylen, 'bodyoff': off + metalen}
    if htype == 3:  # RecordBatch
        info['nrows'] = hdr.scalar(0, '<q', 0)
        vp, n = hdr.vector(1)  # nodes: FieldNode structs (long length, long null_count) 16B
        info['nodes'] = [struct.unpack_from('<qq', mraw, vp + i*16) for i in range(n)]
        vp, n = hdr.vector(2)  # buffers: Buffer structs (long offset, long length) 16B
        info['buffers'] = [struct.unpack_from('<qq', mraw, vp + i*16) for i in range(n)]
        comp = hdr.table(3)
        if comp is not None:
            info['codec'] = comp.scalar(0, '<b', 0)  # 0=lz4,1=zstd
            info['method'] = comp.scalar(1, '<b', 0) # 0=buffer
        else:
            info['codec'] = None
    return info

if __name__ == '__main__':
    import sys
    fd, fields, blocks = parse_ipc_file(sys.argv[1])
    print('fields:', fields)
    print('blocks:', blocks)
    for (off, ml, bl) in blocks:
        info = parse_message(fd, off, ml)
        print('batch rows:', info.get('nrows'), 'codec:', info.get('codec'), 'nodes:', info.get('nodes'), 'buffers:', info.get('buffers'), 'bodylen', info['bodylen'])