#!/usr/bin/env python3
"""
FVP (FAVORITE View Point) archive helpers.  Standard library only.

  * .bin archives (graphics, voice, ...): u32 count, u32 name-table size,
    count x (u32 name_off, u32 data_off, u32 size), then the cp932 name table.
    Voice archive entries are named by voice id (8 digits) and hold Ogg Vorbis clips.
  * Image entries are "hzc1": u32 raw size, u32 header size (0x20), then an NVSG header
    (u16 ?, u16 type, u16 w, u16 h, u16 off_x, u16 off_y, u32 ?, u32 frames, ...) and a zlib stream.
    type 1 = one BGRA image, type 2 = <frames> BGRA images of w x h stacked vertically.
"""
import struct
import zlib


def read_index(path):
    with open(path, 'rb') as f:
        n, ntsz = struct.unpack('<II', f.read(8))
        ents = [struct.unpack('<III', f.read(12)) for _ in range(n)]
        names = f.read(ntsz)
    out = {}
    for no, off, sz in ents:
        e = names.index(b'\0', no)
        out[names[no:e].decode('cp932')] = (off, sz)
    return out


def decode_hzc(blob):
    assert blob[:4] == b'hzc1', blob[:4]
    raw_size, hdr_size = struct.unpack_from('<II', blob, 4)
    nv = blob[12:12 + hdr_size]
    assert nv[:4] == b'NVSG'
    _, typ, w, h, ox, oy, _, frames = struct.unpack_from('<HHHHHHII', nv, 4)
    data = zlib.decompress(blob[12 + hdr_size:])
    assert len(data) == raw_size
    return {'type': typ, 'w': w, 'h': h, 'ox': ox, 'oy': oy,
            'frames': max(frames, 1), 'data': data}


def read_entry(path, index, name):
    """Raw bytes of one archive entry."""
    off, sz = index[name]
    with open(path, 'rb') as f:
        f.seek(off)
        return f.read(sz)
