#!/usr/bin/env python3
"""
Sprite extractor for FVP (FAVORITE View Point) games.  Standard library only.

Usage:  python -I extract_sprites.py <profile.json> <game_dir> <out_dir> <char> [char ...]
        (<out_dir>/<char>/ is created; <char> is the name used in the sprite file names)

Reads the sprite archive named in the profile (sprites.archive).  Archive and image formats are
in fvp_archive.py.
  * Sprite name = <prefix><char>_<pose>_<outfit>[L], where pose and outfit are the names the sprite
    loader maps the script's pose/outfit arguments to, and "L" is the close-up size.
    The body is a type-1 image; its face strip "<sprite><face_suffix>" is type 2.
  * The face strip frame N is what the script's face argument selects (PartsSelect(prim, N)).
    It is drawn at (face.off_x, face.off_y) inside the body image.

Output per character:
  body/<sprite>.png            the body without a face
  faces/<sprite>_fNN.png       every face frame
  full/<sprite>_fNN.png        body + face NN composited (normal size only, to save space)
  sheets/<sprite>.png          contact sheet of all faces (6 per row, index in the corner)
"""
import json
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fvp_archive import read_index, decode_hzc  # noqa: E402


def bgra_to_rgba(b):
    a = bytearray(b)
    a[0::4], a[2::4] = b[2::4], b[0::4]
    return a


def write_png(path, w, h, rgba):
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    stride = w * 4
    raw = b''.join(b'\0' + bytes(rgba[y * stride:(y + 1) * stride]) for y in range(h))
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
        f.write(chunk(b'IDAT', zlib.compress(raw, 6)))
        f.write(chunk(b'IEND', b''))


def blit(dst, dw, dh, src, sw, sh, x0, y0):
    """Alpha-composite src (RGBA, straight alpha) onto dst at (x0, y0)."""
    for y in range(sh):
        dy = y0 + y
        if not 0 <= dy < dh:
            continue
        for x in range(sw):
            dx = x0 + x
            if not 0 <= dx < dw:
                continue
            s = (y * sw + x) * 4
            sa = src[s + 3]
            if sa == 0:
                continue
            d = (dy * dw + dx) * 4
            if sa == 255:
                dst[d:d + 4] = src[s:s + 4]
                continue
            da = dst[d + 3]
            oa = sa + da * (255 - sa) // 255
            for c in range(3):
                dst[d + c] = (src[s + c] * sa + dst[d + c] * da * (255 - sa) // 255) // max(oa, 1)
            dst[d + 3] = oa


DIGITS = {  # 3x5 bitmap font for sheet labels
    '0': '111101101101111', '1': '010110010010111', '2': '111001111100111',
    '3': '111001111001111', '4': '101101111001001', '5': '111100111001111',
    '6': '111100111101111', '7': '111001001001001', '8': '111101111101111',
    '9': '111101111001111'}


def label(img, w, x0, y0, text, scale=3):
    for k, ch in enumerate(text):
        bits = DIGITS[ch]
        for i, bit in enumerate(bits):
            if bit != '1':
                continue
            for yy in range(scale):
                for xx in range(scale):
                    x = x0 + k * 4 * scale + (i % 3) * scale + xx
                    y = y0 + (i // 3) * scale + yy
                    p = (y * w + x) * 4
                    img[p:p + 4] = b'\xff\x00\x00\xff'


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    prof_path, game, out = sys.argv[1:4]
    chars = sys.argv[4:]
    with open(prof_path, encoding='utf-8') as f:
        sp = json.load(f)['sprites']
    prefix = sp['name_prefix'].split('/', 1)[-1]   # '<archive>/<prefix>' -> '<prefix>'
    face_sfx = sp['face_suffix']
    arc = os.path.join(game, sp['archive'])
    ix = read_index(arc)
    with open(arc, 'rb') as f:
        def load(name):
            off, sz = ix[name]
            f.seek(off)
            return decode_hzc(f.read(sz))
        for ch in chars:
            base = os.path.join(out, ch)
            for sub in ('body', 'faces', 'full', 'sheets'):
                os.makedirs(os.path.join(base, sub), exist_ok=True)
            bodies = sorted(n for n in ix if n.startswith('%s%s_' % (prefix, ch)) and not n.endswith(face_sfx))
            for bn in bodies:
                body = load(bn)
                face = load(bn + face_sfx)
                bw, bh = body['w'], body['h']
                brgba = bgra_to_rgba(body['data'])
                write_png(os.path.join(base, 'body', bn + '.png'), bw, bh, brgba)
                fw, fh, nf = face['w'], face['h'], face['frames']
                frgba = bgra_to_rgba(face['data'])
                fsz = fw * fh * 4
                cols = 6
                rows = (nf + cols - 1) // cols
                sw, sh = cols * fw, rows * fh
                sheet = bytearray(b'\xff\xff\xff\xff' * (sw * sh))
                for k in range(nf):
                    fr = frgba[k * fsz:(k + 1) * fsz]
                    write_png(os.path.join(base, 'faces', '%s_f%02d.png' % (bn, k)), fw, fh, fr)
                    # sheet cell = face over the body area it covers, so the face reads naturally
                    cell = bytearray(b'\xff\xff\xff\xff' * (fw * fh))
                    for y in range(fh):
                        sy = face['oy'] + y
                        if 0 <= sy < bh:
                            x1 = min(fw, bw - face['ox'])
                            s = (sy * bw + face['ox']) * 4
                            cell[y * fw * 4:y * fw * 4 + x1 * 4] = brgba[s:s + x1 * 4]
                    blit(cell, fw, fh, fr, fw, fh, 0, 0)
                    # drop transparent pixels to white
                    for p in range(0, len(cell), 4):
                        if cell[p + 3] == 0:
                            cell[p:p + 4] = b'\xff\xff\xff\xff'
                    label(cell, fw, 2, 2, '%d' % k)
                    cx, cy = (k % cols) * fw, (k // cols) * fh
                    for y in range(fh):
                        d = ((cy + y) * sw + cx) * 4
                        sheet[d:d + fw * 4] = cell[y * fw * 4:(y + 1) * fw * 4]
                    if not bn.endswith('L'):
                        full = bytearray(brgba)
                        blit(full, bw, bh, fr, fw, fh, face['ox'], face['oy'])
                        write_png(os.path.join(base, 'full', '%s_f%02d.png' % (bn, k)), bw, bh, full)
                write_png(os.path.join(base, 'sheets', bn + '.png'), sw, sh, sheet)
                print(bn, 'body %dx%d' % (bw, bh), 'faces %d x %dx%d at (%d,%d)' % (nf, fw, fh, face['ox'], face['oy']))


if __name__ == '__main__':
    main()
