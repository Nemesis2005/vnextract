#!/usr/bin/env python3
"""
Measurable "tone of voice" features for voice clips (stdlib + ffmpeg on PATH).

Usage:
  python -I audio_features.py <voice_sprites.jsonl> <voice archive> <out.tsv> [--voices id,id,...]

<voice archive> is the game's FVP voice archive (e.g. voice.bin), entries named by voice id.
For every voiced line (or only --voices) it decodes the Ogg clip with ffmpeg to 8 kHz mono and writes:
  voice  dur_s  chars_per_s  rms_db  peak_db  loud_p90_db  f0_med_hz  f0_p10_hz  f0_p90_hz  f0_range_st  voiced_ratio
  * rms_db / loud_p90_db : mean and 90th-percentile frame loudness (dBFS), 20 ms frames
  * f0_*                 : pitch from normalised autocorrelation on voiced frames (100-700 Hz)
  * f0_range_st          : p90/p10 pitch spread in semitones (intonation movement)
  * chars_per_s          : text characters (no punctuation) per second, a speaking-rate proxy
These are a stand-in for listening, not a replacement: they separate loud/high/wide (excited,
angry, surprised, happy) from quiet/low/flat (neutral, serious, sad), and nothing finer.
An existing out.tsv is kept and only missing voices are added, so the full run is done once.
"""
import array
import audioop
import json
import math
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fvp'))
from fvp_archive import read_index  # noqa: E402

SR = 8000
FRAME = 320   # 40 ms
HOP = 160     # 20 ms
LAG_MIN, LAG_MAX = SR // 700, SR // 100
COLS = ['voice', 'dur_s', 'chars_per_s', 'rms_db', 'peak_db', 'loud_p90_db', 'f0_med_hz',
        'f0_p10_hz', 'f0_p90_hz', 'f0_range_st', 'voiced_ratio']
PUNCT = re.compile(r'[\s、。！？!?…‥・―ー〜～「」『』（）()]')


def decode(ogg):
    p = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-ac', '1', '-ar', str(SR),
                        '-f', 's16le', 'pipe:1'], input=ogg, capture_output=True, check=True)
    return p.stdout


def db(x):
    return 20 * math.log10(max(x, 1) / 32768.0)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else 0.0


def features(pcm, text):
    s = array.array('h', pcm)
    n = len(s)
    dur = n / SR
    rms = []
    f0 = []
    for st in range(0, max(n - FRAME, 0), HOP):
        fr = pcm[st * 2:(st + FRAME) * 2]
        rms.append(audioop.rms(fr, 2))
    if not rms:
        return None
    top = max(rms)
    for k, r in enumerate(rms):
        if r < top * 0.15:
            continue
        x = s[k * HOP:k * HOP + FRAME]
        m = sum(x) / len(x)
        x = [v - m for v in x]
        e0 = sum(v * v for v in x)
        if e0 <= 0:
            continue
        best, blag = 0.0, 0
        for lag in range(LAG_MIN, LAG_MAX + 1):
            a, b = x[:-lag], x[lag:]
            c = sum(map(float.__mul__, a, b)) / math.sqrt(sum(v * v for v in a) * sum(v * v for v in b) or 1)
            if c > best:
                best, blag = c, lag
        if best > 0.6 and blag:
            f0.append(SR / blag)
    chars = len(PUNCT.sub('', text))
    loud = [db(r) for r in rms]
    p10, p90 = pct(f0, 0.1), pct(f0, 0.9)
    return {
        'dur_s': round(dur, 2),
        'chars_per_s': round(chars / dur, 2) if dur else 0,
        'rms_db': round(db(audioop.rms(pcm, 2)), 1),
        'peak_db': round(db(audioop.max(pcm, 2)), 1),
        'loud_p90_db': round(pct(loud, 0.9), 1),
        'f0_med_hz': round(pct(f0, 0.5)) if f0 else '',
        'f0_p10_hz': round(p10) if f0 else '',
        'f0_p90_hz': round(p90) if f0 else '',
        'f0_range_st': round(12 * math.log2(p90 / p10), 1) if f0 and p10 else '',
        'voiced_ratio': round(len(f0) / len(rms), 2),
    }


def load(path):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        for line in f:
            row = dict(zip(head, line.rstrip('\n').split('\t')))
            out[row['voice']] = row
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    only = None
    if '--voices' in a:
        i = a.index('--voices')
        only = set(a[i + 1].split(','))
        del a[i:i + 2]
    if len(a) != 3:
        sys.exit(__doc__)
    src, arc, out = a
    rows = [json.loads(l) for l in open(src, encoding='utf-8')]
    have = load(out)
    ix = read_index(arc)
    todo = [r for r in rows if r['voice'] not in have and (only is None or r['voice'] in only)]
    with open(arc, 'rb') as f:
        for k, r in enumerate(todo):
            off, sz = ix[r['voice']]
            f.seek(off)
            feat = features(decode(f.read(sz)), r['text'])
            if feat:
                feat['voice'] = r['voice']
                have[r['voice']] = {c: str(feat[c]) for c in COLS}
            if k % 200 == 199:
                print('%d/%d' % (k + 1, len(todo)), flush=True)
    order = {r['voice']: i for i, r in enumerate(rows)}
    with open(out, 'w', encoding='utf-8', newline='\n') as g:
        g.write('\t'.join(COLS) + '\n')
        for v in sorted(have, key=lambda v: order.get(v, 1 << 30)):
            g.write('\t'.join(have[v].get(c, '') for c in COLS) + '\n')
    print('features for %d clips -> %s (%d new)' % (len(have), out, len(todo)))


if __name__ == '__main__':
    main()
