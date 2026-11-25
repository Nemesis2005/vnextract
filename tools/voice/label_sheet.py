#!/usr/bin/env python3
"""
Write a labelling sheet: for each voiced line, the three signals used to pick its style.

Usage:
  python -I label_sheet.py <char> <out.md> [--from N] [--limit M] [--voices id,id] [--unlabelled]
                          [--project DIR] [--emotions DIR]
  --project   the extraction folder (default: current folder).  Reads <project>/dataset/voice/
              <char>_voice_sprites.jsonl, <char>_audio.tsv, <char>_ser.tsv and <project>/script/script.jsonl.
  --emotions  folder with <char>_labels.tsv (default: <project>/emotions).

Per line:
  1. Context : the 4 script items before the line and 2 after (narration after a line often
               describes how she said it), plus the scene and any event CG.
  2. Voice   : audio_features.py numbers as z-scores against her own median (robust, MAD), and
               a few words (loud / quiet, high / low, wide / flat intonation, fast / slow);
               plus voice_emotion.py (emotion2vec+) class logits as z-scores against her own
               median: "more angry / happy / sad ... than she usually sounds".
  3. Face    : the sprite face on screen and its default style from <char>.tsv (a hint only).
Lines already in <char>_labels.tsv show the current label.
"""
import json
import os
import sys


def robust_z(rows, key):
    vals = sorted(float(r[key]) for r in rows if r.get(key) not in (None, ''))
    if not vals:
        return lambda v: None
    med = vals[len(vals) // 2]
    mad = sorted(abs(v - med) for v in vals)[len(vals) // 2] * 1.4826 or 1.0
    return lambda v: None if v in (None, '') else (float(v) - med) / mad


SER_CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised']

WORDS = [  # (feature, z-threshold, word if high, word if low)
    ('loud_p90_db', 1.0, 'loud', 'quiet'),
    ('f0_med_hz', 1.0, 'high-pitched', 'low-pitched'),
    ('f0_range_st', 1.0, 'wide intonation', 'flat intonation'),
    ('chars_per_s', 1.0, 'fast', 'slow'),
]


def load_labels(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding='utf-8-sig'):
            if line.startswith('#') or not line.strip():
                continue
            p = (line.rstrip('\n').split('\t') + ['', '', '', ''])[:4]
            out[p[0]] = p
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--from': '0', '--limit': '50', '--voices': '', '--project': '.', '--emotions': ''}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    unl = '--unlabelled' in a
    a = [x for x in a if x != '--unlabelled']
    if len(a) != 2:
        sys.exit(__doc__)
    char, out = a
    proj = opt['--project']
    vdir = os.path.join(proj, 'dataset', 'voice')
    emo_dir = opt['--emotions'] or os.path.join(proj, 'emotions')
    rows = [json.loads(l) for l in open(os.path.join(vdir, '%s_voice_sprites.jsonl' % char), encoding='utf-8')]
    audio = {}
    ap = os.path.join(vdir, '%s_audio.tsv' % char)
    if os.path.exists(ap):
        with open(ap, encoding='utf-8') as f:
            head = f.readline().rstrip('\n').split('\t')
            for line in f:
                r = dict(zip(head, line.rstrip('\n').split('\t')))
                audio[r['voice']] = r
    z = {k: robust_z(list(audio.values()), k) for k, _, _, _ in WORDS}
    ser = {}
    sp = os.path.join(vdir, '%s_ser.tsv' % char)
    if os.path.exists(sp):
        with open(sp, encoding='utf-8-sig') as f:
            head = f.readline().rstrip('\n').split('\t')
            for line in f:
                r = dict(zip(head, line.rstrip('\n').split('\t')))
                ser[r['voice']] = r
    zser = {c: robust_z(list(ser.values()), 'logit_' + c) for c in SER_CLASSES}
    labels = load_labels(os.path.join(emo_dir, '%s_labels.tsv' % char))
    script = [json.loads(l) for l in open(os.path.join(proj, 'script', 'script.jsonl'), encoding='utf-8')]
    items = [r for r in script if r['type'] in ('speech', 'narration')]
    pos = {r['voice']: i for i, r in enumerate(items) if r['type'] == 'speech' and r.get('voice') is not None}

    sel = rows
    if opt['--voices']:
        want = set(opt['--voices'].split(','))
        sel = [r for r in rows if r['voice'] in want]
    else:
        if unl:
            sel = [r for r in sel if r['voice'] not in labels]
        sel = sel[int(opt['--from']):int(opt['--from']) + int(opt['--limit'])]

    def show(r):
        if r['type'] == 'speech':
            return '【%s】%s' % (r['name'], r['text'])
        return '（地）' + r['text']

    L = ['# %s labelling sheet (%d lines)' % (char, len(sel)), '',
         'Decide each style from all three signals; see the README for the rules.',
         'Write decisions to %s_labels.tsv as: voice<TAB>style<TAB>basis<TAB>note' % char,
         'basis = the signals that support the label: c (context), v (measured voice), f (face),'
         ' L (listened by a person).', '']
    for r in sel:
        i = pos.get(int(r['voice']))
        L.append('## %s  %s' % (r['voice'], r['scene']))
        if r['voice'] in labels:
            _, st, basis, note = labels[r['voice']]
            L.append('current label: **%s** (basis %s) %s' % (st, basis, note))
        L.append('')
        L.append('1. Context%s' % ('  [event CG %s]' % r['cg'] if r.get('cg') else ''))
        if i is not None:
            for k in range(max(0, i - 4), i):
                L.append('   ' + show(items[k]))
            L.append('   ▶ ' + show(items[i]))
            for k in range(i + 1, min(len(items), i + 3)):
                L.append('   ' + show(items[k]))
        au = audio.get(r['voice'])
        if au:
            parts, words = [], []
            for k, thr, hi, lo in WORDS:
                zz = z[k](au.get(k))
                if zz is None:
                    continue
                parts.append('%s %s (%+.1f)' % (k, au[k], zz))
                if zz >= thr:
                    words.append(hi)
                elif zz <= -thr:
                    words.append(lo)
            L.append('2. Voice   %s s, %s' % (au['dur_s'], ', '.join(parts)))
            L.append('           → %s' % (', '.join(words) or 'typical for this character (no feature beyond ±1)'))
        else:
            L.append('2. Voice   (no audio features yet: run audio_features.py)')
        se = ser.get(r['voice'])
        if se:
            zs = sorted(((zser[c](se.get('logit_' + c)), c) for c in SER_CLASSES if c != 'other'),
                        reverse=True)
            hi = ['%s %+.1f' % (c, v) for v, c in zs if v is not None and v >= 1.0]
            L.append('           emotion model vs their norm: ' + (', '.join(hi) if hi else
                     'nothing above +1 (typical delivery)') +
                     '   [all: %s]' % ', '.join('%s %+.1f' % (c, v) for v, c in zs if v is not None))
        if r.get('sprite'):
            L.append('3. Face    %s f%02d%s "%s" → hint %s   (%s)' % (
                r['pose'], r['face'], ' blush' if r['blush'] else '', r['emotion'],
                r.get('face_style'), r['image']))
        else:
            L.append('3. Face    none on screen')
        L.append('')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))
    print('wrote %d lines -> %s' % (len(sel), out))


if __name__ == '__main__':
    main()
