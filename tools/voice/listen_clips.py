#!/usr/bin/env python3
"""
Extract the clips a person should listen to, as WAV, plus a sheet to fill in.

Usage:
  python -I listen_clips.py <char> --archive <voice archive> [--check id,id,...] [--out DIR]
                            [--project DIR] [--emotions DIR]
  --archive   the game's FVP voice archive (e.g. ..\\voice.bin)
  --project / --emotions as in label_sheet.py (defaults: current folder, <project>/emotions)
  --out       default <project>/dataset/voice/listen

Reads <project>/dataset/voice/label_sheets/*_listen.tsv (voice, candidates, why) and *_newcat.tsv
(voice, proposed category, why), and <emotions>/<char>_labels.tsv for --check ids.
Writes:
  <out>\\1_check_first\\<id>_<label>_<text>.wav           weakest current labels (--check)
  <out>\\2_new_category\\<Category>\\...                    proposed new emotions (Wordless skipped)
  <out>\\3_listen_list\\<first-guess style>\\<id>_<A-B>_<text>.wav
  <out>\\listen_sheet.csv                                   one row per clip; the user fills 'verdict'
The CSV is UTF-8 with BOM so Excel opens it correctly.  The output folder is rebuilt each run.
Clips outside 2-12 s are left out (they never reach the training file); their ids are printed instead.
"""
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fvp'))
from fvp_archive import read_index  # noqa: E402


def tsv(path):
    return [l.rstrip('\n').split('\t') for l in open(path, encoding='utf-8-sig')
            if l.strip() and not l.startswith('#')]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--check': '', '--out': '', '--project': '.', '--emotions': '', '--archive': ''}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    if len(a) != 1 or not opt['--archive']:
        sys.exit(__doc__)
    char = a[0]
    proj = opt['--project']
    vdir = os.path.join(proj, 'dataset', 'voice')
    emo_dir = opt['--emotions'] or os.path.join(proj, 'emotions')
    out = opt['--out'] or os.path.join(vdir, 'listen')
    rows = {}
    for l in open(os.path.join(vdir, char + '_voice_sprites.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        rows[r['voice']] = r
    labels = {r[0]: r for r in tsv(os.path.join(emo_dir, char + '_labels.tsv'))}
    items = []  # (group, sub folder, voice, label or candidates, why)
    for v in [x for x in opt['--check'].split(',') if x]:
        items.append(('1_check_first', '', v, labels[v][1], labels[v][3]))
    sheets = os.path.join(vdir, 'label_sheets')
    for p in sorted(glob.glob(os.path.join(sheets, '*_newcat.tsv'))):
        for v, cat, why in tsv(p):
            if not cat.startswith('Wordless'):
                items.append(('2_new_category', cat, v, cat, why))
    for p in sorted(glob.glob(os.path.join(sheets, '*_listen.tsv'))):
        for v, cand, why in tsv(p):
            items.append(('3_listen_list', cand.split('/')[0], v, cand, why))

    # clips outside 2-12 s never reach the training file (build_sbv2.py), so the user is not asked about them;
    # label those from context and voice alone (the label still counts for chat data)
    audio = os.path.join(vdir, char + '_audio.tsv')
    dur = {r[0]: float(r[1]) for r in tsv(audio)[1:]}
    short = [it for it in items if not 2.0 <= dur.get(it[2], 0) <= 12.0]
    items = [it for it in items if it not in short]
    if short:
        print('not listed (outside 2-12 s, decide from context and voice): %d  %s'
              % (len(short), ','.join(it[2] for it in short)))

    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)  # the sheet is written even when nothing is left to listen to
    bad = re.compile(r'[\\/:*?"<>|\s]')
    arc = opt['--archive']
    ix = read_index(arc)
    sheet = []
    with open(arc, 'rb') as f:
        for group, sub, v, cur, why in items:
            text = rows[v]['text'].strip()
            d = os.path.join(out, group, sub)
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, '%s_%s_%s.wav' % (v, cur.replace('/', '-'), bad.sub('', text)[:24]))
            off, sz = ix[v]
            f.seek(off)
            with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as t:
                t.write(f.read(sz))
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', t.name,
                            '-ac', '1', '-ar', '44100', '-sample_fmt', 's16', dst], check=True)
            os.unlink(t.name)
            sheet.append([os.path.relpath(dst, out), v, group, cur, text, why, ''])
    with open(os.path.join(out, 'listen_sheet.csv'), 'w', encoding='utf-8-sig', newline='') as g:
        w = csv.writer(g)
        w.writerow(['file', 'voice', 'group', 'current label / candidates', 'text', 'why', 'verdict'])
        w.writerows(sheet)
    print('%d clips -> %s' % (len(sheet), out))


if __name__ == '__main__':
    main()
