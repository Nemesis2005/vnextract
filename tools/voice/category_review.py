#!/usr/bin/env python3
"""
Write a review sheet for one style: every line labelled with it, every listen-list line that names it as a
candidate, and (optionally) the lines of another style to compare with, each with its script context,
voice numbers and face.  Use it when a style looks doubtful (e.g. "should Annoyed merge into Angry?").

Usage:
  python -I category_review.py <char> <Style> [--compare Style2] [--out <file.md>]
                               [--project DIR] [--emotions DIR]
  --project / --emotions as in label_sheet.py (defaults: current folder, <project>/emotions).
  default out: <project>/dataset/voice/label_sheets/<style>_review.md
Uses label_sheet.py for the context (4 script lines before, 2 after).
"""
import glob
import os
import subprocess
import sys
import tempfile


def tsv(path):
    return [l.rstrip('\n').split('\t') for l in open(path, encoding='utf-8-sig')
            if l.strip() and not l.startswith('#')]


def sheet(char, ids, extra=None, paths=()):
    if not ids:
        return '(none)\n'
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, 's.md')
        subprocess.run([sys.executable, '-I', os.path.join(here, 'label_sheet.py'), char, out,
                        '--voices', ','.join(ids)] + list(paths), check=True, stdout=subprocess.DEVNULL)
        text = open(out, encoding='utf-8').read()
    body = text[text.index('\n## ') + 1:]
    lines = []
    for l in body.split('\n'):
        if l.startswith('## '):
            lines.append('#' + l)
            if extra and l[3:11] in extra:
                lines.append('candidates: ' + extra[l[3:11]])
        else:
            lines.append(l)
    return '\n'.join(lines)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--compare': '', '--out': '', '--project': '.', '--emotions': ''}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    if len(a) != 2:
        sys.exit(__doc__)
    char, style = a[0], a[1]
    proj = opt['--project']
    emo_dir = opt['--emotions'] or os.path.join(proj, 'emotions')
    paths = ['--project', proj, '--emotions', emo_dir]
    out = opt['--out'] or os.path.join(proj, 'dataset', 'voice', 'label_sheets', style.lower() + '_review.md')
    labels = tsv(os.path.join(emo_dir, char + '_labels.tsv'))
    lab = [r[0] for r in labels if r[1] == style]
    cand = {}
    for p in sorted(glob.glob(os.path.join(proj, 'dataset', 'voice', 'label_sheets', '*_listen.tsv'))):
        for r in tsv(p):
            if style in r[1].split('/'):
                cand[r[0]] = r[1]
    doc = ['# %s lines for review' % style, '',
           'Labelled %s: %d.  Listen list with %s as a candidate: %d.' % (style, len(lab), style, len(cand)), '',
           '## Labelled %s (%d)' % (style, len(lab)), '', sheet(char, lab, paths=paths), '',
           '## Listen list, %s as a candidate (%d)' % (style, len(cand)), '', sheet(char, list(cand), cand, paths)]
    if opt['--compare']:
        cmp_ids = [r[0] for r in labels if r[1] == opt['--compare']]
        doc += ['', '## %s labels for comparison (%d)' % (opt['--compare'], len(cmp_ids)), '', sheet(char, cmp_ids, paths=paths)]
    open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(doc))
    print('%s: %d labelled, %d listen candidates -> %s' % (style, len(lab), len(cand), out))


if __name__ == '__main__':
    main()
