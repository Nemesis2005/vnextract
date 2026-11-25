#!/usr/bin/env python3
"""
Build a Style-Bert-VITS2 training folder from <char>_voice_sprites.jsonl (voice_sprites.py).

Usage:
  python -I build_sbv2.py <voice_sprites.jsonl> <voice archive> <Data/<model> dir>
                          --speaker <Name> [--limit N] [--face-fallback] [--ogg]

  <voice archive>  the game's FVP voice archive (e.g. voice.bin), entries named by voice id.
  --speaker        speaker name written into esd.list (ASCII).
  default          only lines with a final style in <char>_labels.tsv
                   (decided from context + tone of voice + face; voice_sprites.py copies it in).
  --face-fallback  also take unlabelled lines, using the face hint (face_style) as the style.
                   Rougher: the face is a guide only, so use it for experiments.
  --limit N        stop after N lines (for a quick sample).
  --ogg            keep the original Ogg Vorbis (48 kHz) instead of converting to WAV.
  Always skipped:  clips outside 2-12 s (Style-Bert-VITS2 limit; dur_s from <char>_audio.tsv next to the
                   jsonl) and lines whose text has no actual characters (only punctuation, e.g. 「…………」).

Output (the layout Style-Bert-VITS2 expects; one sub-folder per style):
  <out>/raw/<Style>/<voice>.wav        44.1 kHz mono 16-bit WAV (ffmpeg)
  <out>/esd.list                       <Style>/<voice>.wav|<speaker>|JP|<text>
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fvp'))
from fvp_archive import read_index  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--speaker': None, '--limit': None}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    face_fallback = '--face-fallback' in a
    keep_ogg = '--ogg' in a
    a = [x for x in a if x not in ('--face-fallback', '--ogg')]
    if len(a) != 3 or not opt['--speaker']:
        sys.exit(__doc__)
    src, arc, out = a
    limit = int(opt['--limit']) if opt['--limit'] else None
    ix = read_index(arc)
    rows = [json.loads(l) for l in open(src, encoding='utf-8')]
    # training-clip rules: 2-12 s long (Style-Bert-VITS2 limit), and text with actual characters
    dur = {}
    audio = src.replace('_voice_sprites.jsonl', '_audio.tsv')
    if os.path.exists(audio):
        for l in open(audio, encoding='utf-8-sig').read().splitlines()[1:]:
            f = l.split('\t')
            dur[f[0]] = float(f[1])
    else:
        print('warning: %s not found, clip length not checked' % audio)
    skipped = {'length': 0, 'no text': 0}
    lines = []
    with open(arc, 'rb') as f:
        for r in rows:
            # the SBV2 folder is the style's voice_style (e.g. Question trains as Neutral)
            style = r.get('voice_style') or r.get('style') or (r.get('face_style') if face_fallback else None)
            if not style:
                continue
            text = r['text'].strip()
            if not re.sub(r'[\W_]', '', text):   # only punctuation, e.g. 「…………」
                skipped['no text'] += 1
                continue
            if r['voice'] in dur and not 2.0 <= dur[r['voice']] <= 12.0:
                skipped['length'] += 1
                continue
            off, sz = ix[r['voice']]
            f.seek(off)
            ogg = f.read(sz)
            d = os.path.join(out, 'raw', style)
            os.makedirs(d, exist_ok=True)
            ext = '.ogg' if keep_ogg else '.wav'
            dst = os.path.join(d, r['voice'] + ext)
            if keep_ogg:
                open(dst, 'wb').write(ogg)
            else:
                with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as t:
                    t.write(ogg)
                subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', t.name,
                                '-ac', '1', '-ar', '44100', '-sample_fmt', 's16', dst], check=True)
                os.unlink(t.name)
            lines.append('%s/%s%s|%s|JP|%s' % (style, r['voice'], ext, opt['--speaker'], text))
            if limit and len(lines) >= limit:
                break
    with open(os.path.join(out, 'esd.list'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    print('wrote %d lines -> %s  (skipped: %d outside 2-12 s, %d without text)'
          % (len(lines), os.path.join(out, 'esd.list'), skipped['length'], skipped['no text']))


if __name__ == '__main__':
    main()
