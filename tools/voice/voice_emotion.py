#!/usr/bin/env python3
"""
Speech-emotion scores per voice clip with emotion2vec+ (FunASR), as the "tone of voice" signal.

Usage:
  python -I voice_emotion.py <voice_sprites.jsonl> <voice archive> <out.tsv> [--voices id,id] [--model ...]

Needs: torch (CUDA build for the GPU), funasr, numpy, ffmpeg on PATH.
Default model: emotion2vec/emotion2vec_plus_large from Hugging Face (downloaded on first run).
Anime-style voices are outside the model's training domain: its absolute top class can be useless
(in one test nearly every clip came out "surprised" at ~1.0, while its own example clip was classified
correctly).  The raw class logits still move with the delivery, so they are saved and label_sheet.py compares each clip with the character's own median per class
(z-score).  Use it as "more angry / happy / sad ... than they usually sound", never as an absolute label.

Writes: voice  ser_top  logit_angry ... logit_surprised  (raw logits, 2 decimals).
An existing out.tsv is kept and only missing voices are added.
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fvp'))
from fvp_archive import read_index  # noqa: E402

CLASSES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'other', 'sad', 'surprised']
COLS = ['voice', 'ser_top'] + ['logit_' + c for c in CLASSES]


def decode16k(ogg):
    p = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', 'pipe:0', '-ac', '1', '-ar', '16000',
                        '-f', 'f32le', 'pipe:1'], input=ogg, capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def load(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8-sig') as f:
            head = f.readline().rstrip('\n').split('\t')
            for line in f:
                r = dict(zip(head, line.rstrip('\n').split('\t')))
                out[r['voice']] = r
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--voices': '', '--model': 'emotion2vec/emotion2vec_plus_large'}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    if len(a) != 3:
        sys.exit(__doc__)
    src, arc, out = a
    rows = [json.loads(l) for l in open(src, encoding='utf-8')]
    have = load(out)
    want = set(opt['--voices'].split(',')) if opt['--voices'] else None
    todo = [r['voice'] for r in rows if r['voice'] not in have and (want is None or r['voice'] in want)]
    if todo:
        from funasr import AutoModel
        import torch
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = AutoModel(model=opt['--model'], hub='hf', device=dev, disable_update=True)
        ix = read_index(arc)
        with open(arc, 'rb') as f:
            for k, v in enumerate(todo):
                off, sz = ix[v]
                f.seek(off)
                wav = decode16k(f.read(sz))
                res = model.generate(wav, granularity='utterance', extract_embedding=True,
                                     disable_pbar=True)[0]
                p0 = next(model.model.parameters())
                with torch.no_grad():
                    lg = model.model.proj(torch.tensor(np.asarray(res['feats']), dtype=p0.dtype,
                                                       device=p0.device).reshape(1, -1))[0].tolist()
                names = [lab.split('/')[-1] for lab in res['labels']]
                sc = {n: x for n, x in zip(names, lg) if n in CLASSES}
                top = max(sc, key=sc.get)
                have[v] = {'voice': v, 'ser_top': top,
                           **{'logit_' + c: '%.2f' % sc.get(c, 0.0) for c in CLASSES}}
                if k % 200 == 199:
                    print('%d/%d' % (k + 1, len(todo)), flush=True)
    order = {r['voice']: i for i, r in enumerate(rows)}
    with open(out, 'w', encoding='utf-8', newline='\n') as g:
        g.write('\t'.join(COLS) + '\n')
        for v in sorted(have, key=lambda v: order.get(v, 1 << 30)):
            g.write('\t'.join(have[v].get(c, '') for c in COLS) + '\n')
    print('emotion scores for %d clips -> %s (%d new)' % (len(have), out, len(todo)))


if __name__ == '__main__':
    main()
