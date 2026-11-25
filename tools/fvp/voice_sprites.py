#!/usr/bin/env python3
"""
Link every voiced line of one character to the sprite (pose / outfit / face) on screen when the
line plays, for FVP (FAVORITE View Point) games.  Standard library only.

Usage:
  python -I voice_sprites.py <profile.json> <Game.hcb> <out_dir> --char <name> [--emotions DIR]

  --char      the character's canonical name (the `char` field of script.jsonl); it must also be
              the name in the sprite file names (<prefix><name>_...).
  --emotions  folder with the hand-edited <char>.tsv, <char>_styles.tsv, <char>_motions.tsv and
              <char>_labels.tsv (default: ./emotions).  Missing files are treated as empty.

Writes  <out_dir>/<char>_voice_sprites.jsonl  and  .tsv  (one row per voiced line, story order),
and <out_dir>/<char>_emotions_motions.txt (readable list of emotions and motions with counts).

How the link is made (profile "sprites" block):
  * Sprite loader loader(char, pose, outfit, face, size, pos, ...).  It builds
    "<name_prefix><char>_<pose>_<outfit>[L|S|LL]" and selects face frame <face> of "<...><face_suffix>"
    (PartsSelect).  size: 0 -> "L" close-up, 1 -> normal, 2 -> "S", -1 -> "LL", nil -> keep last.
  * The scenario never calls the loader directly: many tiny wrappers each hard-code
    (char, pose, outfit, face) and pass 9 args through (first = size).
  * hide(char, x) removes one character.
  * Event CGs: small functions that push a CG name and call cg_show.
The extractor sweep (fvp_hcb.Extractor) is reused through its on_call hook, so the sprite state
follows the same story order as script/*.txt.  The state of the character is:
  shown  : from a sprite wrapper call
  cleared: on her hide call, any event CG, a background change, or a screen clear / new scene.
Lines spoken while she is not on screen (off-screen voice, during a CG, H-scenes) get sprite=null.
This is a static sweep: if/else branches are walked in file order, so state can leak across a
branch edge.  It does not run the game's transition code.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fvp_hcb as E  # noqa: E402

# set from the profile's "sprites" block by configure()
F_SPRITE = F_HIDE = F_CG_SHOW = None
NAME_PREFIX = FACE_SUFFIX = None
ARG_CHAR = ARG_POSE = ARG_OUTFIT = None
SIZE_SUFFIX = {}
NO_SIZE_OUTFITS = ()
CG_IGNORE_PREFIX = None


def configure(prof):
    global F_SPRITE, F_HIDE, F_CG_SHOW, NAME_PREFIX, FACE_SUFFIX, ARG_CHAR, ARG_POSE, ARG_OUTFIT
    global SIZE_SUFFIX, NO_SIZE_OUTFITS, CG_IGNORE_PREFIX
    sp = prof['sprites']
    F_SPRITE, F_HIDE, F_CG_SHOW = (E._addr(sp[k]) for k in ('loader', 'hide', 'cg_show'))
    NAME_PREFIX, FACE_SUFFIX = sp['name_prefix'], sp['face_suffix']
    ARG_CHAR, ARG_POSE, ARG_OUTFIT = sp['arg_char'], sp['arg_pose'], sp['arg_outfit']
    SIZE_SUFFIX = {int(k): v for k, v in sp['size_suffix'].items()}
    NO_SIZE_OUTFITS = tuple(sp.get('no_size_outfits', ()))
    CG_IGNORE_PREFIX = sp.get('cg_ignore_prefix') or None


def sprite_tables(h):
    """Parse the loader: char id -> name, and per char {pose id -> name}, {outfit id -> name}."""
    body = [x[1] for x in h.body(F_SPRITE)]
    chars, poses, outfits = {}, {}, {}
    cur = None
    for k, (a, n, ar) in enumerate(body):
        if n == 'push_str' and ar[0].startswith(NAME_PREFIX):
            # nearest char compare before it:  push_stack <arg_char>; push_i8 K; sete; jz
            for j in range(k - 1, 0, -1):
                if body[j][1] == 'push_stack' and body[j][2][0] == ARG_CHAR and body[j + 1][1] == 'push_i8':
                    cur = body[j + 1][2][0]
                    break
            chars[cur] = ar[0][len(NAME_PREFIX):]
            poses[cur], outfits[cur] = {}, {}
        elif n == 'push_str' and cur is not None and k >= 6 and body[k - 1][1] == 'push_stack' \
                and body[k - 1][2][0] == 0:
            # ...push_stack <arg_pose>|<arg_outfit>; push_i8 V [push_i8 W; add]; sete; jz; push_stack 0; push_str S
            j = k - 4
            if body[k - 3][1] != 'sete' or body[j][1] not in ('add', 'push_i8'):
                continue  # size suffix etc.
            if body[j][1] == 'add':
                val = body[j - 2][2][0] + body[j - 1][2][0]
                arg = body[j - 3][2][0]
            else:
                val = body[j][2][0]
                arg = body[j - 1][2][0]
            if arg == ARG_POSE:
                poses[cur].setdefault(val, ar[0].lstrip('_'))
            elif arg == ARG_OUTFIT:
                outfits[cur].setdefault(val, ar[0].lstrip('_'))
        elif n == 'push_str' and ar[0] == FACE_SUFFIX:
            break
    return chars, poses, outfits


def wrappers(h):
    res = {}
    for fa in h.funcs:
        body = [x[1] for x in h.body(fa)]
        if len(body) == 16 and body[14][1] == 'call' and body[14][2][0] == F_SPRITE \
                and all(x[1].startswith('push_i') for x in body[1:5]):
            res[fa] = tuple(x[2][0] for x in body[1:5])  # (char, pose, outfit, face)
    return res


def cg_functions(h):
    res = {}
    for fa in h.funcs:
        body = [x[1] for x in h.body(fa)]
        if len(body) > 40:
            continue
        if any(x[1] == 'call' and x[2][0] == F_CG_SHOW for x in body):
            s = next((x[2][0] for x in body if x[1] == 'push_str'), None)
            if s:
                res[fa] = s
    return res



def load_emotions(path):
    m = {}
    if not os.path.exists(path):
        return m
    for line in open(path, encoding='utf-8-sig'):
        if line.startswith('#') or not line.strip():
            continue
        pose, expr, faces, desc, style = line.rstrip('\n').split('\t')
        for f in faces.split(','):
            m[(pose, int(f))] = (int(expr), desc, style)
    return m


def load_tsv(path, ncols):
    """Rows of a '#'-commented TSV, or [] when the file does not exist."""
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding='utf-8-sig'):
        if line.startswith('#') or not line.strip():
            continue
        rows.append((line.rstrip('\n').split('\t') + [''] * ncols)[:ncols])
    return rows


def write_summary(path, char, rows, emo_path, motions, styles):
    """Plain-text list of emotions (styles) and motions (poses) with line counts."""
    import collections
    faces = collections.defaultdict(list)  # style -> ['<pose> f12,13 description', ...]
    for pose, expr, fl, desc, style in load_tsv(emo_path, 5):
        faces[style].append('%s f%s  %s' % (pose, fl, desc))
    by_style = collections.Counter(r['style'] for r in rows if r['style'])
    by_hint = collections.Counter(r['face_style'] for r in rows if r['face_style'])
    by_pose = collections.Counter(r['pose'] for r in rows if r['sprite'])
    by_pose_outfit = collections.Counter((r['pose'], r['outfit']) for r in rows if r['sprite'])
    by_pose_size = collections.Counter((r['pose'], r['size']) for r in rows if r['sprite'])
    by_pose_hint = collections.Counter((r['pose'], r['face_style']) for r in rows if r['sprite'])
    n_lab = sum(by_style.values())
    L = ['%s: emotions (Style-Bert-VITS2 styles) and motions (sprite poses)' % char,
         'Generated by tools/fvp/voice_sprites.py from emotions/%s_styles.tsv, %s.tsv, %s_labels.tsv'
         ' and %s_motions.tsv.' % (char, char, char, char),
         'Edit those files and re-run; do not edit this file.',
         '',
         'A style is decided from three signals: 1) context of the line, 2) tone of voice, 3) face.',
         'The face is a guide only.  "labelled" = decided in %s_labels.tsv; "face hint" = what the' % char,
         'face alone would suggest (not used for training).',
         '%d voiced lines, %d labelled, %d not labelled yet.' % (len(rows), n_lab, len(rows) - n_lab),
         '',
         'The list is the 9 aichan emotions (Emotion enum, packages/stage-ui/src/constants/emotions.ts),',
         'always listed even with no fitting voice yet, plus extra voice styles aichan does not have yet',
         '(they emit their own name and are meant to be added to the Emotion enum).',
         'act = what chat training data emits: <|ACT {"emotion":{"name":<act>,"intensity":<i>}}|>.',
         '', '== Emotions ==', '']
    names = [s[0] for s in styles] + sorted((set(faces) | set(by_style)) - {s[0] for s in styles})
    cues = {s[0]: s for s in styles}
    for style in names:
        if style in cues:
            _, src, act, inten, ctx, voice, vstyle = cues[style]
            tag = 'aichan' if src == 'aichan' else 'extra, add "%s" to emotions.ts' % act
            if vstyle and vstyle != style:
                tag += ', voice trains as %s' % vstyle
            L.append('%-10s [%s, intensity %s]   labelled %4d   face hint %4d' % (
                style, tag, inten, by_style[style], by_hint[style]))
            L.append('    context: ' + ctx)
            L.append('    voice  : ' + voice)
        else:
            L.append('%-10s [NOT IN %s_styles.tsv]   labelled %4d   face hint %4d' % (
                style, char, by_style[style], by_hint[style]))
        L.append('    face   : ' + ('; '.join(faces.get(style, [])) or '(no face maps here)'))
        L.append('')
    L += ['== Motions ==', '',
          'act_motion = what chat training data emits: <|ACT {"motion":"<act_motion>"}|>.',
          'They are placeholders until replaced with the real motion files of the model.', '']
    for pose, motion, act_motion, desc in motions:
        L.append('%-16s (pose %s) %5d lines   act_motion %s' % (motion, pose, by_pose[pose], act_motion))
        L.append('    ' + desc)
        L.append('    outfit: ' + ', '.join('%s %d' % (o, n) for (p, o), n in sorted(by_pose_outfit.items()) if p == pose))
        L.append('    size  : ' + ', '.join('%s %d' % ({'N': 'normal', 'L': 'close-up'}.get(s, s), n)
                                            for (p, s), n in sorted(by_pose_size.items()) if p == pose))
        L.append('    face hints: ' + ', '.join('%s %d' % (s, n) for (p, s), n in
                                                sorted(by_pose_hint.items(), key=lambda x: -x[1]) if p == pose))
        L.append('')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(L))


def speech_text(t):
    """Text as spoken: drop the engine's outer 「」, keep inner punctuation."""
    if len(t) >= 2 and t[0] == '「' and t[-1] == '」':
        t = t[1:-1]
    return t


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    args = sys.argv[1:]
    opt = {'--char': None, '--emotions': 'emotions'}
    for k in list(opt):
        if k in args:
            i = args.index(k)
            opt[k] = args[i + 1]
            del args[i:i + 2]
    if len(args) != 3 or not opt['--char']:
        sys.exit(__doc__)
    prof_path, hcb, outdir = args
    char, emo_dir = opt['--char'], opt['--emotions']
    prof = E.use_profile(prof_path)
    configure(prof)
    h = E.HCB(hcb, prof.get('encoding', 'cp932'))
    ex = E.Extractor(h)
    chars, poses, outfits = sprite_tables(h)
    cid = next(k for k, v in chars.items() if v == char)
    wr = wrappers(h)
    cgs = cg_functions(h)
    emo_path = os.path.join(emo_dir, char + '.tsv')
    emo = load_emotions(emo_path)
    labels = {r[0]: r[1:] for r in load_tsv(os.path.join(emo_dir, char + '_labels.tsv'), 5)}
    # style, source, act_emotion, act_intensity, context cues, voice cues, voice_style
    styles = load_tsv(os.path.join(emo_dir, char + '_styles.tsv'), 7)
    act_of = {s[0]: (s[2], float(s[3])) for s in styles}
    voice_of = {s[0]: s[6] or s[0] for s in styles}
    # pose, motion, act_motion, description
    motions = load_tsv(os.path.join(emo_dir, char + '_motions.tsv'), 4)
    motion_of = {p: (m, am) for p, m, am, _ in motions}

    def hook(tgt, a, out):
        if tgt in wr:
            c, pose, outfit, face = wr[tgt]
            out.append(('x_sprite', c, pose, outfit, face, a[0] if a else None))
        elif tgt == F_HIDE:
            out.append(('x_hide', a[0] if a else None))
        elif tgt in cgs:
            out.append(('x_cg', cgs[tgt]))
    ex.on_call = hook

    rows = []
    size = 1
    for fa, fn, title in E.CHAPTERS:
        cur = None   # (pose, outfit, face)
        cg = None
        scene = None
        for r in ex.extract(fa):
            k = r[0]
            if k == 'scene':
                scene = '%s-%03d' % (fn, r[1])
                cur, cg = None, None
            elif k == 'comment' and (r[1].startswith('bg:') or r[1].startswith('screen:')):
                cur = None
                cg = None
            elif k == 'x_cg':
                cur = None
                cg = None if CG_IGNORE_PREFIX and r[1].startswith(CG_IGNORE_PREFIX) else r[1]
            elif k == 'x_hide' and r[1] in (cid, E.Unknown):
                cur = None
            elif k == 'x_sprite' and r[1] == cid:
                _, _, pose, outfit, face, sz = r
                if isinstance(sz, int) and not isinstance(sz, bool):
                    size = sz
                cur = (pose, outfit, face)
                cg = None
            elif k == 'line' and r[3] == char and r[5] is not None:
                _, disp, text, canon, alt, voice = r
                row = {'file': fn, 'scene': scene, 'voice': '%08d' % voice, 'name': disp,
                       'text': speech_text(text), 'cg': cg, 'sprite': None,
                       'pose': None, 'motion': None, 'outfit': None, 'face': None, 'size': None,
                       'expr': None, 'blush': None, 'emotion': None, 'face_style': None,
                       'style': None, 'basis': None, 'label_note': None,
                       'act_emotion': None, 'act_intensity': None, 'act_motion': None,
                       'voice_style': None}
                if cur:
                    pose, outfit, face = cur
                    pn = poses[cid].get(pose % 10, str(pose))
                    on = outfits[cid].get(outfit, str(outfit))
                    sfx = SIZE_SUFFIX.get(size, '')
                    if NO_SIZE_OUTFITS and on.endswith(NO_SIZE_OUTFITS):
                        sfx = ''
                    cp = NAME_PREFIX.split('/', 1)[-1]
                    sprite = '%s%s_%s_%s%s' % (cp, char, pn, on, sfx)
                    e = emo.get((pn, face))
                    row.update({'sprite': sprite, 'pose': pn, 'motion': motion_of.get(pn, (None,))[0],
                                'act_motion': motion_of.get(pn, (None, None))[1],
                                'outfit': on, 'face': face,
                                'size': sfx or 'N',
                                'image': 'sprites/%s/full/%s%s_%s_%s_f%02d.png' % (char, cp, char, pn, on, face),
                                'expr': e[0] if e else face // 2, 'blush': face % 2 == 1,
                                'emotion': e[1] if e else None, 'face_style': e[2] if e else None})
                if row['voice'] in labels:
                    row['style'], row['basis'], row['label_note'], inten = labels[row['voice']]
                    if row['style'] not in act_of:
                        sys.exit('label %s: style %r is not in %s_styles.tsv' % (row['voice'], row['style'], char))
                    row['act_emotion'], row['act_intensity'] = act_of[row['style']]
                    row['voice_style'] = voice_of[row['style']]
                    if inten:
                        row['act_intensity'] = float(inten)
                rows.append(row)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, '%s_voice_sprites' % char)
    with open(base + '.jsonl', 'w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    cols = ['file', 'scene', 'voice', 'style', 'voice_style', 'basis', 'act_emotion', 'act_intensity', 'act_motion',
            'face_style', 'emotion', 'motion', 'pose', 'outfit', 'face', 'blush', 'size', 'cg', 'text']
    with open(base + '.tsv', 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(cols) + '\n')
        for row in rows:
            f.write('\t'.join('' if row.get(c) is None else str(row.get(c)) for c in cols) + '\n')
    write_summary(os.path.join(outdir, '%s_emotions_motions.txt' % char), char, rows, emo_path, motions,
                  styles)
    import collections
    n = len(rows)
    ws = sum(1 for r in rows if r['sprite'])
    wc = sum(1 for r in rows if not r['sprite'] and r['cg'])
    print('%s: %d voiced lines, %d with sprite, %d during event CG, %d none' % (char, n, ws, wc, n - ws - wc))
    print('labelled:', sum(1 for r in rows if r['style']), 'of', len(rows),
          dict(collections.Counter(r['style'] for r in rows if r['style']).most_common()))
    print('face hints:', dict(collections.Counter(r['face_style'] for r in rows if r['face_style']).most_common()))
    print('poses :', {chars[cid]: poses[cid]}, 'outfits:', outfits[cid])


if __name__ == '__main__':
    main()
