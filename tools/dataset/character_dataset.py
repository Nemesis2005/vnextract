#!/usr/bin/env python3
"""Build a character-roleplay chat dataset (one character is the assistant) from an extracted script.

The script must be in the normalized script format (repository README, "Normalized script format"),
e.g. the output of tools/fvp/fvp_hcb.py.  Everything about the game and
the character (names, persona card, routes, merges, spans, overrides, route context) lives in a
per-character config module (fields: see the repository README, "Character config").

Usage (Python 3.8+; stdlib + Jinja2 for exact Qwen3.5 validation):
    python -I tools/dataset/character_dataset.py --config CONFIG.py --project DIR [--tokenizer DIR] COMMAND
      plan         segment scenes -> <project>/dataset/work/plan.json + annotator worksheet
      build        full rebuild: plan + annotations -> <project>/dataset/<OUTPUT>   (default command)
      append STEM  append only new docs from script file STEM; existing JSONL bytes are preserved
      validate     structural checks + stats (also run by build/append)

  --config     the character's config module (a .py file).  Its folder must hold annotations/
               (or set ANN_DIR in the config).
  --project    the extraction folder: reads <project>/script/*.txt, writes <project>/dataset/.
  --tokenizer  Qwen3.5 tokenizer folder (tokenizer.json + chat_template.jinja); default
               tools/dataset/qwen35_9b_tokenizer.

Rules (see the character's DATASET.md): narration stays in script order and goes to the role of its
narrator.  The protagonist's narration is `<narration>` in the user turn; the target's own visible actions
are `<narration>` in their turn, in first person; in target-POV scenes their thoughts and solitary actions
are `<think>`.  Omniscient narration and scene notes are folded into user `<narration>` as a final
assembly step, leaving only the opening setup as a system message.  Roles: system first, then user; last
message is an assistant turn with speech; two user (or two assistant) messages are never directly adjacent.
"""
import json
import os
import re
import sys
import glob
import itertools
import collections
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROLES_FILE = 'narration_roles.json'   # per-line narration roles; not a per-document annotation file

sys.path.insert(0, HERE)
import qwen35_validation as QV  # noqa: E402

# Set by setup() from the command line and the config module.
CFG = None
SCRIPT_DIR = OUT_DIR = WORK_DIR = ANN_DIR = OUTPUT = None
TOKENIZER_DIR = None
TARGET = None              # the assistant character's canonical name (script `char` field)
PROTAGONIST = None         # the default narrator (the user side)
FIRST_PERSON = None        # the target's first-person pronoun used when their narration is moved into their turn
NARRATOR_PRONOUNS = None   # regex: the narrator's first-person pronouns
ROLE_KEY = None            # key of the target's lines in narration_roles.json
COND_GLOBAL = None         # the choice-result global whose `== N` conditions are enumerated per value
CUTAWAY_BG = set()
UNKNOWN_PLATES = set()

# ---------------------------------------------------------------- tunables (a config may override each)
# Keep them fixed once annotation has started: document ids (and the annotation keys) depend on them.
# Fragments that belong together are joined afterwards by CFG.MERGE_GROUPS.
GAP_SPLIT = 1500      # non-target material (chars) between two target lines that starts a new document
LEAD_CHARS = 600      # how much material before the target's first line to keep as the opening
LEAD_MAX = 1400       # hard cap for the lead-in, even when looking for a user line
PART_BUDGET = 2900    # max estimated dialogue+narration chars per document (persona/situation come on top)
NARR_CONDENSE = 260   # narration stretches longer than this are flagged in the worksheets
NARR_EST = 160        # assumed length of a condensed stretch when budgeting
TOKEN_LIMIT = QV.TOKEN_LIMIT  # actual Qwen3.5-9B tokens, including conservative source-preserving rendering
TUNABLES = ('GAP_SPLIT', 'LEAD_CHARS', 'LEAD_MAX', 'PART_BUDGET', 'NARR_CONDENSE', 'NARR_EST', 'TOKEN_LIMIT')


def setup(config_path, project, tokenizer=None):
    global CFG, SCRIPT_DIR, OUT_DIR, WORK_DIR, ANN_DIR, OUTPUT, TOKENIZER_DIR, TARGET, PROTAGONIST
    global FIRST_PERSON, NARRATOR_PRONOUNS, ROLE_KEY, COND_GLOBAL, CUTAWAY_BG, UNKNOWN_PLATES
    config_path = os.path.abspath(config_path)
    cdir = os.path.dirname(config_path)
    sys.path.insert(0, cdir)   # the config may import its own helper modules (e.g. route_context.py)
    spec = importlib.util.spec_from_file_location('character_config', config_path)
    CFG = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(CFG)
    SCRIPT_DIR = os.path.join(project, 'script')
    OUT_DIR = os.path.join(project, 'dataset')
    WORK_DIR = os.path.join(OUT_DIR, 'work')
    ANN_DIR = getattr(CFG, 'ANN_DIR', None) or os.path.join(cdir, 'annotations')
    OUTPUT = CFG.OUTPUT
    TOKENIZER_DIR = tokenizer
    TARGET, PROTAGONIST = CFG.TARGET, CFG.PROTAGONIST
    FIRST_PERSON = CFG.FIRST_PERSON
    NARRATOR_PRONOUNS = CFG.NARRATOR_PRONOUNS
    ROLE_KEY = getattr(CFG, 'ROLE_KEY', 'target')
    COND_GLOBAL = getattr(CFG, 'COND_GLOBAL', None)
    CUTAWAY_BG = set(getattr(CFG, 'CUTAWAY_BG', ()))
    UNKNOWN_PLATES = set(getattr(CFG, 'UNKNOWN_PLATES', {'？？？'}))
    g = globals()
    for k in TUNABLES:
        if hasattr(CFG, k):
            g[k] = getattr(CFG, k)


def route_of(stem):
    return CFG.route_of(stem)


def include_file(stem):
    f = getattr(CFG, 'include_file', None)
    return f(stem) if f else True


# ---------------------------------------------------------------- parsing
SPEECH_RE = re.compile(r'^【(【[^】]*】|[^】]*)】(.*?)(?:  # (.*))?$')


def clean_plate(p):
    return p.replace('【', '').replace('】', '')


def parse_file(path):
    """Yield items: dict(kind, line, ...). kind in scene/bg/choice/branch/if/else/endif/call/U/S/N."""
    stem = os.path.splitext(os.path.basename(path))[0]
    items = []
    with open(path, encoding='utf-8') as f:
        for ln, raw in enumerate(f, 1):
            l = raw.rstrip('\n')
            if not l.strip():
                continue
            m = re.match(r'^=== scene: (.+) ===$', l)
            if m:
                items.append(dict(kind='scene', line=ln, scene=m.group(1)))
                continue
            if l.startswith('#'):
                body = l[1:].strip()
                if body.startswith('bg:'):
                    items.append(dict(kind='bg', line=ln, text=body[3:].strip()))
                elif body.startswith('choice:'):
                    items.append(dict(kind='choice', line=ln, text=body[7:].strip()))
                elif body.startswith('branch:'):
                    items.append(dict(kind='branch', line=ln, text=body.split('→', 1)[-1].strip()))
                elif body.startswith('if '):
                    items.append(dict(kind='if', line=ln, text=body[3:].strip()))
                elif body == 'else':
                    items.append(dict(kind='else', line=ln))
                elif body == 'endif':
                    items.append(dict(kind='endif', line=ln))
                elif body.startswith('call:'):
                    items.append(dict(kind='call', line=ln, text=body[5:].strip()))
                # bgm / screen / header / source lines are ignored
                continue
            m = SPEECH_RE.match(l)
            if m and l.startswith('【'):
                plate = clean_plate(m.group(1))
                text = m.group(2).strip()
                meta = {}
                if m.group(3):
                    for kv in m.group(3).split():
                        if '=' in kv:
                            k, v = kv.split('=', 1)
                            meta[k] = v
                char = meta.get('char', plate)
                kind = 'S' if char == TARGET else 'U'
                items.append(dict(kind=kind, line=ln, plate=plate, char=char, alt=meta.get('alt', ''), text=text))
                continue
            items.append(dict(kind='N', line=ln, text=l.strip()))
    return stem, items


TRIVIAL_RE = re.compile(r'^[\s…‥・。、―ー－\-～〜！？!?\.]*$')


def is_trivial_narr(t):
    return bool(TRIVIAL_RE.match(t))


def target_trivial(t):
    core = re.sub(r'[「」『』（）\s…‥・。、―ー－～〜！？!?\.っッ]', '', t)
    return len(core) < 3


# ---------------------------------------------------------------- scenes and branch paths
def split_scenes(stem, items):
    scenes = []
    cur = None
    for it in items:
        if it['kind'] == 'scene':
            cur = dict(scene=it['scene'], start=it['line'], items=[])
            scenes.append(cur)
        elif cur is not None:
            cur['items'].append(it)
    for s in scenes:
        s['end'] = s['items'][-1]['line'] if s['items'] else s['start']
    return scenes


def cond_key(c):
    """Normalise a raw condition on COND_GLOBAL (say g10): '((g10 == 3) || (g11 == 1))' -> ('g10', 3).
    Other terms in it (e.g. a debug/all flag) are assumed false.  Anything else stays ('raw', c)."""
    if COND_GLOBAL:
        m = re.match(r'^\(*\s*(%s) == (\d+)\)' % re.escape(COND_GLOBAL), c)
        if m:
            return (COND_GLOBAL, int(m.group(2)))
    return ('raw', c)


def enumerate_paths(scene_items, open_stack):
    """Split a scene into pieces at choice/branch markers, and within each piece enumerate if/else paths.

    Returns (list of dict(branch, items), new_open_stack).
    open_stack: list of [cond_key, in_else] blocks still open at scene start (blocks spanning scenes).
    """
    pieces = [[]]
    labels = ['']
    for it in scene_items:
        if it['kind'] == 'choice':
            pieces.append([])
            labels.append(labels[-1])
            continue
        if it['kind'] == 'branch':
            pieces.append([])
            labels.append('choice→' + it['text'])
            continue
        pieces[-1].append(it)
    results = []
    stack = [list(b) for b in open_stack]
    for pidx, (lab, piece) in enumerate(zip(labels, pieces)):
        conds = []
        for b in stack:
            if b[0] not in conds:
                conds.append(b[0])
        for it in piece:
            if it['kind'] == 'if':
                k = cond_key(it['text'])
                if k not in conds:
                    conds.append(k)
        g_vals = sorted({k[1] for k in conds if k[0] == COND_GLOBAL})
        raw = [k for k in conds if k[0] == 'raw']
        g_opts = [None] + g_vals if g_vals else [None]
        assigns = []
        for gv in g_opts:
            for rv in itertools.product([False, True], repeat=len(raw)):
                a = {}
                for k in conds:
                    if k[0] == COND_GLOBAL:
                        a[k] = (gv == k[1])
                for k, v in zip(raw, rv):
                    a[k] = v
                assigns.append(a)
        end_stack = None
        for a in assigns:
            st = [list(b) for b in stack]
            out = []
            for it in piece:
                if it['kind'] == 'if':
                    st.append([cond_key(it['text']), False])
                    continue
                if it['kind'] == 'else':
                    if st:
                        st[-1][1] = True
                    continue
                if it['kind'] == 'endif':
                    if st:
                        st.pop()
                    continue
                active = all((a.get(b[0], False) != b[1]) for b in st)
                if active:
                    it2 = dict(it)
                    it2['ctx'] = tuple((b[0], b[1]) for b in st)
                    out.append(it2)
            end_stack = st
            results.append(((pidx, lab), dict(a), out))
        stack = end_stack if end_stack is not None else stack
    sigs = []
    for lab, used, out in results:
        out = [i for i in out if i['kind'] != 'call']
        if not any(i['kind'] in ('U', 'S', 'N') for i in out):
            continue
        sigs.append((lab, used, out, frozenset(i['line'] for i in out if i['kind'] == 'S')))
    final = []
    for i, (lab, used, out, sh) in enumerate(sigs):
        dominated = any(lab2 == lab and (sh < sh2 or (sh == sh2 and j < i))
                        for j, (lab2, _, _, sh2) in enumerate(sigs) if j != i)
        if dominated:
            continue
        final.append(dict(branch=lab[1], items=out))
    return final, stack


# ---------------------------------------------------------------- clustering / parts
def mat_len(it):
    if it['kind'] in ('U', 'S'):
        return len(it['text']) + len(it.get('plate', '')) + 1
    if it['kind'] == 'N':
        return 0 if is_trivial_narr(it['text']) else len(it['text'])
    return 0


def est_len(items):
    """Estimated rendered length: long narration stretches count as condensed."""
    tot = 0
    run = 0
    for it in items:
        if it['kind'] == 'N':
            run += mat_len(it)
            continue
        if run:
            tot += run if run <= NARR_CONDENSE else NARR_EST
            run = 0
        tot += mat_len(it)
    if run:
        tot += run if run <= NARR_CONDENSE else NARR_EST
    return tot


def make_parts(path_items):
    """Return list of (start_idx, end_idx_inclusive) into path_items, each ending on an S item."""
    its = path_items
    s_idx = [i for i, it in enumerate(its) if it['kind'] == 'S']
    if not s_idx:
        return []
    clusters = [[s_idx[0]]]
    for a, b in zip(s_idx, s_idx[1:]):
        gap = sum(mat_len(it) for it in its[a + 1:b])
        if gap > GAP_SPLIT:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    parts = []
    prev_end = -1
    for cl in clusters:
        first, last = cl[0], cl[-1]
        start = first
        acc = 0
        seen_u = False
        j = first - 1
        while j > prev_end:
            acc += mat_len(its[j])
            if its[j]['kind'] == 'U':
                seen_u = True
            if acc > LEAD_MAX:
                break
            if acc > LEAD_CHARS and seen_u:
                start = j
                break
            start = j
            j -= 1
        cur_start = start
        while True:
            seg_end = last
            if est_len(its[cur_start:seg_end + 1]) <= PART_BUDGET:
                parts.append((cur_start, seg_end))
                break
            best = None
            for e in cl:
                if e < cur_start:
                    continue
                if est_len(its[cur_start:e + 1]) > PART_BUDGET:
                    break
                nxt = [k for k in cl if k > e]
                if not nxt:
                    break
                between = its[e + 1:nxt[0]]
                if any(x['kind'] == 'U' for x in between) and any(x['kind'] == 'S' for x in its[cur_start:e + 1]):
                    best = e
            if best is None:
                for e in cl:
                    if e < cur_start:
                        continue
                    nxt = [k for k in cl if k > e]
                    if not nxt:
                        break
                    between = its[e + 1:nxt[0]]
                    if any(x['kind'] == 'U' for x in between) and est_len(its[cur_start:e + 1]) > PART_BUDGET * 0.4:
                        best = e
                        break
            if best is None:
                parts.append((cur_start, seg_end))
                break
            parts.append((cur_start, best))
            cur_start = best + 1
        prev_end = last
    return parts


SCENE_PATHS = {}   # scene id -> list of path item lists (U/S/N/bg only)
SCENE_ORDER = {}   # stem -> [scene ids in file order]
SCENE_RAW = {}     # scene id -> raw items (for bg-only scenes)


def build_plan():
    files = sorted(glob.glob(os.path.join(SCRIPT_DIR, '*.txt')))
    plan = []
    for path in files:
        stem, items = parse_file(path)
        if not include_file(stem):
            continue
        scenes = split_scenes(stem, items)
        SCENE_ORDER[stem] = [s['scene'] for s in scenes]
        open_stack = []
        bg = None
        for sc in scenes:
            bg_at_start = bg
            for it in sc['items']:
                if it['kind'] == 'bg' and it['text'] not in CUTAWAY_BG:
                    bg = it['text']
            paths, open_stack = enumerate_paths(sc['items'], open_stack)
            SCENE_PATHS[sc['scene']] = [[x for x in p['items'] if x['kind'] in ('U', 'S', 'N', 'bg')] for p in paths]
            SCENE_RAW[sc['scene']] = sc['items']
            n_part = 0
            seen_sh = []
            for pi, p in enumerate(paths):
                its = p['items']
                parts = make_parts(its)
                for a, b in parts:
                    seg = its[a:b + 1]
                    sh = [x for x in seg if x['kind'] == 'S']
                    if all(target_trivial(x['text']) for x in sh):
                        continue
                    shset = frozenset(x['line'] for x in sh)
                    if any(shset <= s for s in seen_sh):
                        continue
                    seen_sh.append(shset)
                    pbg = bg_at_start
                    for x in its[:a]:
                        if x['kind'] == 'bg' and x['text'] not in CUTAWAY_BG:
                            pbg = x['text']
                    ctx = set()
                    for x in seg:
                        ctx.update(x.get('ctx', ()))
                    cond = []
                    for (k, in_else) in sorted(ctx, key=str):
                        expr = ('%s == %d' % k) if k[0] == COND_GLOBAL else k[1].strip('()')
                        cond.append(('not (%s)' % expr) if in_else else expr)
                    n_part += 1
                    did = '%s-%s' % (sc['scene'], n_part)
                    plan.append(dict(
                        id=did,
                        source_file=stem, scene=sc['scene'], scenes=[sc['scene']], path=pi, route=route_of(stem),
                        branch=p['branch'] or CFG.BRANCH_SCENES.get(sc['scene'], ''),
                        cond='; '.join(cond),
                        bg=pbg,
                        scene_speakers=sorted({x['plate'] for x in its if x['kind'] in ('U', 'S')}),
                        lines=[x['line'] for x in seg if x['kind'] in ('U', 'S', 'N', 'bg')],
                        prev_lines=[x['line'] for x in its[:a]],
                        scene_first=all(x['kind'] == 'bg' for x in its[:a]),
                    ))
    return apply_merges(plan)


def stream(stem, lo, hi, prefer):
    """All U/S/N/bg items with lo <= line <= hi, following one branch path per scene."""
    out = []
    for sc in SCENE_ORDER[stem]:
        paths = SCENE_PATHS.get(sc) or []
        raw = SCENE_RAW.get(sc) or []
        if not raw or raw[-1]['line'] < lo or raw[0]['line'] > hi:
            continue
        if not paths:   # bg-only scene
            out += [x for x in raw if x['kind'] == 'bg' and lo <= x['line'] <= hi]
            continue
        if sc in prefer:
            idx = prefer[sc]
        elif len(paths) == 1:
            idx = 0
        elif sc in CFG.PATH_CHOICE:
            idx = CFG.PATH_CHOICE[sc]
        else:
            raise SystemExit('stream %s %d-%d: scene %s has %d paths; set CFG.PATH_CHOICE' % (stem, lo, hi, sc, len(paths)))
        out += [x for x in paths[idx] if lo <= x['line'] <= hi]
    return out


def apply_merges(plan):
    """Apply CFG.SPANS (move a document's start/end) and CFG.MERGE_GROUPS (join adjacent fragments)."""
    by_id = {d['id']: d for d in plan}
    for gid, sp in CFG.SPANS.items():
        if gid not in by_id:
            raise SystemExit('SPANS: unknown document %s' % gid)
    gone = set()
    for grp in CFG.MERGE_GROUPS:
        for g in grp:
            if g not in by_id:
                raise SystemExit('MERGE_GROUPS: unknown document %s' % g)
        ds = [by_id[g] for g in grp]
        first = ds[0]
        stems = {d['source_file'] for d in ds}
        if len(stems) != 1:
            raise SystemExit('MERGE_GROUPS: %s spans files' % grp)
        prefer = {}
        for d in ds:
            prefer.setdefault(d['scene'], d['path'])
        lo = min(first['lines'])
        hi = max(ds[-1]['lines'])
        items = stream(first['source_file'], lo, hi, prefer)
        first['lines'] = [x['line'] for x in items]
        first['scenes'] = sorted({s for d in ds for s in d['scenes']} | {sc for sc in SCENE_ORDER[first['source_file']]
                                                                           if SCENE_RAW.get(sc) and lo <= SCENE_RAW[sc][0]['line'] <= hi})
        first['merged_from'] = list(grp)
        first['scene_speakers'] = sorted({s for d in ds for s in d['scene_speakers']})
        conds = [d['cond'] for d in ds if d['cond']]
        first['cond'] = '; '.join(dict.fromkeys('; '.join(conds).split('; '))) if conds else ''
        for d in ds[1:]:
            gone.add(d['id'])
    plan = [d for d in plan if d['id'] not in gone]
    for d in plan:
        sp = CFG.SPANS.get(d['id'])
        if not sp:
            continue
        lo = sp.get('start', min(d['lines']))
        hi = sp.get('end', max(d['lines']))
        prefer = {d['scene']: d['path']}
        items = stream(d['source_file'], lo, hi, prefer)
        d['lines'] = [x['line'] for x in items]
        d['scenes'] = sorted(set(d['scenes']) | {sc for sc in SCENE_ORDER[d['source_file']]
                                                 if SCENE_RAW.get(sc) and lo <= SCENE_RAW[sc][-1]['line'] and SCENE_RAW[sc][0]['line'] <= hi})
        d['span'] = sp
        if 'start' in sp:
            d['prev_lines'] = [l for l in d['prev_lines'] if l < lo]
    # target-POV documents keep the rest of their last scene: thoughts after the last target line are not dropped
    # but carried into the next POV document's first <think> (see cmd_build)
    for d in plan:
        if d['scene'] not in CFG.POV_SCENES or 'end' in CFG.SPANS.get(d['id'], {}):
            continue
        last = max(d['lines'])
        sc = next(s for s in reversed(SCENE_ORDER[d['source_file']]) if SCENE_RAW.get(s) and SCENE_RAW[s][0]['line'] <= last)
        tail = [x['line'] for x in SCENE_PATHS[sc][0] if x['line'] > last]
        d['lines'] = d['lines'] + tail
    return plan


# ---------------------------------------------------------------- item lookup
_CACHE = {}
_ORDER = {}


def file_items(stem):
    if stem not in _CACHE:
        _, items = parse_file(os.path.join(SCRIPT_DIR, stem + '.txt'))
        _CACHE[stem] = {it['line']: it for it in items}
        _ORDER[stem] = items
    return _CACHE[stem]


def file_order(stem):
    file_items(stem)
    return _ORDER[stem]


def inherited_bg(stem, line):
    """Latest non-cutaway background set before `line` in the file."""
    bg = None
    for x in file_order(stem):
        if x['line'] >= line:
            break
        if x['kind'] == 'bg' and x['text'] not in CUTAWAY_BG:
            bg = x['text']
    return bg


def narr_stretches(items):
    out = []
    cur = []
    for it in items:
        if it['kind'] == 'N':
            if not is_trivial_narr(it['text']):
                cur.append(it)
        elif it['kind'] == 'bg':
            continue
        else:
            if cur:
                out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------- worksheets for the judgment pass
def render_item(it, stretch_tag=None):
    if it['kind'] == 'U':
        return '%d U %s：%s' % (it['line'], it['plate'], it['text'])
    if it['kind'] == 'S':
        return '%d S %s：%s' % (it['line'], TARGET, it['text'])
    if it['kind'] == 'N':
        return '%d N %s%s' % (it['line'], ('[%s] ' % stretch_tag) if stretch_tag else '', it['text'])
    if it['kind'] == 'bg':
        return '%d BG %s' % (it['line'], it['text'])
    return None


def preceding_context(d, limit=1300):
    its = file_items(d['source_file'])
    out = []
    if d['prev_lines']:
        out = [its[l] for l in d['prev_lines']]
    total = sum(mat_len(x) for x in out)
    if total < limit and d['scene_first']:
        anchor_scene = CFG.CONTEXT_FROM.get(d['scene'])
        allits = file_order(d['source_file'])
        start_line = min(d['lines'] + d['prev_lines'])
        if anchor_scene:
            idx = [k for k, x in enumerate(allits) if x['kind'] == 'scene' and x['scene'] == anchor_scene]
            nxt = [k for k, x in enumerate(allits) if x['kind'] == 'scene' and k > idx[0]]
            start_line = allits[nxt[0]]['line'] if nxt else start_line
        before = [x for x in allits if x['line'] < start_line and x['kind'] in ('U', 'S', 'N', 'bg')]
        extra = []
        for x in reversed(before):
            extra.append(x)
            total += mat_len(x)
            if total >= limit:
                break
        out = list(reversed(extra)) + out
    else:
        keep = []
        tot = 0
        for x in reversed(out):
            keep.append(x)
            tot += mat_len(x)
            if tot >= limit:
                break
        out = list(reversed(keep))
    return out


def write_worksheets(plan, batch_chars=95000):
    os.makedirs(WORK_DIR, exist_ok=True)
    blocks = []
    for d in plan:
        its = file_items(d['source_file'])
        seg = [its[l] for l in d['lines']]
        hdr = '##### DOC %s\nfile: %s | route: %s | bg at start: %s | branch: %s | cond: %s | merged: %s\n' % (
            d['id'], d['source_file'], d['route'], d['bg'] or '-', d['branch'] or '-', d['cond'] or '-',
            ','.join(d.get('merged_from', [])) or '-')
        ctx = preceding_context(d)
        body = ['-- preceding context (NOT in the document; use it for the recap) --']
        body += [render_item(x) for x in ctx if render_item(x)]
        body += ['-- document content --']
        body += [render_item(x) for x in seg if render_item(x)]
        blocks.append((d, hdr + '\n'.join(body) + '\n'))
    with open(os.path.join(WORK_DIR, 'sheet_all.txt'), 'w', encoding='utf-8') as f:
        for d, b in blocks:
            f.write(b + '\n')
    print('worksheet: %d docs -> work/sheet_all.txt' % len(blocks))


# ---------------------------------------------------------------- build
def load_annotations():
    ann = {}
    for p in sorted(glob.glob(os.path.join(ANN_DIR, '*.json'))):
        if os.path.basename(p) == ROLES_FILE:
            continue
        with open(p, encoding='utf-8') as f:
            for k, v in json.load(f).items():
                if k.startswith('_'):
                    continue
                if k in ann:
                    raise SystemExit('duplicate annotation for %s in %s' % (k, p))
                ann[k] = v
    return ann


def is_bracketed_thought(t):
    return t.startswith('（') and t.endswith('）')


def _intkeys(d):
    return {int(k): v for k, v in (d or {}).items()}


def scene_num(scene):
    return int(scene.rsplit('-', 1)[1])


_ROLES = None


def load_roles():
    """annotations/narration_roles.json: per script file, the narration lines that are the target's own
    visible actions (key ROLE_KEY, with logged "rewrite"s) or omniscient ("system"); all others are the
    narrator's ("user")."""
    global _ROLES
    if _ROLES is None:
        with open(os.path.join(ANN_DIR, ROLES_FILE), encoding='utf-8') as f:
            raw = json.load(f)
        _ROLES = {}
        for stem, v in raw.items():
            if stem.startswith('_'):
                continue
            _ROLES[stem] = dict(target=set(v.get(ROLE_KEY, [])), system=set(v.get('system', [])),
                                rewrite={int(k): t for k, t in v.get('rewrite', {}).items()})
    return _ROLES


def first_person(text, stem, line):
    """Mechanical rewrite of a narration line about the target into their first person:
    TARGET → FIRST_PERSON, and the narrator's pronouns (NARRATOR_PRONOUNS) → CFG.narrator_name(stem, line)."""
    who = CFG.narrator_name(stem, line)
    return re.sub(NARRATOR_PRONOUNS, who, text.replace(TARGET, FIRST_PERSON))


def target_narration(text, stem, line):
    r = load_roles().get(stem)
    if r and line in r['rewrite']:
        return r['rewrite'][line]
    return first_person(text, stem, line)


def narration_desc(pov, narrator):
    """The `地の文：` line of the scene block; written by the config in the dataset's language."""
    return CFG.narration_desc(pov, narrator)


def render(m):
    out = []
    for kind, val in m['parts']:
        if kind == 'n':
            out.append('<narration>%s</narration>' % '\n'.join(val))
        else:
            out.append(val)
    content = '\n'.join(out)
    if m['role'] == 'assistant' and m['think']:
        if content:
            content = '<think>\n%s\n</think>\n\n%s' % ('\n'.join(m['think']), content)
        else:   # a think-only beat (annotation ``think_break``)
            content = '<think>\n%s\n</think>' % '\n'.join(m['think'])
    return content


def fold_system_into_user(messages):
    """Keep only the opening system message; fold later system text into user narration.

    Converted text remains in script order. Any resulting adjacent user turns are joined,
    which also creates a real user turn for formerly assistant-only POV documents.
    """
    out = []
    for i, message in enumerate(messages):
        m = dict(message)
        if i and m['role'] == 'system':
            m = dict(role='user', content='<narration>%s</narration>' % m['content'])
        if m['role'] == 'user' and out and out[-1]['role'] == 'user':
            out[-1]['content'] += '\n' + m['content']
        else:
            out.append(m)
    return out


def assemble(d, a, member_anns, carry_in):
    """Turn one planned document + its annotation into (messages, meta, trailing_thoughts, counts) or None."""
    stem = d['source_file']
    its = file_items(stem)
    seg = [its[l] for l in d['lines']]
    pov = any(s in CFG.POV_SCENES for s in d['scenes'])
    roles = load_roles().get(stem, dict(target=set(), system=set(), rewrite={}))
    drop_lines = set(a.get('drop', []))
    drop_bg = set(a.get('drop_bg', []))
    relabel = _intkeys(a.get('relabel'))
    insert_after = _intkeys(a.get('insert_after'))
    insert_before = _intkeys(a.get('insert_before'))
    think_break = set(a.get('think_break', []))
    notes_at = {}     # line -> (group index)
    note_groups = []
    for g in a.get('as_note', []):
        note_groups.append(g['text'])
        for l in g['lines']:
            notes_at[l] = len(note_groups) - 1
    cranges = [(r['from'], r['to'], r['text']) for r in a.get('condense_ranges', [])]
    if pov:
        sit = set(a.get('situational', []))
        own = set(a.get('own_action', []))
        for m in member_anns:
            sit |= set(m.get('situational', []))
            own |= set(m.get('own_action', []))
    # ---- location of the header
    first_line = min(d['lines'])
    loc = inherited_bg(stem, first_line)
    first_s = next((x['line'] for x in seg if x['kind'] == 'S'), None)
    header_bg_line = None
    for x in seg:
        if x['kind'] in ('U', 'S'):
            break
        if x['kind'] == 'bg' and x['text'] not in CUTAWAY_BG and x['line'] not in drop_bg:
            loc = x['text']
            header_bg_line = x['line']
    if header_bg_line is None and d.get('scene_first', True) and first_s is not None:
        for x in seg:
            if x['line'] >= first_s:
                break
            if x['kind'] == 'bg' and x['text'] not in CUTAWAY_BG and x['line'] not in drop_bg:
                loc = x['text']
                header_bg_line = x['line']
                break
    if not loc:
        loc = next((x['text'] for x in seg if x['kind'] == 'bg' and x['text'] not in CUTAWAY_BG), None)
    loc = a.get('location') or CFG.LOCATION_OVERRIDES.get(d['id'], loc)
    # ---- events in script order
    # sys: system note / omniscient narration; U, S: speech; UN: the narrator's narration (user turn);
    # AN: the target's own visible action (their turn); T: the target's thought (POV only, <think>)
    events = []
    cur_loc = loc
    started = False
    done_ranges = set()
    done_notes = set()
    for x in seg:
        k = x['kind']
        L = x['line']
        if L in insert_before:
            events.append(('sys', insert_before[L]))
        rng = next((r for r in cranges if r[0] <= L <= r[1]), None)
        if rng is not None:
            if k == 'S':
                raise SystemExit('%s: condense range %d-%d covers a %s line %d' % (d['id'], rng[0], rng[1], TARGET, L))
            if rng not in done_ranges:
                done_ranges.add(rng)
                events.append(('sys', rng[2]))
            if k == 'bg' and x['text'] not in CUTAWAY_BG and L not in drop_bg:
                cur_loc = x['text']
            continue
        if L in notes_at:
            gi = notes_at[L]
            if gi not in done_notes:
                done_notes.add(gi)
                events.append(('sys', note_groups[gi]))
        elif L in drop_lines:
            pass
        elif k == 'bg':
            if L == header_bg_line:
                cur_loc = x['text']
            elif x['text'] in CUTAWAY_BG or L in drop_bg:
                pass
            else:
                if started and x['text'] != cur_loc:
                    events.append(('sys', '（場所：%s）' % x['text']))
                cur_loc = x['text']
        elif k == 'N':
            if not is_trivial_narr(x['text']):
                if pov:
                    if L in own:
                        events.append(('AN', x['text']))
                    elif L in sit:
                        events.append(('sys', x['text']))
                    else:
                        events.append(('T', x['text']))
                elif L in roles['system']:
                    events.append(('sys', x['text']))
                elif L in roles['target']:
                    events.append(('AN', target_narration(x['text'], stem, L), x['text']))
                else:
                    events.append(('UN', x['text']))
        elif k in ('U', 'S'):
            started = True
            if k == 'S' and pov and is_bracketed_thought(x['text']) and d['id'] not in CFG.POV_BRACKET_KEEP:
                # in target-POV scenes their plated （…） lines are private thoughts -> <think> (verbatim)
                events.append(('T', x['text']))
            elif k == 'S':
                events.append(('S', x['text']))
            else:
                events.append(('U', '%s：%s' % (relabel.get(L, x['plate']), x['text'])))
        if L in insert_after:
            events.append(('sys', insert_after[L]))
        if L in think_break:
            events.append(('TB', ''))
    if not any(e[0] == 'S' for e in events):
        return None
    # ---- messages: narration joins the turn of its role, so same-role turns are no longer split by it
    # target actions narrated before anyone (the protagonist included) has said or done anything in the document
    # stay in the header's opening narration, as script text; a conversation cannot open with the target's narration
    while True:
        first = next(i for i, e in enumerate(events) if e[0] in ('U', 'UN', 'S', 'AN'))
        if events[first][0] == 'AN' and not pov:
            events[first] = ('sys', events[first][2])
            continue
        break
    events = [e[:2] for e in events]
    opening_narr = [e[1] for e in events[:first] if e[0] == 'sys']
    pend_think = list(carry_in) + [e[1] for e in events[:first] if e[0] == 'T']
    msgs = []        # dict(role, parts=[('n', [lines]) | ('l', text)], think)
    pend_sys = []
    cur = None
    for kind, txt in events[first:]:
        if kind == 'sys':
            pend_sys.append(txt)
            cur = None
            continue
        if kind == 'T':
            pend_think.append(txt)
            continue
        if kind == 'TB':
            # think_break: the thoughts so far form their own beat here instead of waiting for her next line
            if pend_think:
                if cur is not None and cur['role'] == 'assistant':
                    cur['think'] += pend_think
                else:
                    if pend_sys:
                        msgs.append(dict(role='system', parts=[('l', t) for t in pend_sys], think=[]))
                        pend_sys = []
                    cur = dict(role='assistant', parts=[], think=list(pend_think))
                    msgs.append(cur)
                pend_think = []
            continue
        role = 'assistant' if kind in ('S', 'AN') else 'user'
        if pend_sys:
            msgs.append(dict(role='system', parts=[('l', t) for t in pend_sys], think=[]))
            pend_sys = []
            cur = None
        if cur is None or cur['role'] != role:
            cur = dict(role=role, parts=[], think=[])
            msgs.append(cur)
        if kind in ('UN', 'AN'):
            if cur['parts'] and cur['parts'][-1][0] == 'n':
                cur['parts'][-1][1].append(txt)
            else:
                cur['parts'].append(('n', [txt]))
        else:
            cur['parts'].append(('l', txt))
        if role == 'assistant' and pend_think:
            cur['think'] += pend_think
            pend_think = []
    trailing_think = pend_think

    def speech(m):
        return [v for k, v in m['parts'] if k == 'l']

    def trim_end():
        nonlocal trailing_think
        # the last message must be an assistant turn that contains speech; a final narration-only turn of hers
        # is dropped (its thoughts move on with the trailing thoughts)
        while msgs and (msgs[-1]['role'] != 'assistant' or not speech(msgs[-1])):
            m = msgs.pop()
            if m['role'] == 'assistant':
                trailing_think = m['think'] + trailing_think
    trim_end()
    if d['id'] in CFG.TRIM_FILLER_END:
        while msgs and msgs[-1]['role'] == 'assistant' and all(target_trivial(t) for t in speech(msgs[-1])):
            trailing_think = msgs[-1]['think'] + trailing_think
            msgs.pop()
            trim_end()
    if not msgs:
        return None
    n_spoken = sum(len(speech(m)) for m in msgs if m['role'] == 'assistant')
    # a single target line without any other character's speech is no exchange (narration doesn't count)
    has_user = any(m['role'] == 'user' and speech(m) for m in msgs)
    if n_spoken == 1 and not has_user and not a.get('keep_single'):
        return None
    synthetic = False
    opening_system = None
    if msgs[0]['role'] != 'user':
        while msgs and msgs[0]['role'] == 'system':
            opening_narr += [v for _, v in msgs.pop(0)['parts']]
        if msgs[0]['role'] == 'assistant':
            op = a.get('opening')
            if not op:
                raise SystemExit('missing opening for %s' % d['id'])
            if pov:
                # the target is alone and the protagonist is not there: the stage direction is omniscient (system)
                opening_system = op
            else:
                msgs.insert(0, dict(role='user', parts=[('l', op)], think=[]))
                synthetic = True
    # ---- header
    present = list(a.get('present') or [s for s in d['scene_speakers'] if s not in UNKNOWN_PLATES])
    if not a.get('present_fixed'):
        for m in member_anns:
            for p in m.get('present', []):
                if p not in present:
                    present.append(p)
    if TARGET in present:
        present.remove(TARGET)
    present = [TARGET] + present
    narrator = a.get('narrator') or PROTAGONIST
    scene_lines = ['―― 場面 ――']
    rc = CFG.route_context(d, a)
    if rc:
        scene_lines.append('ルート：%s' % rc)
    scene_lines += ['場所：%s' % (loc or '不明'),
                    '登場人物：%s' % '、'.join(present),
                    '地の文：%s' % narration_desc(pov, narrator),
                    '状況：%s' % a['situation'].strip(),
                    'これまでの経緯：%s' % a['recap'].strip()]
    if opening_narr:
        scene_lines.append('直前の地の文：\n' + '\n'.join(opening_narr))
    out = [dict(role='system', content=CFG.PERSONA.strip() + '\n\n' + '\n'.join(scene_lines))]
    if opening_system:
        out.append(dict(role='system', content=opening_system))
    n_think = 0
    counts = collections.Counter()
    for m in msgs:
        if m['role'] == 'assistant' and m['think']:
            n_think += 1
            counts['think_lines'] += len(m['think'])
        for k, v in m['parts']:
            if k == 'n':
                counts['%s_narration' % m['role']] += len(v)
            elif m['role'] == 'system':
                counts['system_lines'] += 1
        if m['role'] == 'assistant' and not speech(m) and m['parts']:
            counts['narration_only_assistant_turns'] += 1
        out.append(dict(role=m['role'], content=render(m)))
    out = fold_system_into_user(out)
    last_line = max(d['lines'])
    if d['id'] in CFG.TRIM_FILLER_END:
        kept = speech(msgs[-1])[-1]
        last_line = next(L for L in reversed(d['lines']) if its[L]['kind'] == 'S' and its[L]['text'] == kept)
    meta = dict(source_file=stem, scene=d['scene'], route=d['route'],
                line_range=[min(d['lines']), last_line])
    if d.get('merged_from'):
        meta['merged_from'] = d['merged_from']
    if d['branch']:
        meta['branch'] = d['branch']
    if d['cond']:
        meta['condition'] = d['cond']
        for key, note in CFG.COND_NOTES.items():
            if key in d['cond']:
                meta['condition_note'] = note
    if pov:
        meta['pov'] = TARGET
    if synthetic:
        meta['synthetic_opening'] = True
    meta['think_turns'] = n_think
    return out, meta, trailing_think


def cmd_build():
    plan = build_plan()
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(os.path.join(WORK_DIR, 'plan.json'), 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=0)
    ann = load_annotations()
    missing = [d['id'] for d in plan if d['id'] not in ann]
    if missing:
        raise SystemExit('missing annotations: %s' % missing)
    out = []
    skipped = []
    carry = []
    carry_from = None
    for d in plan:
        if d['id'] in getattr(CFG, 'SKIP_DOCS', ()):
            skipped.append(d['id'])
            continue
        a = ann[d['id']]
        members = [ann[m] for m in d.get('merged_from', [])[1:] if m in ann]
        pov = any(s in CFG.POV_SCENES for s in d['scenes'])
        r = assemble(d, a, members, carry if pov else [])
        if r is None:
            skipped.append(d['id'])
            continue
        msgs, meta, trailing = r
        if pov:
            carry = trailing
            carry_from = d['id']
        elif carry:
            # the POV run is over: its last thoughts must be quoted in the recap of the document named in the config
            target = CFG.POV_TRAILING_TO_RECAP.get(carry_from)
            if target != d['id'] or not all(t in a['recap'] for t in carry):
                raise SystemExit('trailing thoughts of %s were not carried into a following document: %s' % (carry_from, carry))
            carry = []
        out.append(dict(id=d['id'], meta=meta, messages=msgs))
    if carry:
        raise SystemExit('trailing thoughts of %s were dropped: %s' % (carry_from, carry))
    with open(os.path.join(OUT_DIR, OUTPUT), 'w', encoding='utf-8', newline='\n') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('wrote %d documents, skipped %d: %s' % (len(out), len(skipped), ' '.join(skipped)))
    cmd_validate()


def cmd_append(source_stem):
    """Append new documents from one source file without rewriting any existing JSONL record.

    The original file is retained as an exact byte prefix of the result. Only planned ids absent
    from that file are assembled; previously omitted documents from other sources are irrelevant.
    """
    path = os.path.join(OUT_DIR, OUTPUT)
    with open(path, 'rb') as f:
        original = f.read()
    existing = [json.loads(l) for l in original.decode('utf-8').splitlines() if l.strip()]
    existing_ids = {r['id'] for r in existing}
    if len(existing_ids) != len(existing):
        raise SystemExit('existing dataset has duplicate ids; refusing to append')

    plan = build_plan()
    selected = [d for d in plan if d['source_file'] == source_stem]
    if not selected:
        raise SystemExit('no planned documents for source %s' % source_stem)
    ann = load_annotations()
    skip_docs = getattr(CFG, 'SKIP_DOCS', set())
    missing = [d['id'] for d in selected
               if d['id'] not in existing_ids and d['id'] not in skip_docs and d['id'] not in ann]
    if missing:
        raise SystemExit('missing annotations: %s' % missing)

    additions = []
    skipped = []
    carry = []
    carry_from = None
    for d in selected:
        if d['id'] in existing_ids or d['id'] in skip_docs:
            continue
        a = ann[d['id']]
        members = [ann[m] for m in d.get('merged_from', [])[1:] if m in ann]
        pov = any(s in CFG.POV_SCENES for s in d['scenes'])
        r = assemble(d, a, members, carry if pov else [])
        if r is None:
            skipped.append(d['id'])
            continue
        msgs, meta, trailing = r
        if pov:
            carry = trailing
            carry_from = d['id']
        elif carry:
            raise SystemExit('trailing thoughts of %s cross the append source boundary' % carry_from)
        additions.append(dict(id=d['id'], meta=meta, messages=msgs))
    if carry:
        raise SystemExit('trailing thoughts of %s were dropped' % carry_from)
    if not additions:
        raise SystemExit('no new documents to append for %s' % source_stem)
    new_ids = {r['id'] for r in additions}
    if len(new_ids) != len(additions) or new_ids & existing_ids:
        raise SystemExit('new document ids are not unique')

    os.makedirs(WORK_DIR, exist_ok=True)
    backup = os.path.join(WORK_DIR, '%s.before_%s.jsonl' % (os.path.splitext(OUTPUT)[0], source_stem))
    with open(backup, 'wb') as f:
        f.write(original)
    separator = b'' if not original or original.endswith(b'\n') else b'\n'
    encoded = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in additions).encode('utf-8')
    try:
        with open(path, 'wb') as f:
            f.write(original + separator + encoded)
        print('appended %d new documents from %s, skipped %d: %s' % (
            len(additions), source_stem, len(skipped), ' '.join(skipped)))
        cmd_validate()
    except BaseException:
        with open(path, 'wb') as f:
            f.write(original)
        raise


FILLER_RE = re.compile(r'^[「『（]?[\s…‥・。、―ー－～〜！？!?\.っッうんえあ？]*[」』）]?$')


NARR_RE = re.compile(r'(<narration>.*?</narration>)', re.S)


def split_parts(c):
    """Split a user/assistant body into [('n', [lines]) | ('l', line)] and a list of tag problems."""
    probs = []
    if c.count('<narration>') != c.count('</narration>'):
        probs.append('unbalanced <narration> tags')
    parts = []
    for chunk in NARR_RE.split(c):
        if chunk.startswith('<narration>') and chunk.endswith('</narration>'):
            inner = chunk[len('<narration>'):-len('</narration>')]
            if '<narration>' in inner or '</narration>' in inner:
                probs.append('nested <narration>')
            lines = inner.split('\n')
            if not inner.strip() or any(not l.strip() for l in lines):
                probs.append('empty narration line or block')
            parts.append(('n', lines))
        else:
            if '<narration>' in chunk or '</narration>' in chunk:
                probs.append('stray <narration> tag')
            for l in chunk.split('\n'):
                if l.strip():
                    parts.append(('l', l))
    return parts, probs


def cmd_validate():
    path = os.path.join(OUT_DIR, OUTPUT)
    target_lines = collections.defaultdict(set)
    narr_lines = collections.defaultdict(set)
    errors = []
    warnings = []
    recs = []
    with open(path, encoding='utf-8') as f:
        for n, l in enumerate(f, 1):
            try:
                recs.append(json.loads(l))
            except Exception as e:
                errors.append('line %d: bad json %s' % (n, e))
    ids = collections.Counter(r['id'] for r in recs)
    for k, v in ids.items():
        if v > 1:
            errors.append('%s: duplicate id' % k)
    ann = load_annotations()
    roles_all = load_roles()
    # what may appear in her <narration>: the first-person form of each labelled line (or its logged rewrite),
    # and, in the POV epilogue, her own_action lines verbatim
    allowed_an = collections.defaultdict(set)
    notes = set()      # hand-written system texts
    for k, v in ann.items():
        notes |= {g['text'] for g in v.get('as_note', [])}
        notes |= set(v.get('insert_after', {}).values()) | set(v.get('insert_before', {}).values())
        notes |= {g['text'] for g in v.get('condense_ranges', [])}
        if v.get('opening'):
            notes.add(v['opening'])
    persona = CFG.PERSONA.strip()
    stat = collections.Counter()
    for r in recs:
        stem = r['meta']['source_file']
        its = file_items(stem)
        if stem not in target_lines:
            for it in its.values():
                if it['kind'] == 'S':
                    target_lines[stem].add(it['text'])
                elif it['kind'] == 'N':
                    narr_lines[stem].add(it['text'])
            ro = roles_all.get(stem)
            if ro:
                for L in ro['target']:
                    allowed_an[stem].add(target_narration(its[L]['text'], stem, L))
        pov = bool(r['meta'].get('pov'))
        if pov:
            a = ann[r['id']]
            own = set(a.get('own_action', []))
            for mid in r['meta'].get('merged_from', [])[1:]:
                own |= set(ann.get(mid, {}).get('own_action', []))
            pov_own = {its[L]['text'] for L in own}
        m = r['messages']
        roles = [x['role'] for x in m]
        if roles[0] != 'system':
            errors.append('%s: first message not system' % r['id'])
        if 'system' in roles[1:]:
            errors.append('%s: system message after the opening setup' % r['id'])
        conv = roles[1:]
        if not conv or conv[0] != 'user':
            errors.append('%s: first non-system message is not user' % r['id'])
        if roles[-1] != 'assistant':
            errors.append('%s: does not end with assistant' % r['id'])
        for a_, b_ in zip(conv, conv[1:]):
            pass
        for a_, b_ in zip(roles, roles[1:]):
            if a_ == b_ and a_ in ('user', 'assistant'):
                errors.append('%s: %s directly followed by %s' % (r['id'], a_, b_))
                break
        if not m[0]['content'].startswith(persona):
            errors.append('%s: first system message does not start with the persona card' % r['id'])
        for i, x in enumerate(m):
            if x['role'] not in ('system', 'user', 'assistant'):
                errors.append('%s: bad role %s' % (r['id'], x['role']))
            c = x['content']
            if i == 0:
                c = c[len(persona):]
            if not c.strip():
                errors.append('%s: empty %s message' % (r['id'], x['role']))
            if '【' in c or '】' in c:
                errors.append('%s: plate bracket leaked' % r['id'])
            if any(ln.startswith('#') or ln.startswith('===') for ln in c.split('\n')):
                errors.append('%s: comment line leaked' % r['id'])
            if re.search(r'[0-9]', c):
                errors.append('%s: ASCII digit (line-number or meta leak) in %s: %s' % (
                    r['id'], x['role'], re.search(r'.{0,15}[0-9].{0,15}', c).group(0)))
            if re.search(r'script|行目|line ', c, re.I):
                errors.append('%s: meta note leaked in %s' % (r['id'], x['role']))
            if x['role'] != 'assistant' and ('<think>' in c or '</think>' in c):
                errors.append('%s: think tag outside assistant' % r['id'])
            if x['role'] == 'system':
                continue
            body = c
            if x['role'] == 'assistant' and body.startswith('<think>\n'):
                if body.endswith('\n</think>') and body.count('</think>') == 1:
                    th, body = body[len('<think>\n'):-len('\n</think>')], ''
                    stat['think_only_turns'] += 1
                elif '\n</think>\n\n' in body:
                    th, body = body[len('<think>\n'):].split('\n</think>\n\n', 1)
                else:
                    errors.append('%s: malformed think block' % r['id'])
                    continue
                for t in th.split('\n'):
                    stat['think_lines'] += 1
                    if t in narr_lines[stem]:
                        continue
                    if pov and is_bracketed_thought(t) and t in target_lines[stem]:
                        stat['think_bracket_lines'] += 1   # her plated （…） private thought, verbatim
                        continue
                    errors.append('%s: think line is neither narration nor a %s （） line: %s' % (r['id'], TARGET, t[:40]))
            if '<think>' in body or '</think>' in body:
                errors.append('%s: stray think tag' % r['id'])
            parts, probs = split_parts(body)
            for p in probs:
                errors.append('%s: %s in %s' % (r['id'], p, x['role']))
            speech = [v for k, v in parts if k == 'l']
            narr = [l for k, v in parts if k == 'n' for l in v]
            if x['role'] == 'user':
                stat['user_narration_lines'] += len(narr)
                for ln in narr:
                    if ln not in narr_lines[stem] and ln not in notes and not re.match(r'^（場所：[^）]+）$', ln):
                        errors.append('%s: user narration is neither script narration nor a known folded system note: %s' % (r['id'], ln[:40]))
                for ln in speech:
                    if not (re.match(r'^[^「『：]{1,30}：[「『（]', ln) or (r['meta'].get('synthetic_opening') and ln in notes)):
                        errors.append('%s: user line is neither speech nor the synthetic opening: %s' % (r['id'], ln[:40]))
            else:
                stat['assistant_narration_lines'] += len(narr)
                stat['target_speech_lines'] += len(speech)
                if not speech and narr:
                    stat['narration_only_assistant_turns'] += 1
                for ln in narr:
                    if pov:
                        stat['pov_own_action_lines'] += 1
                        if ln not in pov_own:
                            errors.append('%s: POV narration is not an own_action line: %s' % (r['id'], ln[:40]))
                    else:
                        if ln not in allowed_an[stem]:
                            errors.append('%s: her narration is not a labelled line (after the pronoun map / rewrite): %s' % (r['id'], ln[:40]))
                        if re.search('%s|%s' % (NARRATOR_PRONOUNS, re.escape(TARGET)), ln):
                            errors.append('%s: third person or narrator pronoun left in her narration: %s' % (r['id'], ln[:40]))
                        if ln not in narr_lines[stem]:
                            stat['rewritten_narration_lines'] += 1
                for ln in speech:
                    if ln not in target_lines[stem]:
                        errors.append('%s: assistant line not a %s line: %s' % (r['id'], TARGET, ln[:40]))
        last_parts, _ = split_parts(m[-1]['content'].split('</think>')[-1].strip())
        last = [v for k, v in last_parts if k == 'l']
        if not last:
            errors.append('%s: last assistant turn has no speech' % r['id'])
        elif r['id'] in getattr(CFG, 'FILLER_END_OK', ()):
            pass
        elif all(target_trivial(ln) for ln in last):
            errors.append('%s: ends on a filler %s turn: %s' % (r['id'], TARGET, ' / '.join(last)[:40]))
        elif target_trivial(last[-1]):
            warnings.append('%s: last line is filler (%s) after a meaningful line' % (r['id'], last[-1]))
    # stats
    def L(r):
        return sum(len(x['content']) for x in r['messages'])
    lens = sorted(L(r) for r in recs)
    turns = collections.Counter()
    for r in recs:
        for x in r['messages']:
            turns[x['role']] += 1
    a_per = sorted(sum(1 for x in r['messages'] if x['role'] == 'assistant') for r in recs)
    q = lambda arr, p: arr[min(len(arr) - 1, int(p * len(arr)))]
    hist = collections.Counter((x // 1000) * 1000 for x in lens)
    token_counts = [(r, QV.conservative_token_count(r['messages'], TOKENIZER_DIR)) for r in recs]
    token_lens = sorted(v for _, v in token_counts)
    stats = collections.OrderedDict(
        documents=len(recs), messages=dict(turns),
        assistant_turns_per_doc=dict(min=a_per[0], median=q(a_per, .5), mean=round(sum(a_per) / len(a_per), 2), max=a_per[-1]),
        narration=dict(user=stat['user_narration_lines'], assistant=stat['assistant_narration_lines'],
                       assistant_rewritten=stat['rewritten_narration_lines'], system=stat['system_narration_lines'],
                       think=stat['think_lines'], pov_own_action=stat['pov_own_action_lines']),
        narration_only_assistant_turns=stat['narration_only_assistant_turns'],
        think_only_turns=stat['think_only_turns'],
        think_bracket_lines=stat['think_bracket_lines'],
        think_documents=sum(1 for r in recs if r['meta'].get('think_turns')),
        think_turns=sum(r['meta'].get('think_turns', 0) for r in recs),
        synthetic_openings=sum(1 for r in recs if r['meta'].get('synthetic_opening')),
        system_openings=sum(1 for r in recs if r['meta'].get('opening_system')),
        merged_documents=sum(1 for r in recs if r['meta'].get('merged_from')),
        by_route=dict(collections.Counter(r['meta']['route'] for r in recs)),
        chars=dict(min=lens[0], p10=q(lens, .1), p25=q(lens, .25), median=q(lens, .5), p75=q(lens, .75), p90=q(lens, .9), max=lens[-1]),
        chars_histogram={'%d-%d' % (k, k + 999): hist[k] for k in sorted(hist)},
        tokenizer=QV.CHECKPOINT,
        token_limit=TOKEN_LIMIT,
        qwen35_tokens=dict(min=token_lens[0], median=q(token_lens, .5), p95=q(token_lens, .95), max=token_lens[-1]),
        over_token_limit=sum(1 for x in token_lens if x > TOKEN_LIMIT),
        assistant_chars=sum(len(x['content']) for r in recs for x in r['messages'] if x['role'] == 'assistant'),
        target_lines_used=stat['target_speech_lines'],
    )
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    with open(os.path.join(WORK_DIR, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    for r, tokens in token_counts:
        if tokens > TOKEN_LIMIT:
            errors.append('%s: %d Qwen3.5 tokens exceeds the %d-token limit' % (r['id'], tokens, TOKEN_LIMIT))
    for w in warnings:
        print('warning:', w)
    if errors:
        print('ERRORS (%d):' % len(errors))
        for e in errors[:80]:
            print(' ', e)
        sys.exit(1)
    print('validation OK')


# ---------------------------------------------------------------- main
def cmd_plan():
    plan = build_plan()
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(os.path.join(WORK_DIR, 'plan.json'), 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=0)
    print('docs', len(plan), 'scenes', len({d['scene'] for d in plan}))
    write_worksheets(plan)
    return plan


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    a = sys.argv[1:]
    opt = {'--config': None, '--project': None, '--tokenizer': None}
    for k in list(opt):
        if k in a:
            i = a.index(k)
            opt[k] = a[i + 1]
            del a[i:i + 2]
    if not opt['--config'] or not opt['--project']:
        sys.exit(__doc__)
    setup(opt['--config'], opt['--project'], opt['--tokenizer'])
    cmd = a[0] if a else 'build'
    if cmd == 'plan':
        cmd_plan()
    elif cmd == 'build':
        cmd_build()
    elif cmd == 'append':
        if len(a) != 2:
            sys.exit('usage: character_dataset.py --config CONFIG --project DIR append SOURCE_STEM')
        cmd_append(a[1])
    elif cmd == 'validate':
        cmd_validate()
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main()
