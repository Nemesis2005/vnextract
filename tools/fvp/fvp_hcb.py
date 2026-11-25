#!/usr/bin/env python3
"""
Scenario extractor for FAVORITE View Point (FVP) .hcb bytecode.  Standard library only.

Reads the .hcb file (read-only) and writes human-readable script files (UTF-8), a script.jsonl
companion and stats.txt, in the normalized script format (repository README).

Usage:  python -I fvp_hcb.py <profile.json> <path/to/Game.hcb> <output_dir>

The opcode table, the HCB header layout, the MiniVM and the output format are engine-generic.
Everything game-specific (function addresses, global numbers, the chapter table) comes from the
profile (repository README, "FVP profile"); rediscover each value for a new title.

  * HCB layout: u32 offset of the system header at file offset 0; bytecode from 4 up to that
    offset; header = u32 entry, u16 nv-globals, u16 v-globals, u16 game mode, u8+title,
    u16 syscall count, then (u8 argc, u8 len, name) per syscall.
  * functions.text     : text(text, size, speed, bracket_mode); bracket_mode nil/0 -> the engine
                         wraps speaker lines in 「」, 1 -> no brackets.
  * functions.wait     : page/click wait (speaker and name plate are cleared afterwards)
  * functions.name     : name-plate function name(plate, internal_name).  Speaker functions are
                         found automatically: they set globals.speaker and call it; arg0 = voice id.
  * choice_add/show    : add an option / show the menu; the result is in globals.choice.
  * bg_label           : background helpers start with  push_str <location>; call bg_label
  * bgm_play/bgm_stop  : BGM helpers call bgm_play(track, ...); bgm_stop stops it.
  * screen/screen_flag : screen-clear helpers set globals.screen_flag = 1 and call screen(..., kind)
"""
import os
import re
import struct
import sys
import collections

# ----------------------------------------------------------------------------- HCB parsing
OPS = {
    0x00: ('nop', ''), 0x01: ('init_stack', 'BB'), 0x02: ('call', 'I'),
    0x03: ('syscall', 'H'), 0x04: ('ret', ''), 0x05: ('retv', ''),
    0x06: ('jmp', 'I'), 0x07: ('jz', 'I'), 0x08: ('push_nil', ''),
    0x09: ('push_true', ''), 0x0A: ('push_i32', 'i'), 0x0B: ('push_i16', 'h'),
    0x0C: ('push_i8', 'b'), 0x0D: ('push_f32', 'f'), 0x0E: ('push_str', 'S'),
    0x0F: ('push_global', 'H'), 0x10: ('push_stack', 'b'),
    0x11: ('push_gtable', 'H'), 0x12: ('push_ltable', 'b'),
    0x13: ('push_top', ''), 0x14: ('push_ret', ''),
    0x15: ('pop_global', 'H'), 0x16: ('pop_stack', 'b'),
    0x17: ('pop_gtable', 'H'), 0x18: ('pop_ltable', 'b'),
    0x19: ('neg', ''), 0x1A: ('add', ''), 0x1B: ('sub', ''), 0x1C: ('mul', ''),
    0x1D: ('div', ''), 0x1E: ('mod', ''), 0x1F: ('bittest', ''),
    0x20: ('and', ''), 0x21: ('or', ''), 0x22: ('sete', ''), 0x23: ('setne', ''),
    0x24: ('setg', ''), 0x25: ('setle', ''), 0x26: ('setl', ''), 0x27: ('setge', ''),
}
SIZES = {'B': 1, 'b': 1, 'H': 2, 'h': 2, 'I': 4, 'i': 4, 'f': 4}


class HCB:
    def __init__(self, path, enc='cp932'):
        with open(path, 'rb') as f:
            self.data = d = f.read()
        self.enc = enc
        self.hdr_off = struct.unpack_from('<I', d, 0)[0]
        p = self.hdr_off
        self.entry, self.nv_globals, self.v_globals, self.mode = struct.unpack_from('<IHHH', d, p)
        p += 10
        tlen = d[p]; p += 1
        self.title = d[p:p + tlen].rstrip(b'\0').decode(enc); p += tlen
        n = struct.unpack_from('<H', d, p)[0]; p += 2
        self.syscalls = []
        for _ in range(n):
            argc, nl = d[p], d[p + 1]; p += 2
            self.syscalls.append((d[p:p + nl].rstrip(b'\0').decode('ascii'), argc))
            p += nl
        self.ins = list(self._disasm())
        self.idx = {a: i for i, (a, _, _) in enumerate(self.ins)}
        # functions = init_stack boundaries
        self.funcs = {}
        order = [a for a, n, _ in self.ins if n == 'init_stack']
        for k, a in enumerate(order):
            end = order[k + 1] if k + 1 < len(order) else self.hdr_off
            nargs, nlocals = self.ins[self.idx[a]][2]
            self.funcs[a] = (end, nargs, nlocals)

    def _disasm(self):
        d, p, end = self.data, 4, self.hdr_off
        while p < end:
            addr = p
            op = d[p]; p += 1
            name, fmt = OPS[op]  # KeyError => unknown opcode, format assumption wrong
            args = []
            for c in fmt:
                if c == 'S':
                    ln = d[p]; p += 1
                    args.append(d[p:p + ln].rstrip(b'\0').decode(self.enc, errors='replace'))
                    p += ln
                else:
                    args.append(struct.unpack_from('<' + c, d, p)[0]); p += SIZES[c]
            yield addr, name, args

    def body(self, faddr):
        end = self.funcs[faddr][0]
        i = self.idx[faddr]
        while i < len(self.ins) and self.ins[i][0] < end:
            yield i, self.ins[i]
            i += 1

    def callee_argc(self, target):
        f = self.funcs.get(target)
        return f[1] if f else 0


# ----------------------------------------------------------------------------- game-specific constants
# Function addresses, globals and the chapter table differ per title. They are loaded from a JSON
# profile JSON by use_profile(), which sets the module globals below.
F_TEXT = F_WAIT = F_NAME = F_CHOICE_ADD = F_CHOICE_SHOW = None
F_BG_LABEL = F_BGM_PLAY = F_BGM_STOP = F_SCREEN = None
G_SPEAKER = G_CHOICE = G_SCREEN_FLAG = None
CHAPTERS = []      # (function address, output file stem, title), in story order
PROFILE = {}


def _addr(v):
    return int(v, 0) if isinstance(v, str) else int(v)


def use_profile(path):
    """Load a per-game FVP profile and set the module constants from it.  Returns the profile dict."""
    global F_TEXT, F_WAIT, F_NAME, F_CHOICE_ADD, F_CHOICE_SHOW, F_BG_LABEL, F_BGM_PLAY, F_BGM_STOP
    global F_SCREEN, G_SPEAKER, G_CHOICE, G_SCREEN_FLAG, CHAPTERS, PROFILE
    import json
    with open(path, encoding='utf-8') as f:
        prof = json.load(f)
    fn, gl = prof['functions'], prof['globals']
    F_TEXT, F_WAIT, F_NAME = _addr(fn['text']), _addr(fn['wait']), _addr(fn['name'])
    F_CHOICE_ADD, F_CHOICE_SHOW = _addr(fn['choice_add']), _addr(fn['choice_show'])
    F_BG_LABEL, F_BGM_PLAY, F_BGM_STOP = _addr(fn['bg_label']), _addr(fn['bgm_play']), _addr(fn['bgm_stop'])
    F_SCREEN = _addr(fn['screen'])
    G_SPEAKER, G_CHOICE, G_SCREEN_FLAG = int(gl['speaker']), int(gl['choice']), int(gl['screen_flag'])
    CHAPTERS = [(_addr(a), stem, title) for a, stem, title in prof['chapters']]
    PROFILE = prof
    return prof


class Unknown:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


def fmt_val(v):
    if isinstance(v, Unknown):
        return v.s
    if v is None:
        return 'nil'
    if v is True:
        return 'true'
    if isinstance(v, str):
        return '"%s"' % v
    return str(v)


def strip_name(s):
    """Name plates are centred with full-width spaces; '~' separates stacked names."""
    if s is None:
        return None
    parts = [p.strip('　 ') for p in s.split('~')]
    return '／'.join(p for p in parts if p)


RUBY = re.compile(r'\[([^\[\]|]*)\|([^\[\]|]*)\]')


def clean_text(s):
    # [reading|base] ruby -> base text  (also covers emphasis dots [・|字])
    return RUBY.sub(lambda m: m.group(2), s)


# ----------------------------------------------------------------------------- mini VM (speaker functions)
class MiniVM:
    """Executes a speaker function with concrete args until it calls the name-plate function."""

    def __init__(self, h):
        self.h = h

    def run(self, faddr, args):
        h = self.h
        end, nargs, nlocals = h.funcs[faddr]
        args = list(args)[:nargs] + [None] * (nargs - len(args))
        frame = {}
        stack = []
        g = collections.defaultdict(lambda: None)
        i = h.idx[faddr] + 1
        for _ in range(5000):
            a, n, ar = h.ins[i]
            i += 1
            if n.startswith('push_') and n not in ('push_global', 'push_stack', 'push_ret', 'push_gtable',
                                                     'push_ltable', 'push_top', 'push_nil', 'push_true'):
                stack.append(ar[0])
            elif n == 'push_nil' or n == 'push_ret':
                stack.append(None)
            elif n == 'push_true':
                stack.append(True)
            elif n == 'push_global':
                stack.append(g[ar[0]])
            elif n == 'push_stack':
                k = ar[0]
                stack.append(args[nargs + 1 + k] if k < 0 else frame.get(k))
            elif n == 'pop_global':
                g[ar[0]] = stack.pop()
            elif n == 'pop_stack':
                k = ar[0]
                v = stack.pop()
                if k < 0:
                    args[nargs + 1 + k] = v
                else:
                    frame[k] = v
            elif n in ('push_gtable', 'push_ltable'):
                stack.pop(); stack.append(None)
            elif n in ('pop_gtable', 'pop_ltable'):
                stack.pop(); stack.pop()
            elif n == 'push_top':
                stack.append(stack[-1])
            elif n == 'neg':
                v = stack.pop(); stack.append(-v if isinstance(v, (int, float)) and v is not True else v)
            elif n in ('add', 'sub', 'mul', 'div', 'mod', 'bittest', 'and', 'or',
                       'sete', 'setne', 'setg', 'setle', 'setl', 'setge'):
                b = stack.pop(); x = stack.pop()
                stack.append(binop(n, x, b))
            elif n == 'jz':
                v = stack.pop()
                if not truthy(v):
                    i = h.idx[ar[0]]
            elif n == 'jmp':
                i = h.idx[ar[0]]
            elif n == 'call':
                c = h.callee_argc(ar[0])
                cargs = stack[len(stack) - c:] if c else []
                del stack[len(stack) - c:]
                if ar[0] == F_NAME:
                    return cargs
            elif n == 'syscall':
                c = h.syscalls[ar[0]][1]
                del stack[len(stack) - c:]
            elif n in ('ret', 'retv'):
                return None
        return None


def truthy(v):
    return not (v is None or v is False or v == 0)


def binop(n, x, b):
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    if n == 'sete':
        return (type(x) == type(b) and x == b) or (x is None and b is None)
    if n == 'setne':
        return not ((type(x) == type(b) and x == b) or (x is None and b is None))
    if n == 'and':
        return truthy(x) and truthy(b)
    if n == 'or':
        return truthy(x) or truthy(b)
    if not (num(x) and num(b)):
        if n == 'add' and isinstance(x, str) and isinstance(b, str):
            return x + b
        return None
    # comparison direction verified in the bytecode (e.g. "x 8 setle  x 64 setge and" = 8<=x<=64)
    if n == 'setle':
        return x >= b
    if n == 'setge':
        return x <= b
    if n == 'setl':
        return x > b
    if n == 'setg':
        return x < b
    if n == 'add':
        return x + b
    if n == 'sub':
        return x - b
    if n == 'mul':
        return x * b
    if n == 'div':
        return int(x / b) if b else None
    if n == 'mod':
        return x % b if b else None
    if n == 'bittest':
        return bool((int(x) >> int(b)) & 1)
    return None


OPSYM = {'sete': '==', 'setne': '!=', 'setle': '>=', 'setge': '<=', 'setl': '>', 'setg': '<',
         'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'mod': '%', 'and': '&&', 'or': '||',
         'bittest': 'bit'}


# ----------------------------------------------------------------------------- analysis of helper functions
def classify_functions(h):
    speakers = {}   # faddr -> speaker id
    bg = {}         # faddr -> location label
    bgm = {}        # faddr -> track no.
    for fa in h.funcs:
        body = list(h.body(fa))
        if len(body) < 3:
            continue
        a1, n1, ar1 = body[1][1]
        a2, n2, ar2 = body[2][1]
        if n1 == 'push_i8' and n2 == 'pop_global' and ar2[0] == G_SPEAKER and ar1[0] != 0:
            if any(x[1][1] == 'call' and x[1][2][0] == F_NAME for x in body):
                speakers[fa] = ar1[0]
        if n1 == 'push_str' and n2 == 'call' and ar2[0] == F_BG_LABEL:
            bg[fa] = ar1[0]
        if fa != F_BGM_PLAY and len(body) < 20:
            for k, (_, (a, n, ar)) in enumerate(body):
                if n == 'call' and ar[0] == F_BGM_PLAY and k >= 5:
                    t = body[k - 5][1]
                    if t[1] in ('push_i8', 'push_i16', 'push_i32'):
                        bgm[fa] = t[2][0]
    return speakers, bg, bgm


# ----------------------------------------------------------------------------- scenario sweep
class Extractor:
    def __init__(self, h):
        self.h = h
        self.vm = MiniVM(h)
        self.speakers, self.bg, self.bgm = classify_functions(h)
        self.name_cache = {}
        self.canon = {}  # speaker function -> canonical name (variant nil)
        for fa in self.speakers:
            r = self.vm.run(fa, [])
            self.canon[fa] = strip_name(r[0]) if r else '?'
        self.chapter_of = {fa: (fn, title) for fa, fn, title in CHAPTERS}
        self.screens = find_screens(h)
        self.on_call = None   # optional hook(target, args, out) for calls the extractor does not know

    def speaker_name(self, fa, args):
        key = (fa, tuple(fmt_val(a) for a in args))
        if key not in self.name_cache:
            r = self.vm.run(fa, [None if isinstance(a, Unknown) else a for a in args])
            if r:
                disp = strip_name(r[0])
                alt = strip_name(r[1]) if len(r) > 1 and r[1] is not None else disp
            else:
                disp = alt = self.canon[fa]
            self.name_cache[key] = (disp, alt)
        return self.name_cache[key]

    def has_text(self, i0, i1):
        for k in range(i0, i1):
            a, n, ar = self.h.ins[k]
            if n == 'call' and ar[0] == F_TEXT:
                return True
        return False

    def extract(self, fa):
        h = self.h
        out = []         # list of (kind, ...) records
        stack = []
        speaker = None   # (disp, canonical, alt, voice)
        options = []     # pending choice options
        slot_opt = {}    # local slot -> option label (choice dispatch via locals)
        branch_label = {}  # address -> label to print when reached
        pending_end = collections.defaultdict(list)  # address -> list of markers
        end_addr = h.funcs[fa][0]
        body = list(h.body(fa))
        scene_no = 0
        lines_since_scene = 0

        def pop(c=1):
            r = stack[len(stack) - c:] if c else []
            del stack[len(stack) - c:]
            return [None] * (c - len(r)) + r

        def new_scene(reason):
            nonlocal scene_no, lines_since_scene
            scene_no += 1
            lines_since_scene = 0
            out.append(('scene', scene_no, reason))

        new_scene('start')
        for bi, (i, (a, n, ar)) in enumerate(body):
            for m in pending_end.pop(a, []):
                out.append(('comment', m))
            if a in branch_label:
                out.append(('comment', 'branch: %s' % branch_label[a]))
            if n == 'init_stack':
                continue
            if n in ('push_i8', 'push_i16', 'push_i32', 'push_f32', 'push_str'):
                stack.append(ar[0])
            elif n == 'push_nil':
                stack.append(None)
            elif n == 'push_true':
                stack.append(True)
            elif n == 'push_ret':
                stack.append(Unknown('ret'))
            elif n == 'push_global':
                stack.append(Unknown('g%d' % ar[0]))
            elif n == 'push_stack':
                stack.append(Unknown('local%d' % ar[0] if ar[0] >= 0 else 'arg%d' % ar[0]))
            elif n in ('push_gtable', 'push_ltable'):
                k = pop()[0]
                stack.append(Unknown('%s%d[%s]' % ('g' if n == 'push_gtable' else 'local', ar[0], fmt_val(k))))
            elif n == 'push_top':
                stack.append(stack[-1] if stack else Unknown('?'))
            elif n in ('pop_global', 'pop_stack'):
                pop()
            elif n in ('pop_gtable', 'pop_ltable'):
                pop(2)
            elif n == 'neg':
                v = pop()[0]
                stack.append(-v if isinstance(v, (int, float)) and not isinstance(v, bool) else Unknown('-%s' % fmt_val(v)))
            elif n in OPSYM:
                x, b = pop(2)
                if isinstance(x, Unknown) or isinstance(b, Unknown):
                    stack.append(Unknown('(%s %s %s)' % (fmt_val(x), OPSYM[n], fmt_val(b))))
                else:
                    stack.append(binop(n, x, b))
            elif n == 'jz':
                cond = pop()[0]
                tgt = ar[0]
                if tgt > a and tgt <= end_addr:
                    ti = h.idx[tgt]
                    if self.has_text(i, ti):
                        out.append(('comment', 'if %s' % fmt_val(cond)))
                        # else-part: forward jmp right before the target
                        pa, pn, par = h.ins[ti - 1]
                        if pn == 'jmp' and par[0] > tgt and par[0] <= end_addr and \
                                self.has_text(ti, h.idx[par[0]]):
                            pending_end[tgt].append('else')
                            pending_end[par[0]].append('endif')
                        else:
                            pending_end[tgt].append('endif')
            elif n == 'jmp':
                pass
            elif n == 'syscall':
                pop(h.syscalls[ar[0]][1])
            elif n == 'call':
                tgt = ar[0]
                args = pop(h.callee_argc(tgt))
                if tgt == F_TEXT:
                    text, size, speed, mode = args
                    if not isinstance(text, str):
                        out.append(('comment', 'text from expression: %s' % fmt_val(text)))
                        continue
                    t = clean_text(text)
                    lines_since_scene += 1
                    if speaker:
                        disp, canon, alt, voice = speaker
                        if mode in (None, 0):
                            t = '「%s」' % t
                        out.append(('line', disp, t, canon, alt, voice))
                        speaker = (disp, canon, alt, None)  # voice belongs to first chunk only
                    else:
                        out.append(('narr', t))
                elif tgt == F_WAIT:
                    speaker = None
                elif tgt in self.speakers:
                    disp, alt = self.speaker_name(tgt, args)
                    voice = args[0] if args and isinstance(args[0], int) and not isinstance(args[0], bool) else None
                    speaker = (disp, self.canon[tgt], alt, voice)
                elif tgt in self.bg:
                    out.append(('comment', 'bg: %s' % self.bg[tgt]))
                elif tgt in self.bgm:
                    out.append(('comment', 'bgm: %s' % self.bgm[tgt]))
                elif tgt == F_BGM_STOP:
                    out.append(('comment', 'bgm: stop'))
                elif tgt == F_CHOICE_ADD:
                    options.append(clean_text(args[0]) if isinstance(args[0], str) else fmt_val(args[0]))
                    # option index stored into a local slot? (push_stack k; pop_stack s)
                    for k in range(bi + 1, min(bi + 4, len(body))):
                        _, (xa, xn, xar) = body[k]
                        if xn == 'pop_stack':
                            slot_opt[xar[0]] = options[-1]
                            break
                elif tgt == F_CHOICE_SHOW:
                    out.append(('comment', 'choice: ' + ' | '.join(
                        '%d) %s' % (k + 1, o) for k, o in enumerate(options))))
                    self._map_dispatch(body, bi, options, slot_opt, branch_label)
                    options = []
                    slot_opt = {}
                elif tgt in self.chapter_of:
                    out.append(('comment', 'call: %s (%s) -- that file\'s text plays here' % self.chapter_of[tgt]))
                elif tgt in self.screens:
                    # a full-screen clear after some text = scene boundary (heuristic)
                    if lines_since_scene:
                        new_scene('screen clear')
                    out.append(('comment', 'screen: %s' % self.screens[tgt]))
                elif self.on_call is not None:
                    # used by voice_sprites.py; may append extra records to out, which render() ignores
                    self.on_call(tgt, args, out)
        for k in sorted(pending_end):
            for m in pending_end[k]:
                out.append(('comment', m))
        return out

    def _map_dispatch(self, body, bi, options, slot_opt, branch_label):
        """After the choice menu: '<local slot | const k> g100 sete jz X; jmp/call T' -> T is option."""
        k = bi + 1
        lim = min(len(body), bi + 80)
        while k < lim:
            seq = [body[j][1] for j in range(k, min(k + 5, lim))]
            if len(seq) == 5 and seq[2][1] == 'sete' and seq[3][1] == 'jz' and \
                    seq[4][1] in ('jmp', 'call'):
                lab = None
                x, y = seq[0], seq[1]
                if y[1] == 'push_global' and y[2][0] == G_CHOICE:
                    if x[1] == 'push_stack' and x[2][0] in slot_opt:
                        lab = slot_opt[x[2][0]]
                    elif x[1] == 'push_i8' and 1 <= x[2][0] <= len(options):
                        lab = options[x[2][0] - 1]
                elif x[1] == 'push_global' and x[2][0] == G_CHOICE and y[1] == 'push_i8' \
                        and 1 <= y[2][0] <= len(options):
                    lab = options[y[2][0] - 1]
                if lab is not None:
                    tgt = seq[4][2][0]
                    if seq[4][1] == 'jmp':
                        branch_label.setdefault(tgt, 'choice → %s' % lab)
                    k += 5
                    continue
            k += 1


def find_screens(h):
    """Screen-clear helpers: no args, set g<screen_flag>=1, call <screen>(nil, 1, <kind>, ...)."""
    res = {}
    for fa, (end, nargs, nl) in h.funcs.items():
        body = [x[1] for x in h.body(fa)]
        if nargs != 0 or len(body) > 40 or len(body) < 6:
            continue
        if not (body[1][1] == 'push_i8' and body[2][1] == 'pop_global' and body[2][2][0] == G_SCREEN_FLAG):
            continue
        if any(x[1] == 'push_str' for x in body):
            continue
        st = []
        for x in body:
            n, ar = x[1], x[2]
            if n in ('push_i8', 'push_i16', 'push_i32'):
                st.append(ar[0])
            elif n in ('push_nil', 'push_true', 'push_global', 'push_stack', 'push_str'):
                st.append(None)
            elif n == 'neg' and st and isinstance(st[-1], int):
                st[-1] = -st[-1]
            elif n == 'call':
                c = h.callee_argc(ar[0])
                args = st[len(st) - c:]
                del st[len(st) - c:]
                if ar[0] == F_SCREEN:
                    # 3rd arg = screen kind (-1, -2, ... one helper per kind)
                    res[fa] = 'clear kind=%s' % (args[2] if len(args) > 2 else '?')
                    break
            elif n in ('pop_global', 'pop_stack') and st:
                st.pop()
    return res


# ----------------------------------------------------------------------------- output
def render(recs, fn, title, fa, game_title='', source_name=''):
    L = ['# %s — %s' % (game_title, title),
         '# source: %s function 0x%06x' % (source_name, fa),
         '']
    for r in recs:
        k = r[0]
        if k == 'scene':
            L.append('')
            L.append('=== scene: %s-%03d ===' % (fn, r[1]))
        elif k == 'comment':
            L.append('# ' + r[1])
        elif k == 'narr':
            L.append(r[1])
        elif k == 'line':
            _, disp, t, canon, alt, voice = r
            meta = []
            if disp != canon:
                meta.append('char=%s' % canon)
            if alt != disp and alt != canon:
                meta.append('alt=%s' % alt)
            if voice is not None:
                meta.append('voice=%08d' % voice)
            L.append('【%s】%s%s' % (disp, t, ('  # ' + ' '.join(meta)) if meta else ''))
    return '\n'.join(L) + '\n'


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    prof_path, src, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    use_profile(prof_path)
    h = HCB(src, PROFILE.get('encoding', 'cp932'))
    ex = Extractor(h)
    os.makedirs(outdir, exist_ok=True)
    stats = {'lines': 0, 'narr': 0, 'speech': 0, 'by_disp': collections.Counter(),
             'by_canon': collections.Counter(), 'per_file': {}, 'canon_per_file': collections.defaultdict(collections.Counter)}
    ex.cache = {}
    for fa, fn, title in CHAPTERS:
        recs = ex.cache[fa] = ex.extract(fa)
        with open(os.path.join(outdir, fn + '.txt'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(render(recs, fn, title, fa, h.title, os.path.basename(src)))
        nl = sum(1 for r in recs if r[0] in ('narr', 'line'))
        stats['per_file'][fn] = (nl, sum(1 for r in recs if r[0] == 'scene'))
        for r in recs:
            if r[0] == 'narr':
                stats['narr'] += 1
            elif r[0] == 'line':
                stats['speech'] += 1
                stats['by_disp'][r[1]] += 1
                stats['by_canon'][r[3]] += 1
                stats['canon_per_file'][r[3]][fn] += 1
        stats['lines'] += nl
    # summary for README
    with open(os.path.join(outdir, 'stats.txt'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('total text lines: %d (narration %d, speech %d)\n' % (stats['lines'], stats['narr'], stats['speech']))
        f.write('\n[per file: text lines, scenes]\n')
        for fn, (nl, ns) in stats['per_file'].items():
            f.write('%s\t%d\t%d\n' % (fn, nl, ns))
        f.write('\n[speaker by character (canonical)]\n')
        for k, v in stats['by_canon'].most_common():
            f.write('%s\t%d\n' % (k, v))
        f.write('\n[speaker by displayed name plate]\n')
        for k, v in stats['by_disp'].most_common():
            f.write('%s\t%d\n' % (k, v))
        f.write('\n[character lines per file]\n')
        for c, cnt in sorted(stats['canon_per_file'].items(), key=lambda x: -sum(x[1].values())):
            f.write('%s\t%s\n' % (c, ', '.join('%s:%d' % (k, v) for k, v in sorted(cnt.items()))))
        f.write('\n[speaker functions]\n')
        for fa in sorted(ex.speakers):
            f.write('0x%06x id=%d %s\n' % (fa, ex.speakers[fa], ex.canon[fa]))
        f.write('\n[name plates used per speaker function: displayed (internal/alt name)]\n')
        seen = sorted({(fa, d, alt) for (fa, _), (d, alt) in ex.name_cache.items()})
        for fa, d, alt in seen:
            f.write('0x%06x %s -> %s (%s)\n' % (fa, ex.canon[fa], d, alt))
    # machine-readable companion: one JSON object per text line, story order per file
    import json
    with open(os.path.join(outdir, 'script.jsonl'), 'w', encoding='utf-8', newline='\n') as f:
        for fa, fn, title in CHAPTERS:
            scene = None
            for r in ex.cache[fa]:
                if r[0] == 'scene':
                    scene = '%s-%03d' % (fn, r[1])
                elif r[0] == 'narr':
                    f.write(json.dumps({'file': fn, 'scene': scene, 'type': 'narration', 'text': r[1]},
                                       ensure_ascii=False) + '\n')
                elif r[0] == 'line':
                    _, disp, t, canon, alt, voice = r
                    f.write(json.dumps({'file': fn, 'scene': scene, 'type': 'speech', 'name': disp,
                                        'char': canon, 'alt': alt, 'text': t, 'voice': voice},
                                       ensure_ascii=False) + '\n')
                elif r[0] == 'comment':
                    f.write(json.dumps({'file': fn, 'scene': scene, 'type': 'meta', 'text': r[1]},
                                       ensure_ascii=False) + '\n')
    print('done', stats['lines'])


if __name__ == '__main__':
    main()
