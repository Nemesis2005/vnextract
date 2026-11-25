#!/usr/bin/env python3
"""Audit a chat JSONL dataset's context lengths with the official Qwen3.5 tokenizer.

Uses only the downloaded tokenizer.json, chat_template.jinja, the Python standard
library, and Jinja2. The small BPE runner exists for environments where the
`tokenizers` native package is unavailable.
"""
import argparse
import collections
import functools
import hashlib
import heapq
import json
import math
import pathlib
import random
import re
import unicodedata

from jinja2 import Environment


def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~') + 1))
    bs += list(range(ord('¡'), ord('¬') + 1))
    bs += list(range(ord('®'), ord('ÿ') + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def is_l(ch):
    return unicodedata.category(ch).startswith('L')


def is_m(ch):
    return unicodedata.category(ch).startswith('M')


def is_n(ch):
    return unicodedata.category(ch).startswith('N')


def pretokenize(text):
    """Qwen3.5's Split pretokenizer, using unicodedata for Unicode properties."""
    out = []
    i = 0
    n = len(text)
    contractions = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
    while i < n:
        lower = text[i:i + 3].lower()
        hit = next((c for c in contractions if lower.startswith(c)), None)
        if hit is not None:
            out.append(text[i:i + len(hit)])
            i += len(hit)
            continue

        # [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        j = i
        if text[j] not in '\r\n' and not is_l(text[j]) and not is_n(text[j]):
            j += 1
        if j < n and (is_l(text[j]) or is_m(text[j])):
            j += 1
            while j < n and (is_l(text[j]) or is_m(text[j])):
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # \p{N}: one Unicode number character per token
        if is_n(text[i]):
            out.append(text[i])
            i += 1
            continue

        #  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        j = i + 1 if text[i] == ' ' else i
        start = j
        while j < n and not text[j].isspace() and not is_l(text[j]) and not is_m(text[j]) and not is_n(text[j]):
            j += 1
        if j > start:
            while j < n and text[j] in '\r\n':
                j += 1
            out.append(text[i:j])
            i = j
            continue

        # \s*[\r\n]+: greedy prefix leaves the final newline run; equivalently,
        # consume a whitespace run through its last newline.
        if text[i].isspace():
            j = i
            last_newline = -1
            while j < n and text[j].isspace():
                if text[j] in '\r\n':
                    last_newline = j
                j += 1
            if last_newline >= 0:
                out.append(text[i:last_newline + 1])
                i = last_newline + 1
                continue
            # \s+(?!\S): at end take all; otherwise take all but the last
            # whitespace in a run. The final alternative takes the remainder.
            run_len = j - i
            if j == n:
                out.append(text[i:j])
                i = j
                continue
            if run_len >= 2:
                out.append(text[i:j - 1])
                i = j - 1
                continue
            out.append(text[i:j])
            i = j
            continue

        raise AssertionError('unhandled pretokenizer input at %d: %r' % (i, text[i]))
    return out


class Qwen35Tokenizer:
    def __init__(self, tokenizer_path):
        data = json.loads(pathlib.Path(tokenizer_path).read_text(encoding='utf-8'))
        assert data['normalizer'] == {'type': 'NFC'}
        assert data['model']['type'] == 'BPE'
        assert not data['model']['ignore_merges']
        self.vocab = data['model']['vocab']
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for rank, merge in enumerate(data['model']['merges']):
            pair = tuple(merge.split(' ', 1)) if isinstance(merge, str) else tuple(merge)
            assert len(pair) == 2
            self.merges[pair] = rank
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.added = {x['content']: x['id'] for x in data['added_tokens']}
        for token, idx in self.added.items():
            self.id_to_token[idx] = token
        pieces = sorted(self.added, key=len, reverse=True)
        self.added_re = re.compile('(' + '|'.join(re.escape(x) for x in pieces) + ')')
        self.data = data

    @functools.lru_cache(maxsize=500000)
    def bpe(self, word):
        symbols = [self.byte_encoder[b] for b in word.encode('utf-8')]
        if len(symbols) < 2:
            return tuple(symbols)
        nxt = list(range(1, len(symbols))) + [-1]
        prv = [-1] + list(range(len(symbols) - 1))
        alive = [True] * len(symbols)
        heap = []
        for i in range(len(symbols) - 1):
            rank = self.merges.get((symbols[i], symbols[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))
        while heap:
            rank, i = heapq.heappop(heap)
            j = nxt[i]
            if not alive[i] or j < 0 or not alive[j]:
                continue
            if self.merges.get((symbols[i], symbols[j])) != rank:
                continue
            symbols[i] += symbols[j]
            alive[j] = False
            nxt[i] = nxt[j]
            if nxt[i] >= 0:
                prv[nxt[i]] = i
            p = prv[i]
            if p >= 0:
                r = self.merges.get((symbols[p], symbols[i]))
                if r is not None:
                    heapq.heappush(heap, (r, p))
            j = nxt[i]
            if j >= 0:
                r = self.merges.get((symbols[i], symbols[j]))
                if r is not None:
                    heapq.heappush(heap, (r, i))
        return tuple(x for i, x in enumerate(symbols) if alive[i])

    def encode_ordinary(self, text):
        text = unicodedata.normalize('NFC', text)
        ids = []
        for word in pretokenize(text):
            for token in self.bpe(word):
                ids.append(self.vocab[token])
        return ids

    def encode(self, text):
        ids = []
        for chunk in self.added_re.split(text):
            if not chunk:
                continue
            if chunk in self.added:
                ids.append(self.added[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids):
        out = []
        buf = bytearray()
        for idx in ids:
            token = self.id_to_token[idx]
            if token in self.added:
                if buf:
                    out.append(buf.decode('utf-8'))
                    buf.clear()
                out.append(token)
            else:
                buf.extend(self.byte_decoder[ch] for ch in token)
        if buf:
            out.append(buf.decode('utf-8'))
        return ''.join(out)


def raise_exception(message):
    raise ValueError(message)


IM_START = '\x3c|im_start|>'
IM_END = '\x3c|im_end|>'


def render_full_source(messages):
    """Conservative upper bound: retain all roles and every historical think block."""
    chunks = []
    for message in messages:
        chunks.append(IM_START + message['role'] + '\n' + message['content'].strip() + IM_END + '\n')
    return ''.join(chunks)


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize(records, key, limit):
    values = [r[key] for r in records if r[key] is not None]
    if not values:
        return {'documents': 0}
    return collections.OrderedDict(
        documents=len(values),
        total_tokens=sum(values),
        min=min(values),
        p50=percentile(values, .50),
        p90=percentile(values, .90),
        p95=percentile(values, .95),
        p99=percentile(values, .99),
        max=max(values),
        min_headroom=limit - max(values),
        over_2048=sum(x > 2048 for x in values),
        over_4096=sum(x > 4096 for x in values),
        over_8192=sum(x > 8192 for x in values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=8192)
    args = parser.parse_args()

    tokenizer_dir = pathlib.Path(args.tokenizer_dir)
    output = pathlib.Path(args.output)
    records = [json.loads(line) for line in pathlib.Path(args.dataset).read_text(encoding='utf-8').splitlines() if line]
    tokenizer = Qwen35Tokenizer(tokenizer_dir / 'tokenizer.json')
    template_text = (tokenizer_dir / 'chat_template.jinja').read_text(encoding='utf-8')
    config = json.loads((tokenizer_dir / 'config.json').read_text(encoding='utf-8'))
    env = Environment(trim_blocks=False, lstrip_blocks=False)
    official = env.from_string(template_text)

    old_system = """{%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}"""
    new_system = """{%- if not loop.first %}
            {{- '""" + IM_START + """' + message.role + '\\n' + content + '""" + IM_END + """' + '\\n' }}
        {%- endif %}"""
    assert template_text.count(old_system) == 1
    patched_text = template_text.replace(old_system, new_system)
    old_no_user = """{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}"""
    new_no_user = """{%- if ns.multi_step_tool %}
    {%- set ns.last_query_index = -1 %}
{%- endif %}"""
    assert patched_text.count(old_no_user) == 1
    patched = env.from_string(patched_text.replace(old_no_user, new_no_user))
    render_args = dict(tools=None, add_generation_prompt=False, add_vision_id=False,
                       raise_exception=raise_exception)

    # Pretokenizer differential test against Python re for the ASCII subset.
    ascii_re = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
    rng = random.Random(35)
    alphabet = "abcXYZ019 .,'\n\t!?_"
    for _ in range(1000):
        sample = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(80)))
        assert pretokenize(sample) == ascii_re.findall(sample), repr(sample)
    assert tokenizer.added[IM_START] == 248045
    assert tokenizer.added[IM_END] == 248046
    for sample in ['hello world', '今日はいい天気ですね。',
                   '「ちょっと待って……それ、本当なの？」',
                   '<narration>私は窓の外を見る。</narration>']:
        assert tokenizer.decode(tokenizer.encode(sample)) == unicodedata.normalize('NFC', sample)

    audited = []
    for rec in records:
        messages = rec['messages']
        full_text = render_full_source(messages)
        patched_text = patched.render(messages=messages, **render_args)
        official_error = None
        official_text = None
        try:
            official_text = official.render(messages=messages, **render_args)
        except ValueError as exc:
            official_error = str(exc)
        full_ids = tokenizer.encode(full_text)
        patched_ids = tokenizer.encode(patched_text)
        official_ids = tokenizer.encode(official_text) if official_text is not None else None
        # Round-trip every rendered record, covering Japanese, tags, whitespace,
        # special tokens, and all source text rather than just isolated samples.
        assert tokenizer.decode(full_ids) == unicodedata.normalize('NFC', full_text), rec['id']
        assert tokenizer.decode(patched_ids) == unicodedata.normalize('NFC', patched_text), rec['id']
        if official_ids is not None:
            assert tokenizer.decode(official_ids) == unicodedata.normalize('NFC', official_text), rec['id']
        audited.append(collections.OrderedDict(
            id=rec['id'], source_file=rec['meta']['source_file'],
            chars=sum(len(m['content']) for m in messages),
            official_tokens=len(official_ids) if official_ids is not None else None,
            official_error=official_error,
            patched_tokens=len(patched_ids),
            full_source_tokens=len(full_ids),
            full_source_headroom=args.limit - len(full_ids),
            conservative_tokens=max(len(patched_ids), len(full_ids)),
            conservative_headroom=args.limit - max(len(patched_ids), len(full_ids)),
        ))

    mid_docs = [r for r in records if any(m['role'] == 'system' for m in r['messages'][1:])]
    think_turns = sum('<think>' in m['content'] for r in records for m in r['messages'] if m['role'] == 'assistant')
    report = collections.OrderedDict(
        checkpoint='Qwen/Qwen3.5-9B',
        training_limit=args.limit,
        native_context=config['text_config']['max_position_embeddings'],
        tokenizer_sha256=hashlib.sha256((tokenizer_dir / 'tokenizer.json').read_bytes()).hexdigest(),
        template_sha256=hashlib.sha256((tokenizer_dir / 'chat_template.jinja').read_bytes()).hexdigest(),
        tokenizer_validation='100% of tokenizer.json loaded; 1000 ASCII differential tests; all rendered records round-tripped',
        modes={
            'official_stock_template': summarize(audited, 'official_tokens', args.limit),
            'system_compatible_template': summarize(audited, 'patched_tokens', args.limit),
            'full_source_upper_bound': summarize(audited, 'full_source_tokens', args.limit),
            'conservative_worst_case': summarize(audited, 'conservative_tokens', args.limit),
        },
        template_compatibility={
            'stock_errors': sum(r['official_error'] is not None for r in audited),
            'stock_error_types': dict(collections.Counter(r['official_error'] for r in audited if r['official_error'] is not None)),
            'no_user_documents': sum(not any(m['role'] == 'user' for m in r['messages']) for r in records),
            'mid_conversation_system_documents': len(mid_docs),
            'mid_conversation_system_messages': sum(sum(m['role'] == 'system' for m in r['messages'][1:]) for r in records),
            'think_documents': sum(any('<think>' in m['content'] for m in r['messages'] if m['role'] == 'assistant') for r in records),
            'think_turns': think_turns,
            'final_assistant_think_turns': sum('<think>' in r['messages'][-1]['content'] for r in records),
            'historical_think_turns_normally_omitted': think_turns - sum('<think>' in r['messages'][-1]['content'] for r in records),
        },
        scene_61={
            'summary_conservative': summarize([r for r in audited if r['source_file'].startswith('61_')], 'conservative_tokens', args.limit),
            'records': [r for r in audited if r['source_file'].startswith('61_')],
        },
        longest_conservative=sorted(audited, key=lambda r: r['conservative_tokens'], reverse=True)[:15],
        records=audited,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    lines = ['# Qwen3.5-9B context audit', '',
             'Training limit: **%d tokens**. Native model context: **%d tokens**.' % (args.limit, report['native_context']),
             'All counts use the official tokenizer and chat rendering, with no generation prompt.', '',
             '| Rendering | Documents | p95 | p99 | Max | Min headroom | >4096 | >8192 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    labels = [('official_stock_template', 'Official stock template'),
              ('system_compatible_template', 'Compatibility-patched template'),
              ('full_source_upper_bound', 'Source-preserving rendering'),
              ('conservative_worst_case', 'Conservative per-record maximum')]
    for key, label in labels:
        s = report['modes'][key]
        lines.append('| %s | %d | %d | %d | %d | %d | %d | %d |' % (
            label, s['documents'], s['p95'], s['p99'], s['max'], s['min_headroom'], s['over_4096'], s['over_8192']))
    lines += ['', 'Stock-template errors: **%d documents** (%d mid-conversation system errors; %d documents without a user query).' % (
                  report['template_compatibility']['stock_errors'],
                  report['template_compatibility']['stock_error_types'].get('System message must be at the beginning.', 0),
                  report['template_compatibility']['no_user_documents']),
              'Historical think turns omitted by the official reasoning handling: **%d**.' % report['template_compatibility']['historical_think_turns_normally_omitted'],
              '', '## Longest records (conservative per-record maximum)', '',
              '| ID | Tokens | Headroom | Characters |', '|---|---:|---:|---:|']
    for r in report['longest_conservative'][:10]:
        lines.append('| `%s` | %d | %d | %d |' % (r['id'], r['conservative_tokens'], r['conservative_headroom'], r['chars']))
    lines += ['', '## Scene 61', '', '| ID | Conservative tokens | Headroom | Stock tokens |',
              '|---|---:|---:|---:|']
    for r in report['scene_61']['records']:
        lines.append('| `%s` | %d | %d | %s |' % (
            r['id'], r['conservative_tokens'], r['conservative_headroom'],
            str(r['official_tokens']) if r['official_tokens'] is not None else 'template error'))
    md = output.with_suffix('.md')
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'modes': report['modes'], 'template_compatibility': report['template_compatibility'],
                      'scene_61': report['scene_61']['summary_conservative']}, ensure_ascii=False, indent=2))
    print('wrote', output)
    print('wrote', md)


if __name__ == '__main__':
    main()
