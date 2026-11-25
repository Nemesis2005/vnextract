# -*- coding: utf-8 -*-
"""Exact Qwen3.5-9B token counts for completed dataset conversations.

The conservative count is the larger of the stock-template rendering and a
source-preserving rendering that retains every historical <think> block.
"""
import functools
import os

from jinja2 import Environment

from audit_context import Qwen35Tokenizer, render_full_source, raise_exception

TOKEN_LIMIT = 8192
CHECKPOINT = 'Qwen/Qwen3.5-9B'
DEFAULT_TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen35_9b_tokenizer')


@functools.lru_cache(maxsize=4)
def _load(tokenizer_dir):
    required = ['tokenizer.json', 'chat_template.jinja']
    missing = [name for name in required if not os.path.isfile(os.path.join(tokenizer_dir, name))]
    if missing:
        raise RuntimeError('missing Qwen3.5 tokenizer assets in %s: %s' % (tokenizer_dir, ', '.join(missing)))
    tokenizer = Qwen35Tokenizer(os.path.join(tokenizer_dir, 'tokenizer.json'))
    with open(os.path.join(tokenizer_dir, 'chat_template.jinja'), encoding='utf-8') as f:
        template = Environment().from_string(f.read())
    return tokenizer, template


def token_counts(messages, tokenizer_dir=None):
    """Return stock, source-preserving, and conservative token counts."""
    tokenizer, template = _load(tokenizer_dir or DEFAULT_TOKENIZER_DIR)
    render_args = dict(tools=None, add_generation_prompt=False, add_vision_id=False,
                       raise_exception=raise_exception)
    stock_text = template.render(messages=messages, **render_args)
    source_text = render_full_source(messages)
    stock = len(tokenizer.encode(stock_text))
    source = len(tokenizer.encode(source_text))
    return dict(stock=stock, source=source, conservative=max(stock, source))


def conservative_token_count(messages, tokenizer_dir=None):
    return token_counts(messages, tokenizer_dir)['conservative']
