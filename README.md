# vnextract

Tools for turning a visual novel you own into training data:

1. **Extract** the full scenario script from the game files into a normalized, story-ordered text
   format (speakers, narration, choices, branches).
2. **Convert** one character's scenes into a character-roleplay chat dataset (chat-format JSONL with
   that character as the assistant) for fine-tuning Qwen or other small models, validated verbatim
   against the script.
3. **Voice**: link each voiced line to its context and on-screen sprite, measure tone of voice, label
   emotion styles, and build a Style-Bert-VITS2 training folder.

The tools are standard-library Python 3.8+ (except the optional emotion model), read the game files
only, and never modify them. The FAVORITE View Point (FVP) engine is supported for extraction; steps 2
and 3 work for any engine once its script is in the normalized format.

This repository contains only the tools. Nothing from any game is included: no game files, extracted
text, profiles, character configs, annotations, datasets, audio or sprites.

## Layout

```
tools/
  fvp/                     FVP engine (FAVORITE View Point: one .hcb bytecode file + .bin archives)
    fvp_hcb.py             .hcb bytecode → normalized script (driven by a per-game profile)
    fvp_archive.py         .bin archive index and hzc1/NVSG image decoding
    extract_sprites.py     sprite bodies, face strips, composites and face contact sheets
    voice_sprites.py       voiced line → context, sprite pose/outfit/face, style labels
  voice/                   engine-independent voice tools (ffmpeg on PATH)
    audio_features.py      loudness / pitch / intonation / rate per clip
    voice_emotion.py       emotion2vec+ class scores per clip (needs numpy, torch, funasr)
    label_sheet.py         labelling sheet: context + voice + face per line
    category_review.py     review sheet for one style (optionally against another)
    listen_clips.py        WAV clips + CSV sheet for a person to listen to and judge
    build_sbv2.py          Style-Bert-VITS2 raw/<Style>/*.wav + esd.list from labelled lines
  dataset/
    character_dataset.py   normalized script + character config + annotations → chat JSONL
    qwen35_validation.py   exact Qwen3.5 token counts for the validator
    audit_context.py       full context-length audit with the official Qwen3.5 tokenizer
```

Local-only folders (git-ignored; keep your own per-game material there or anywhere else):
`games/<title>/` for profiles, character configs, annotations and notes, and `skills/` for Claude Code
workflow skills. The Qwen3.5 tokenizer is not committed either: put `tokenizer.json`,
`tokenizer_config.json`, `config.json` and `chat_template.jinja` from the `Qwen/Qwen3.5-9B` model
repository into `tools/dataset/qwen35_9b_tokenizer/`, or pass `--tokenizer DIR`.

Every tool prints its usage when run without arguments. Run Python with `-I`.

## Workflow

Work from a folder next to the game, e.g. `<Game>\Extracted\` (the "project"). The tools write
`script\` and `dataset\` there. `<vn>` is this repository.

```
# 1. script → Extracted\script\
python -I <vn>\tools\fvp\fvp_hcb.py <profile.json> ..\<Name>.hcb script

# 2. chat dataset → Extracted\dataset\<OUTPUT>
python -I <vn>\tools\dataset\character_dataset.py --config <config.py> --project . plan    # worksheets for annotators
python -I <vn>\tools\dataset\character_dataset.py --config <config.py> --project . build   # assemble + validate

# 3. voice
python -I <vn>\tools\fvp\extract_sprites.py <profile.json> .. sprites <char>
python -I <vn>\tools\fvp\voice_sprites.py <profile.json> ..\<Name>.hcb dataset\voice --char <char> --emotions <emotions dir>
python -I <vn>\tools\voice\audio_features.py dataset\voice\<char>_voice_sprites.jsonl ..\voice.bin dataset\voice\<char>_audio.tsv
python -I <vn>\tools\voice\label_sheet.py <char> dataset\voice\label_sheets\batch_001.md --emotions <emotions dir> --unlabelled
python -I <vn>\tools\voice\build_sbv2.py dataset\voice\<char>_voice_sprites.jsonl ..\voice.bin <SBV2>\Data\<Speaker> --speaker <Speaker>
```

## Normalized script format

One UTF-8 file per chapter or route, `NN_<area>.txt`, named so that sorting gives story order, plus
`script.jsonl` (one record per line, same order) and `stats.txt`. Line numbers in the `.txt` files are
the reference for every annotation, so never re-extract in a way that shifts them.

```
# <game title> — <chapter title>                   file header
=== scene: 01_common_ch01-002 ===                  scene boundary
# bg: <location label>                             background change
# bgm: 25   /   # bgm: stop                        BGM track or stop
# screen: clear kind=-1                            full-screen clear (scene boundaries follow these)
【Name】「Speech.」  # char=<real name> alt=<internal name> voice=00000020
Narration has no prefix.
# choice: 1) A | 2) B                              choice menu
# branch: choice → B                               text from here on runs after option B
# if (g10 == 3) … # else … # endif                 conditional text (raw condition)
# call: <file stem> (<title>)                      that file's text plays at this point
```

- `【plate】` is the name plate as shown; `char=` gives the real speaker when the plate differs (e.g. an
  unknown-speaker plate). Select a character's lines by `char`, not by plate.
- The engine's 「」 are kept as the game shows them; lines with their own 『』 or （） keep those.
- Ruby `[reading|base]` is reduced to the base text.
- `script.jsonl`: `{"file","scene","type":"narration|speech|meta","name","char","alt","text","voice"}`.

## FVP profile

A JSON file per game. Addresses are strings (hex allowed), globals are numbers. Find each value by
disassembling the `.hcb` (`fvp_hcb.HCB(path).ins` is the full instruction list) and confirm it by
reading the code around it.

```json
{
  "title": "<game title>", "hcb": "<Name>.hcb", "encoding": "cp932",
  "functions": {
    "text": "0x…",        "wait": "0x…",         "name": "0x…",
    "choice_add": "0x…",  "choice_show": "0x…",  "bg_label": "0x…",
    "bgm_play": "0x…",    "bgm_stop": "0x…",     "screen": "0x…"
  },
  "globals": {"speaker": 0, "choice": 0, "screen_flag": 0},
  "chapters": [["0x…", "01_common_ch01", "<chapter title>"]],
  "sprites": {
    "archive": "<sprite archive>.bin", "loader": "0x…", "hide": "0x…", "cg_show": "0x…",
    "name_prefix": "<archive>/<prefix>", "face_suffix": "<face strip suffix>",
    "arg_char": -14, "arg_pose": -13, "arg_outfit": -12,
    "size_suffix": {"0": "L", "1": "", "2": "S", "-1": "LL"},
    "no_size_outfits": [], "cg_ignore_prefix": "BG"
  }
}
```

| key | what to find |
|---|---|
| `functions.text`, `functions.wait` | the text function (most-called, string first argument, 4 args) and the page wait called after it |
| `functions.name`, `globals.speaker` | the name-plate function and the current-speaker global; speaker functions are then found automatically |
| `functions.choice_add`, `choice_show`, `globals.choice` | add option, show menu, result global |
| `functions.bg_label`, `bgm_play`, `bgm_stop` | background label function, BGM play and stop |
| `globals.screen_flag`, `functions.screen` | the screen-clear helpers' flag global and fade function |
| `chapters` | every chapter, route and inline-scene entry point; a debug chapter menu, if the game has one, lists them |
| `sprites.*` | only for the sprite and voice tools: the sprite loader, its argument slots, hide and CG functions, sprite naming |

Check the result: the number printed at the end equals the number of text-function call sites, there
are no U+FFFD characters, and `stats.txt` names every main character once under `[speaker functions]`.

## Character config

`character_dataset.py --config` loads a Python module. Required:

| name | meaning |
|---|---|
| `TARGET` | the assistant character's canonical name (`char` field) |
| `PROTAGONIST` | the default narrator, i.e. the user side |
| `FIRST_PERSON` | the target's first-person pronoun; replaces the target's name when their narration moves into their turn |
| `NARRATOR_PRONOUNS` | regex of the narrator's first-person pronouns, replaced by `narrator_name()` |
| `OUTPUT` | output file name under `<project>/dataset/` |
| `PERSONA` | persona card at the start of every system message |
| `POV_SCENES` | scene ids told from the target's point of view (their narration becomes `<think>`) |
| `route_of(stem)` | route name of a script file |
| `narrator_name(stem, line)` | what the narrator's pronoun becomes in the target's narration |
| `narration_desc(pov, narrator)` | the narration line of the scene header |
| `route_context(doc, annotation)` | route note for the scene header, or None |
| `MERGE_GROUPS`, `SPANS`, `BRANCH_SCENES`, `CONTEXT_FROM`, `PATH_CHOICE`, `TRIM_FILLER_END`, `POV_TRAILING_TO_RECAP`, `POV_BRACKET_KEEP`, `COND_NOTES`, `LOCATION_OVERRIDES` | segmentation fixes and overrides found by reading the script (may be empty) |

Optional: `ROLE_KEY` (key of the target's lines in `annotations/narration_roles.json`, default
`target`), `COND_GLOBAL` (choice global whose `== N` conditions are enumerated per value),
`include_file(stem)`, `CUTAWAY_BG`, `UNKNOWN_PLATES`, `SKIP_DOCS`, `FILLER_END_OK`, `ANN_DIR`, and any
tunable (`GAP_SPLIT`, `LEAD_CHARS`, `LEAD_MAX`, `PART_BUDGET`, `NARR_CONDENSE`, `NARR_EST`,
`TOKEN_LIMIT`). Keep tunables fixed once annotation has started: document ids depend on them.

Annotations (`annotations/*.json` next to the config) are written by people or agents from the
worksheets that `plan` produces: per document `situation`, `recap`, `present`, `narrator`, `opening`,
`location`, and edit keys (`drop`, `relabel`, `as_note`, `condense_ranges`, `insert_before/after`, …).

## Voice labelling rules

Each line's style is decided from three signals, never from the face alone:

1. **Context**: the lines around it and the narration that describes how it was said.
2. **Tone of voice**: listening wins (basis `L`); otherwise `audio_features.py` and `voice_emotion.py`
   z-scores against the character's own median. They separate loud/high/wide from quiet/low/flat, and
   nothing finer.
3. **Face**: the sprite face on screen, as a hint only.

Label a line only when context and voice agree. Put anything doubtful on a listen list
(`listen_clips.py`) for a person to decide. Labels go to `<emotions>/<char>_labels.tsv` as
`voice, style, basis, note[, intensity]`. Styles, ACT emotions and the Style-Bert-VITS2 folder each
style trains under are in `<char>_styles.tsv`; faces in `<char>.tsv`; poses in `<char>_motions.tsv`.
`build_sbv2.py` only takes clips of 2–12 s whose text has actual characters.

## License

MIT (see `LICENSE`). The license covers the tools in this repository only. Use them only with games
you own; the games and their content belong to their publishers.
