# SIR RAMZ — Video Production Toolkit

Editing pipeline for SIR RAMZ videos (XAUUSD, market structure, price action), built on **ffmpeg + Python**.
The full standard is in [`docs/PRODUCTION_SYSTEM.md`](docs/PRODUCTION_SYSTEM.md).

**HOOK FIRST. STORY SECOND. EFFECTS WITH PURPOSE. QUALITY THROUGHOUT.**

## Setup

```bash
# ffmpeg 6+ is required
pip install "opencv-python-headless<5" numpy   # for ramz.privacy
pip install faster-whisper                      # optional: transcript + hook candidates
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # optional: ramz.align
python3 -m ramz.sfx                             # generate the starter SFX kit -> assets/sfx/
```

## Workflow

```bash
# 1. Put the footage in footage/ (download it from Google Drive first)
python3 -m ramz.analyze footage/raw.mp4 --transcribe
# 2. Face privacy — ALWAYS before editing
python3 -m ramz.privacy footage/raw.mp4 -o footage/protected.mp4 [--regions manual.json] [--method soft|pixel|solid]
# 3. Copy plans/example_plan.json -> plans/<video>.json, then set the hook, segments, text, annotations and SFX
#    Optional: give each segment its spoken "text" and force-align it for frame-accurate captions
python3 -m ramz.align plans/<video>.json -o footage/report/<video>.forced.json   # -> captions.transcript
python3 -m ramz.render plans/<video>.json
# 4. QC
python3 -m ramz.qc out/final.mp4 --plan plans/<video>.json
```

The render writes `final.mp4` (captions), `final_clean.mp4` (no captions), `final.edl` (CMX3600 for Resolve/Premiere), `final.srt`, `final.timeline.md` (a per-scene timeline: visual, audio, animation, motion, FX, colour, transition, creative purpose) and `final.edl.json`.

## Modules

| Module | What it does |
|---|---|
| `ramz/analyze.py` | loudness, long pauses (trim candidates), scene changes, contact sheet, transcript + filler words (optional) |
| `ramz/privacy.py` | two-pass face tracking (bidirectional interpolation), feathered blur/mosaic/solid, HDR→SDR + CFR, coverage & detail-leak audit |
| `ramz/align.py` | forced alignment (MMS CTC) of each segment's known `speech` → word timings that don't drift into pauses; acronyms aligned on their spoken form |
| `ramz/captions.py` | short-form captions (2–3 words) from the transcript, remapped to the final timeline; unclear words hidden until confirmed |
| `ramz/render.py` | reorders the hook, fits to 9:16 (crop/blur_pad/pad), sub-pixel zoom, colour grades, controlled FX, text animation, chart annotations, SFX, music ducking, 2-pass loudness, subtitles |
| `ramz/motion.py` | bespoke motion graphics (Pillow, 2x supersampled): title, strike, illustrative diagram, compare card, chip list, versus card, statement, keyword, end card |
| `ramz/music.py` | original procedural ambient bed (royalty-free) |
| `ramz/sfx.py` | procedural starter SFX kit (whoosh, impact, riser, click, blip, typing, glitch, ambient) |
| `ramz/qc.py` | measured QC plus a manual checklist — no automatic score |

## Testing without real footage

```bash
./tests/make_synthetic.sh
python3 -m ramz.privacy tests/out/synthetic_raw.mp4 -o tests/out/synthetic_protected.mp4 --regions tests/manual_regions.json
python3 -m ramz.sfx && python3 -m ramz.render plans/example_plan.json
python3 -m ramz.qc tests/out/final_example.mp4 --plan plans/example_plan.json
```
