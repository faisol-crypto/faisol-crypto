"""Short-form captions built from a word-level transcript, on the FINAL timeline.

The transcript is in source time; each plan segment maps a source window onto
the edit, so captions stay in sync after cuts and reordering.

Plan block:
  "captions": {
    "transcript": "footage/report/x.transcript.json",   # from ramz.analyze --transcribe
    "replace": {"trick": "trade"},   # confirmed corrections (lower-case source word -> text)
    "omit": ["KKMC"],                # unclear words: never shown on screen until confirmed
    "max_words": 3, "uppercase": true, "y": 0.70
  }

Words below `min_prob` that are not in `replace` are listed in the review file
so they can be confirmed — captions never guess.
"""
import json
import os
import re

from .common import fmt_ts


def _clean(word):
    return re.sub(r"[^\w'-]", "", word.strip(), flags=re.UNICODE)


def _ass_time(t):
    t = max(t, 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _srt_time(t):
    t = max(t, 0.0)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build(cfg, transcript_path, segments, offsets, durs, W, H, font_family, out_stem):
    with open(transcript_path, encoding="utf-8") as fh:
        lines = json.load(fh)
    words = [w for ln in lines for w in ln.get("words", [])]
    replace = {k.lower(): v for k, v in cfg.get("replace", {}).items()}
    omit = {o.lower() for o in cfg.get("omit", [])}
    min_prob = float(cfg.get("min_prob", 0.35))
    max_words = int(cfg.get("max_words", 3))
    upper = cfg.get("uppercase", True)

    chunks, review = [], []
    for seg, off, dur in zip(segments, offsets, durs):
        if seg.get("captions") is False:
            continue
        seg_chunks = []
        cur, last_end = [], None
        for w in words:
            ws, we = float(w["s"]), float(w["e"])
            mid = (ws + min(we, ws + 0.5)) / 2
            if not (seg["in"] <= mid <= seg["out"]):
                continue
            raw = _clean(w["w"])
            key = raw.lower()
            if not raw:
                continue
            ts = off + max(ws, seg["in"]) - seg["in"]
            te = off + min(we, ws + 0.6, seg["out"]) - seg["in"]  # whisper pads word ends with silence
            if key in omit:
                review.append({"t": round(ts, 2), "word": raw, "action": "disembunyi (perlu pengesahan)"})
                text = None
            elif key in replace:
                text = replace[key]
                review.append({"t": round(ts, 2), "word": raw, "action": f"ditukar -> {text}"})
            else:
                text = raw
                if float(w.get("p", 1.0)) < min_prob:
                    review.append({"t": round(ts, 2), "word": raw, "action": f"keyakinan rendah ({w.get('p')})"})
            gap = last_end is not None and ts - last_end > 0.35
            punct = last_end is not None and cur and cur[-1].get("brk")
            if cur and (len(cur) >= max_words or gap or punct):
                seg_chunks.append(cur)
                cur = []
            if text:
                cur.append({"text": text, "s": ts, "e": te, "brk": bool(re.search(r"[,.?!]$", w["w"].strip()))})
            last_end = te
        if cur:
            seg_chunks.append(cur)
        # no orphan words: fold a lone word into the phrase it belongs to
        merged = []
        for ch in seg_chunks:
            close = merged and ch[0]["s"] - merged[-1][-1]["e"] < 0.45
            if close and len(ch) == 1 and len(merged[-1]) >= 3:
                ch = [merged[-1].pop()] + ch  # "GOLD TERLALU BANYAK" / "FAKE OUT"
            elif close and len(ch) == 1:
                merged[-1] = merged[-1] + ch
                continue
            merged.append(ch)
        chunks.extend(merged)

    events = []
    for i, ch in enumerate(chunks):
        start = ch[0]["s"]
        end = ch[-1]["e"] + 0.12
        if i + 1 < len(chunks):
            end = min(end + 0.25, chunks[i + 1][0]["s"])
        text = " ".join(c["text"] for c in ch)
        events.append((start, max(end, start + 0.3), text.upper() if upper else text))

    size = int(cfg.get("size", 0.058) * W)
    y = int(cfg.get("y", 0.70) * H)
    ass = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic,"
        " Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment,"
        " MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{font_family},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,"
        f"{max(3, size // 12)},{max(1, size // 30)},5,60,60,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for s, e, text in events:
        # quick scale pop-in: 88% -> 100% over 90 ms
        ass.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,"
                   f"{{\\pos({W // 2},{y})\\fscx88\\fscy88\\t(0,90,\\fscx100\\fscy100)}}{text}")
    ass_path = out_stem + ".captions.ass"
    with open(ass_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ass) + "\n")
    with open(out_stem + ".srt", "w", encoding="utf-8") as fh:
        for i, (s, e, text) in enumerate(events, 1):
            fh.write(f"{i}\n{_srt_time(s)} --> {_srt_time(e)}\n{text}\n\n")
    with open(out_stem + ".captions_review.md", "w", encoding="utf-8") as fh:
        fh.write("# Semakan caption\n\nPerkataan di bawah perlu disahkan. Betulkan melalui `captions.replace` "
                 "atau buang dari `captions.omit` dalam plan, kemudian render semula.\n\n")
        for r in review:
            fh.write(f"- {fmt_ts(r['t'])} — `{r['word']}` — {r['action']}\n")
    return ass_path, review
