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


# words that lean on the previous word: never start a caption line with them
CLITICS = {"out", "lah", "je", "kan", "tu", "guys", "pun", "ni"}


def balance(phrase, max_words):
    """Split a phrase into caption lines.

    Exhaustive over break points (phrases are short): minimise the longest
    line in characters, never start a line with a clitic (lah, je, out ...),
    avoid one-word lines, and respect max_words.
    """
    n = len(phrase)
    if n <= max_words:
        return [phrase]
    lens = [len(w["text"]) for w in phrase]
    best = None

    def line_cost(i, j):
        chars = sum(lens[i:j]) + (j - i - 1)
        c = chars
        if phrase[i]["key"] in CLITICS:
            c += 60
        if j - i == 1:
            c += 25
        return c

    k = -(-n // max_words)  # fewest lines that respect max_words

    def search(i, parts):
        nonlocal best
        if len(parts) > k:
            return
        if i == n:
            # tie-break: longer line first reads more naturally
            cost = max(line_cost(a, b) for a, b in parts) + 0.01 * line_cost(*parts[-1])
            if best is None or cost < best[0]:
                best = (cost, list(parts))
            return
        for j in range(i + 1, min(n, i + max_words) + 1):
            parts.append((i, j))
            search(j, parts)
            parts.pop()

    search(0, [])
    return [phrase[a:b] for a, b in best[1]]

def build(cfg, transcript_path, segments, offsets, durs, W, H, font_family, out_stem):
    with open(transcript_path, encoding="utf-8") as fh:
        lines = json.load(fh)
    words = [dict(w, line=li) for li, ln in enumerate(lines) for w in ln.get("words", [])]
    drop_below = float(cfg.get("drop_below", 0.05))
    replace = {k.lower(): v for k, v in cfg.get("replace", {}).items()}
    omit = {o.lower() for o in cfg.get("omit", [])}
    highlight = {h.lower() for h in cfg.get("highlight", [])}
    hl_col = cfg.get("highlight_color", "#D9B56A").lstrip("#")
    hl_tag = "{\\c&H" + hl_col[4:6] + hl_col[2:4] + hl_col[0:2] + "&}"
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
            # word belongs to the segment if its centre is inside, or it starts clearly inside
            # (Whisper often stretches the last word of a phrase into the following silence)
            if not (seg["in"] <= mid <= seg["out"] or seg["in"] <= ws <= seg["out"] - 0.08):
                continue
            raw = _clean(w["w"])
            key = raw.lower()
            if not raw:
                continue
            ts = off + max(ws, seg["in"]) - seg["in"]
            te = off + min(we, ws + 0.6, seg["out"]) - seg["in"]  # whisper pads word ends with silence
            if key not in replace and key not in highlight and float(w.get("p", 1.0)) < drop_below:
                review.append({"t": round(ts, 2), "word": raw, "action": f"dibuang — hampir pasti bukan pertuturan ({w.get('p')})"})
                text = None
            elif key in omit:
                review.append({"t": round(ts, 2), "word": raw, "action": "disembunyi (perlu pengesahan)"})
                text = None
            elif key in replace:
                text = replace[key]
                review.append({"t": round(ts, 2), "word": raw, "action": f"ditukar -> {text}"})
            else:
                text = raw
                if float(w.get("p", 1.0)) < min_prob:
                    review.append({"t": round(ts, 2), "word": raw, "action": f"keyakinan rendah ({w.get('p')})"})
            # phrases break only on pauses, punctuation or a new transcript line
            gap = last_end is not None and ts - last_end > 0.35
            punct = last_end is not None and cur and (cur[-1].get("brk") or cur[-1].get("line") != w["line"])
            if cur and (gap or punct):
                seg_chunks.append(cur)
                cur = []
            if text:
                cur.append({"text": text, "s": ts, "e": te, "line": w["line"], "key": key,
                            "hl": key in highlight or text.lower() in highlight,
                            "brk": bool(re.search(r"[,.?!]$", w["w"].strip()))})
            last_end = te
        if cur:
            seg_chunks.append(cur)
        # a lone short word ("eh", "tu") flashing for a few frames is unreadable:
        # fold it into the next phrase when that follows straight after
        merged = []
        for ph in seg_chunks:
            if merged and len(merged[-1]) == 1 and merged[-1][0]["e"] - merged[-1][0]["s"] < 0.35 \
                    and ph[0]["s"] - merged[-1][-1]["e"] < 0.3:
                merged[-1] = merged[-1] + ph
            else:
                merged.append(ph)
        for phrase in merged:
            chunks.extend(balance(phrase, max_words))

    events = []
    for i, ch in enumerate(chunks):
        start = ch[0]["s"]
        end = ch[-1]["e"] + 0.12
        if i + 1 < len(chunks):
            end = min(end + 0.25, chunks[i + 1][0]["s"])
        words_out = [(c["text"].upper() if upper else c["text"], c.get("hl")) for c in ch]
        plain = " ".join(w for w, _ in words_out)
        styled = " ".join(f"{hl_tag}{w}{{\\c}}" if hl else w for w, hl in words_out)
        end = max(end, start + 0.3)
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0]["s"])  # never two captions on screen at once
        events.append((start, end, plain, styled))

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
    for s, e, _, styled in events:
        # restrained pop-in: 94% -> 100% with a short fade
        ass.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Cap,,0,0,0,,"
                   f"{{\\pos({W // 2},{y})\\fad(70,50)\\fscx94\\fscy94\\t(0,120,\\fscx100\\fscy100)}}{styled}")
    ass_path = out_stem + ".captions.ass"
    with open(ass_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(ass) + "\n")
    with open(out_stem + ".srt", "w", encoding="utf-8") as fh:
        for i, (s, e, text, _) in enumerate(events, 1):
            fh.write(f"{i}\n{_srt_time(s)} --> {_srt_time(e)}\n{text}\n\n")
    with open(out_stem + ".captions_review.md", "w", encoding="utf-8") as fh:
        fh.write("# Semakan caption\n\nPerkataan di bawah perlu disahkan. Betulkan melalui `captions.replace` "
                 "atau buang dari `captions.omit` dalam plan, kemudian render semula.\n\n")
        for r in review:
            fh.write(f"- {fmt_ts(r['t'])} — `{r['word']}` — {r['action']}\n")
    return ass_path, review
