"""Analyse raw footage before editing: the first step of hook engineering.

Produces (next to the footage, or in --out):
  <name>.analysis.json   — probe, loudness, silences, scene changes
  <name>.analysis.md     — readable report + suggested trim list
  <name>.contact.jpg     — contact sheet (one thumb every N seconds) to spot
                           the strongest visual for the hook
  <name>.transcript.json — only if faster-whisper is installed: word-level
                           transcript with filler words and hook-candidate lines

Usage:
  python3 -m ramz.analyze footage/raw.mp4 [--out reports/] [--transcribe]
"""
import argparse
import json
import os
import re

from .common import ffmpeg_stderr, fmt_ts, probe, require_tools, run

# Malaysian Malay + English fillers commonly heard in trading talk
FILLERS = {"erm", "err", "emm", "umm", "um", "uh", "aa", "aaa", "ha", "hmm", "ehh", "eh", "ok", "okay",
           "so", "basically", "actually", "macam", "kan", "lah", "ni", "tu"}
# words that often signal a strong opener in this niche — for human review only
HOOK_MARKERS = ["salah", "silap", "jangan", "kenapa", "rahsia", "ramai trader", "ramai orang", "sebenarnya",
                "ini sebab", "kalau korang", "stop loss", "kena", "tak pernah", "berhenti", "bahaya", "fake",
                "liquidity", "breakout", "trap", "mistake", "never", "why", "stop"]


def loudness(path):
    err = ffmpeg_stderr(["-i", path, "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    tail = err[err.rfind("Summary:"):]
    get = lambda k: float(m.group(1)) if (m := re.search(k + r":\s+(-?[\d.]+|-inf)", tail)) and m.group(1) != "-inf" else None
    return {"integrated_lufs": get("I"), "lra": get("LRA"), "true_peak_dbfs": get("Peak")}


def silences(path, noise_db=-35, min_dur=0.6):
    err = ffmpeg_stderr(["-i", path, "-vn", "-af", f"silencedetect=n={noise_db}dB:d={min_dur}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [(float(a), float(b)) for a, b in re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", err)]
    return [{"start": round(max(s, 0), 2), "end": round(e, 2), "dur": round(d, 2)} for s, (e, d) in zip(starts, ends)]


def scenes(path, thresh=0.3):
    err = ffmpeg_stderr(["-i", path, "-an", "-vf", f"select='gt(scene,{thresh})',showinfo", "-f", "null", "-"])
    return [round(float(t), 2) for t in re.findall(r"pts_time:([\d.]+)", err)]


def contact_sheet(path, out, every, duration):
    n = max(1, int(duration // every))
    cols = 5
    rows = (n + cols - 1) // cols
    run(["ffmpeg", "-y", "-hide_banner", "-i", path, "-vf",
         f"fps=1/{every},scale=360:-2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
         f"text='%{{pts\\:hms}}':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,"
         f"tile={cols}x{rows}:padding=6:margin=6", "-frames:v", "1", "-q:v", "3", out])


def transcribe(path, model_size="small"):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    import subprocess
    import numpy as np
    # decode ourselves: avoids PyAV/container quirks with phone .MOV files
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True).stdout
    model = WhisperModel(model_size, compute_type="int8")
    segs, _ = model.transcribe(np.frombuffer(pcm, np.float32), language="ms", word_timestamps=True,
                               beam_size=5)
    lines = []
    for s in segs:
        words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2), "p": round(w.probability, 2)}
                 for w in (s.words or [])]
        fillers = [w for w in words if re.sub(r"[^\w]", "", w["w"].lower()) in FILLERS]
        text = s.text.strip()
        score = sum(1 for m in HOOK_MARKERS if m in text.lower()) + (1 if "?" in text else 0)
        lines.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text,
                      "fillers": fillers, "hook_signal": score, "words": words})
    return lines


def main():
    ap = argparse.ArgumentParser(description="Analyse raw footage")
    ap.add_argument("input")
    ap.add_argument("--out", help="output directory (default: beside input)")
    ap.add_argument("--every", type=float, default=0, help="contact sheet interval seconds (auto)")
    ap.add_argument("--transcribe", action="store_true", help="use faster-whisper if installed")
    ap.add_argument("--model", default="medium", help="whisper size; medium is far better for Malay")
    args = ap.parse_args()
    require_tools()

    info = probe(args.input)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, os.path.splitext(os.path.basename(args.input))[0])
    dur = info["duration"]

    result = {"probe": info}
    if info["audio"]:
        result["loudness"] = loudness(args.input)
        result["silences"] = silences(args.input)
    result["scene_changes"] = scenes(args.input) if info["video"] else []
    every = args.every or max(1.0, round(dur / 30, 1))
    if info["video"]:
        contact_sheet(args.input, stem + ".contact.jpg", every, dur)
    transcript = transcribe(args.input, args.model) if args.transcribe else None
    if args.transcribe and transcript is None:
        print("[ramz] faster-whisper tiada — transcript dilangkau (pip install faster-whisper).")
    with open(stem + ".analysis.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    if transcript is not None:
        with open(stem + ".transcript.json", "w", encoding="utf-8") as fh:
            json.dump(transcript, fh, indent=2, ensure_ascii=False)

    v, a = info["video"], info["audio"]
    md = [f"# Analisis rakaman — {os.path.basename(args.input)}", "",
          f"- Durasi: **{dur:.2f}s**",
          f"- Video: {v['width']}x{v['height']} @ {v['fps']:.3f}fps ({v['codec']})" if v else "- Video: tiada",
          f"- Audio: {a['codec']} {a['sample_rate']}Hz {a['channels']}ch" if a else "- Audio: tiada"]
    if v and abs(v["fps"] - v["r_fps"]) > 0.01:
        md.append(f"- ⚠️ Frame rate berubah-ubah (VFR: avg {v['fps']:.3f} vs r {v['r_fps']:.3f}) — "
                  "render akan paksa CFR.")
    if a:
        L = result["loudness"]
        md.append(f"- Loudness: {L['integrated_lufs']} LUFS, true peak {L['true_peak_dbfs']} dBFS, LRA {L['lra']}")
        if L["true_peak_dbfs"] is not None and L["true_peak_dbfs"] > -0.5:
            md.append("- ⚠️ Peak hampir/melebihi 0 dBFS — kemungkinan clipping dalam rakaman asal.")
        sil = result["silences"]
        md += ["", "## Jeda panjang (calon trim)", "",
               "Jeda di bawah dikesan pada -35 dB ≥0.6s. Simpan ~0.25s jeda natural bila memotong.", ""]
        md += [f"- {fmt_ts(s['start'])} → {fmt_ts(s['end'])} ({s['dur']}s)" for s in sil] or ["- tiada"]
        removable = sum(max(s["dur"] - 0.25, 0) for s in sil)
        md.append(f"\nAnggaran masa boleh dijimatkan: **{removable:.1f}s** daripada {dur:.1f}s.")
    md += ["", "## Perubahan babak", "", ", ".join(fmt_ts(t) for t in result["scene_changes"]) or "tiada", "",
           "## Contact sheet", "", f"`{os.path.basename(stem)}.contact.jpg` — satu frame setiap {every}s.", ""]
    if transcript:
        md += ["## Calon hook (dari transkrip — perlu pengesahan manusia)", ""]
        for ln in sorted(transcript, key=lambda x: -x["hook_signal"])[:6]:
            if ln["hook_signal"]:
                md.append(f"- {fmt_ts(ln['start'])}–{fmt_ts(ln['end'])}: “{ln['text']}”")
        nf = sum(len(ln["fillers"]) for ln in transcript)
        md += ["", f"Filler words dikesan: **{nf}** (lihat transcript.json untuk timestamp).", ""]
    else:
        md += ["## Hook", "", "Tiada transkrip automatik. Tonton contact sheet + dengar rakaman, "
               "kemudian pilih ayat paling kuat yang **benar-benar wujud** dalam rakaman.", ""]
    with open(stem + ".analysis.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"[ramz] SIAP: {stem}.analysis.md")


if __name__ == "__main__":
    main()
