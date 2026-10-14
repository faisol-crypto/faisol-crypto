"""Forced alignment of the plan's known dialogue -> word-level caption transcript.

Whisper word timestamps drift into pauses (a word "starts" in the silence
before it), so cuts made on real speech boundaries drop or mistime captions.
When each segment's spoken text is known (segment "text" in the plan, checked
by ear or against the transcript), this aligns that exact text to the audio of
the segment with the multilingual MMS CTC aligner (torchaudio). Malay is Latin
script, so it aligns natively; acronyms are aligned on their spoken form
(SNR -> "esenar") but displayed as written.

Output has the same shape as `ramz.analyze --transcribe` (source time), so it
plugs straight into `captions.transcript`.

Usage:
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  python3 -m ramz.align plans/video.json -o footage/report/video.forced.json
"""
import argparse
import json
import os
import re
import subprocess
import tempfile

import numpy as np

from .common import load_json, resolve

# spoken form of common trading acronyms (letters as pronounced in Malay)
SPOKEN = {
    "snr": "esenar", "snd": "esendi", "ict": "aisiti", "smc": "esemsi", "sop": "esopi",
    "bos": "bios", "choch": "cok", "gbpjpy": "jibipijeipiwai", "gj": "jije", "wti": "dablyutiai",
    "xauusd": "eksyuyuesdi", "ath": "eitieic", "eth": "ief", "usd": "yuesdi",
}


def load_audio(path, sr=16000):
    with tempfile.NamedTemporaryFile(suffix=".f32") as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                        "-f", "f32le", tmp.name], check=True)
        return np.fromfile(tmp.name, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Force-align plan segment text to audio")
    ap.add_argument("plan")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--source", help="audio source (default: the plan's source)")
    args = ap.parse_args()

    import torch
    from torchaudio.pipelines import MMS_FA

    plan_path = os.path.abspath(args.plan)
    plan = load_json(plan_path)
    src = args.source or resolve(plan["source"], os.path.dirname(plan_path))
    sr = 16000
    audio = load_audio(src, sr)
    model = MMS_FA.get_model()
    model.eval()
    tokenizer, aligner = MMS_FA.get_tokenizer(), MMS_FA.get_aligner()

    lines, weak = [], []
    for seg in plan["segments"]:
        text = seg.get("text")
        if not text:
            continue
        words = text.split()
        spoken = []
        for w in words:
            k = re.sub(r"[^a-z']", "", w.lower())
            spoken.append(SPOKEN.get(k, k) or "a")
        t0, t1 = seg["in"], seg["out"]
        clip = torch.from_numpy(audio[int(t0 * sr):int(t1 * sr)].copy()).unsqueeze(0)
        with torch.inference_mode():
            emission, _ = model(clip)
        spans = aligner(emission[0], tokenizer(spoken))
        ratio = clip.shape[1] / emission.shape[1] / sr
        out = []
        for w, sp in zip(words, spans):
            score = float(np.mean([t.score for t in sp]))
            out.append({"w": w, "s": round(t0 + sp[0].start * ratio, 3), "e": round(t0 + sp[-1].end * ratio, 3),
                        "p": round(score, 3)})
            if score < 0.15:
                weak.append((seg.get("id"), w, out[-1]["s"]))
        lines.append({"start": out[0]["s"], "end": out[-1]["e"], "text": text, "words": out})
        print(f"[ramz] {seg.get('id', '?'):8s} " + " ".join(f"{x['w']}@{x['s']:.2f}" for x in out))

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(lines, fh, ensure_ascii=False, indent=1)
    print(f"[ramz] SIAP: {args.output} ({sum(len(l['words']) for l in lines)} perkataan)")
    if weak:
        print("[ramz] Skor rendah (semak dengan telinga): " + ", ".join(f"{s}:{w}@{t:.2f}" for s, w, t in weak))


if __name__ == "__main__":
    main()
