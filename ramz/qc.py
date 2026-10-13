"""Measured quality control for a finished export.

Only checks what can actually be measured; everything subjective (hook
strength, trading accuracy, authenticity) is listed as a manual checklist —
no automatic 10/10.

Usage:
  python3 -m ramz.qc out/final.mp4 [--plan plans/x.json] [--target-lufs -14]
"""
import argparse
import json
import os
import re

from .analyze import loudness
from .common import ffmpeg_stderr, load_json, probe, require_tools

MANUAL = {
    "HOOK": ["Hook muncul dalam 0–3s pertama dan relevan dengan isi?", "Visual menyokong ayat hook?"],
    "VISUAL": ["Teks mudah dibaca di telefon (saiz, kontras, safe area)?", "Chart & warna candle kekal jelas?",
               "Efek tidak berlebihan?"],
    "PRIVACY": ["Tonton setiap julat review_needed dalam laporan privacy — tiada frame mendedahkan wajah?"],
    "AUTHENTICITY": ["Suara & rentak masih natural (bukan terlalu laju/sempurna)?"],
    "TRADING ACCURACY": ["Label HH/HL/LH/LL, zon & level tepat pada chart sebenar?",
                         "Tiada dakwaan keuntungan tanpa asas?"],
}


def main():
    ap = argparse.ArgumentParser(description="RamzlayerG QC")
    ap.add_argument("video")
    ap.add_argument("--plan")
    ap.add_argument("--target-lufs", type=float, default=-14.0)
    args = ap.parse_args()
    require_tools()

    info = probe(args.video)
    v, a = info["video"], info["audio"]
    results = []

    def check(area, name, status, detail):
        results.append({"area": area, "check": name, "status": status, "detail": detail})

    if v:
        check("EXPORT", "Resolusi", "PASS" if min(v["width"], v["height"]) >= 1080 else "WARN",
              f"{v['width']}x{v['height']}")
        cfr = abs(v["fps"] - v["r_fps"]) < 0.01
        check("EXPORT", "Frame rate konsisten", "PASS" if cfr else "FAIL", f"avg {v['fps']:.3f} / r {v['r_fps']:.3f}")
        err = ffmpeg_stderr(["-i", args.video, "-an", "-vf",
                             "blackdetect=d=0.25:pix_th=0.06,freezedetect=n=-55dB:d=1.5", "-f", "null", "-"])
        blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)
        freezes = re.findall(r"freeze_start: ([\d.]+)", err)
        check("EXPORT", "Frame hitam", "PASS" if not blacks else "WARN",
              ", ".join(f"{float(s):.2f}-{float(e):.2f}s" for s, e in blacks) or "tiada")
        check("EXPORT", "Frame beku (>1.5s)", "PASS" if not freezes else "WARN",
              ", ".join(f"{float(s):.2f}s" for s in freezes) or "tiada (chart statik boleh trigger ini — semak)")
    if a:
        L = loudness(args.video)
        i, tp = L["integrated_lufs"], L["true_peak_dbfs"]
        ok = i is not None and abs(i - args.target_lufs) <= 1.0
        check("AUDIO", "Loudness", "PASS" if ok else "WARN", f"{i} LUFS (sasaran {args.target_lufs})")
        check("AUDIO", "True peak", "PASS" if tp is not None and tp <= -0.8 else "FAIL", f"{tp} dBFS")
        err = ffmpeg_stderr(["-i", args.video, "-vn", "-af", "astats=metadata=0:reset=0", "-f", "null", "-"])
        clips = [int(x) for x in re.findall(r"Number of samples clipped: (\d+)", err)] if "clipped" in err else []
        check("AUDIO", "Clipping", "PASS" if not any(clips) else "FAIL",
              f"{sum(clips)} sampel" if clips else "tiada")
    else:
        check("AUDIO", "Trek audio", "FAIL", "tiada audio")

    if args.plan:
        plan = load_json(args.plan)
        segs = plan.get("segments", [])
        if segs:
            first = segs[0]
            hl = first["out"] - first["in"]
            check("HOOK", "Segment pertama = hook", "PASS" if first.get("role") == "hook" else "FAIL",
                  first.get("id", ""))
            check("HOOK", "Panjang hook", "PASS" if hl <= 3.5 else "WARN", f"{hl:.2f}s")
            n_text = sum(len(s.get("text", [])) for s in segs)
            n_fx = sum(len(s.get("fx", [])) for s in segs)
            n_sfx = sum(len(s.get("sfx", [])) for s in segs)
            dens = (n_fx + n_sfx) / max(info["duration"], 1)
            check("VISUAL", "Kepadatan efek", "PASS" if dens <= 0.35 else "WARN",
                  f"{n_fx} fx + {n_sfx} sfx dalam {info['duration']:.1f}s ({dens:.2f}/s)")
            check("VISUAL", "Overlay teks", "PASS", f"{n_text} item")
            grades = {s.get("grade", plan.get("grade", "neutral")) for s in segs}
            check("VISUAL", "Grade konsisten", "PASS" if len(grades) == 1 else "WARN", ", ".join(sorted(grades)))
        side = os.path.splitext(os.path.abspath(plan_source(plan, args.plan)))[0] + ".privacy.json"
        if plan.get("privacy_required"):
            if os.path.exists(side):
                rep = load_json(side)
                check("PRIVACY", "Privacy pass", "PASS" if not rep["review_needed"] else "WARN",
                      f"{len(rep['review_needed'])} julat perlu semakan manual")
            else:
                check("PRIVACY", "Privacy pass", "FAIL", "tiada laporan privacy")

    print(f"\nQC — {os.path.basename(args.video)} ({info['duration']:.2f}s)\n")
    for r in results:
        icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[r["status"]]
        print(f"{icon} [{r['area']}] {r['check']}: {r['detail']}")
    print("\nSemakan manual (tidak boleh diukur automatik):")
    for area, items in MANUAL.items():
        for it in items:
            print(f"  [ ] [{area}] {it}")
    fails = sum(r["status"] == "FAIL" for r in results)
    warns = sum(r["status"] == "WARN" for r in results)
    print(f"\nRingkasan: {len(results) - fails - warns} PASS, {warns} WARN, {fails} FAIL "
          f"— bukan skor akhir; semakan manual masih diperlukan.")
    out = os.path.splitext(args.video)[0] + ".qc.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"measured": results, "manual": MANUAL}, fh, indent=2, ensure_ascii=False)
    return 1 if fails else 0


def plan_source(plan, plan_path):
    src = plan["source"]
    return src if os.path.isabs(src) else os.path.join(os.path.dirname(os.path.abspath(plan_path)), src)


if __name__ == "__main__":
    raise SystemExit(main())
