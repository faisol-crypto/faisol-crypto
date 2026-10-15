"""Render a RamzlayerG edit plan (JSON) into a finished video with ffmpeg.

Pipeline:
  1. Each segment is cut from the (privacy-protected) source, fitted to the
     canvas, zoomed, graded and annotated -> one intermediate clip each.
  2. Clips are joined in plan order (so the hook can come from anywhere).
  3. Voice, ducked music and timed SFX are mixed, then loudness-normalised
     (two-pass EBU R128) and subtitles are burned in if provided.
  4. An EDL + timeline report is written next to the output.

Usage:
  python3 -m ramz.render plans/example_plan.json
"""
import argparse
import json
import os
import re
import shutil
import tempfile

from . import motion
from .common import (ROOT, fmt_ts, ffmpeg_stderr, hex_to_ff, load_json, probe,
                     require_tools, resolve, run)

# Grades stay mild on purpose: candle colours (bull/bear) must not shift.
GRADES = {
    "none": [],
    "neutral": ["eq=contrast=1.03:saturation=1.02"],
    "natural": ["eq=contrast=1.02:gamma=1.01"],
    "dark_cinematic": [
        "curves=master='0/0.02 0.25/0.21 0.5/0.5 0.75/0.79 1/0.98'",
        "colorbalance=rs=-0.02:bs=0.03:rh=0.015:bh=-0.01",
        "eq=contrast=1.05:brightness=-0.015:saturation=0.96",
    ],
    "cool_modern": [
        "colorbalance=rs=-0.015:bs=0.03:rm=-0.01:bm=0.015",
        "eq=contrast=1.04:saturation=0.97",
    ],
    "clean_contrast": ["eq=contrast=1.10:saturation=1.04:gamma=0.98"],
}

POSITIONS = {  # normalised (x, y) anchor for text; x=None -> centred
    "top": (None, 0.13), "upper": (None, 0.26), "center": (None, 0.47),
    "lower": (None, 0.66), "bottom": (None, 0.76),
}


def esc(expr):
    """Escape commas for use inside a filtergraph option value."""
    return expr.replace(",", r"\,")


def ease(t_expr, start, dur, kind="smooth"):
    """Expression for progress 0..1 between start and start+dur."""
    dur = max(dur, 1e-3)
    p = f"clip(({t_expr}-{start:.4f})/{dur:.4f},0,1)"
    if kind == "linear":
        return p
    return f"({p}*{p}*(3-2*{p}))"  # smoothstep


def alpha_expr(start, end, fade=0.22):
    fade = min(fade, (end - start) / 2)
    return (f"if(lt(t,{start:.3f}),0,if(lt(t,{start + fade:.3f}),(t-{start:.3f})/{fade:.3f},"
            f"if(lt(t,{end - fade:.3f}),1,if(lt(t,{end:.3f}),({end:.3f}-t)/{fade:.3f},0))))")


class Renderer:
    def __init__(self, plan_path, keep_temp=False, verbose=False):
        self.plan_path = os.path.abspath(plan_path)
        self.base = os.path.dirname(self.plan_path)
        self.plan = load_json(self.plan_path)
        self.brand = load_json(resolve(self.plan.get("brand", os.path.join(ROOT, "brand/sirramz.json")), self.base))
        out = self.plan["output"]
        self.W, self.H = int(out.get("width", 1080)), int(out.get("height", 1920))
        self.fps = float(out.get("fps", 30))
        self.out_path = resolve(out.get("path", "out/final.mp4"), self.base)
        self.crf = int(out.get("crf", 18))
        self.preset = out.get("preset", "medium")
        self.source = resolve(self.plan["source"], self.base)
        self.verbose = verbose
        self.keep_temp = keep_temp
        self.tmp = tempfile.mkdtemp(prefix="ramz_")
        self.warnings = []

    # ------------------------------------------------------------------ checks
    def preflight(self):
        if not os.path.exists(self.source):
            raise SystemExit(f"[ramz] Source tidak wujud: {self.source}")
        self.src_info = probe(self.source)
        if self.plan.get("privacy_required", False):
            side = os.path.splitext(self.source)[0] + ".privacy.json"
            if not os.path.exists(side):
                raise SystemExit(
                    "[ramz] privacy_required=true tetapi source belum melalui ramz.privacy.\n"
                    "       Jalankan: python3 -m ramz.privacy <raw> -o <protected.mp4> dahulu.")
            report = load_json(side)
            if report.get("review_needed"):
                self.warnings.append(
                    f"Privacy: {len(report['review_needed'])} julat frame perlu semakan manual "
                    f"(lihat {side}).")
        segs = self.plan.get("segments", [])
        if not segs:
            raise SystemExit("[ramz] Plan tiada segments.")
        dur = self.src_info["duration"]
        for s in segs:
            if not (0 <= s["in"] < s["out"] <= dur + 0.05):
                raise SystemExit(f"[ramz] Segment '{s.get('id')}' in/out di luar julat (source {dur:.2f}s).")
        if segs[0].get("role") != "hook":
            self.warnings.append("Segment pertama bukan role='hook' — video mungkin lambat sampai ke isi.")
        hook_len = segs[0]["out"] - segs[0]["in"]
        if segs[0].get("role") == "hook" and hook_len > 4.0:
            self.warnings.append(f"Hook {hook_len:.1f}s — pertimbangkan potong ke bawah ~3s.")

    # ------------------------------------------------------------- fonts/text
    def font(self, key):
        fonts = self.brand["fonts"]
        path = fonts.get(key, key)
        if not os.path.exists(path):
            path = fonts["fallback"]
        return path

    def color(self, key):
        return self.brand["colors"].get(key, key)

    def text_filter(self, item, idx, seg_dur):
        style = dict(self.brand["text_styles"].get(item.get("style", "caption"), {}))
        style.update({k: v for k, v in item.items() if k in ("size", "color", "box", "box_alpha", "anim", "font")})
        start = float(item.get("start", 0.0))
        end = float(item.get("end", seg_dur))
        size = int(round(float(style.get("size", 0.045)) * self.W))
        txt_path = os.path.join(self.tmp, f"txt_{idx}.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(item["text"])

        pos = item.get("pos", "lower")
        if isinstance(pos, str):
            px, py = POSITIONS[pos]
        else:
            px, py = pos
        x = "(w-text_w)/2" if px is None else f"{px * self.W:.1f}"
        y_base = f"{py * self.H:.1f}"
        anim = style.get("anim", "fade")
        prog = ease("t", start, 0.35)
        if anim == "slide_up":
            y = f"{y_base}+{0.03 * self.H:.1f}*(1-{prog})"
        elif anim == "pop":
            # quick overshoot settle: drop in 2% of height with a small bounce
            y = f"{y_base}-{0.012 * self.H:.1f}*sin(PI*{ease('t', start, 0.25, 'linear')})"
        else:
            y = y_base
        if anim == "slide_left":
            x = f"{x}+{0.08 * self.W:.1f}*(1-{prog})"

        opts = [
            f"fontfile='{self.font(style.get('font', 'body'))}'",
            f"textfile='{txt_path}'", "expansion=none", "text_align=C",
            f"fontsize={size}", f"fontcolor={hex_to_ff(self.color(style.get('color', 'text')))}",
            f"x={esc(x)}", f"y={esc(y)}",
            f"alpha='{esc(alpha_expr(start, end))}'",
            f"line_spacing={int(size * 0.18)}",
            f"shadowcolor={hex_to_ff(self.color('text_shadow'), 0.6)}", "shadowx=0", f"shadowy={max(2, size // 18)}",
            f"enable='between(t,{start:.3f},{end:.3f})'",
        ]
        if style.get("box"):
            opts += ["box=1", f"boxcolor={hex_to_ff(self.color(style['box']), style.get('box_alpha', 0.5))}",
                     f"boxborderw={int(size * 0.38)}"]
        return "drawtext=" + ":".join(opts)

    # ------------------------------------------------------------ annotations
    def annotation_filters(self, a, idx, seg_dur):
        """Chart annotations in normalised canvas coordinates (after zoom).

        Positions must come from the real chart — never invent price action.
        """
        kind = a["type"]
        start = float(a.get("start", 0.0))
        end = float(a.get("end", seg_dur))
        draw = float(a.get("draw", 0.5))
        col = self.color(a.get("color", "primary"))
        en = f"enable='between(t,{start:.3f},{end:.3f})'"
        out = []
        if kind == "level":  # horizontal S/R line drawn left -> right
            y = a["y"] * self.H
            x0 = a.get("x0", 0.0) * self.W
            x1 = a.get("x1", 1.0) * self.W
            w = f"{x1 - x0:.1f}*{ease('t', start, draw)}"
            th = int(a.get("thickness", max(3, self.H // 400)))
            out.append(f"drawbox=x={x0:.1f}:y={y - th / 2:.1f}:w='{esc(w)}':h={th}:color={hex_to_ff(col, 0.95)}:t=fill:{en}")
            if a.get("label"):
                out.append(self.text_filter({"text": a["label"], "style": "label", "start": start + draw * 0.6,
                                             "end": end, "pos": [a.get("label_x", 0.06), (a["y"] - 0.045)],
                                             "color": a.get("label_color", "text")}, f"{idx}l", seg_dur))
        elif kind == "zone":  # supply/demand or S/R area revealed left -> right
            x, y, w, h = a["x"] * self.W, a["y"] * self.H, a["w"] * self.W, a["h"] * self.H
            grow = f"{w:.1f}*{ease('t', start, draw)}"
            out.append(f"drawbox=x={x:.1f}:y={y:.1f}:w='{esc(grow)}':h={h:.1f}:"
                       f"color={hex_to_ff(col, a.get('alpha', 0.22))}:t=fill:{en}")
            th = max(2, self.H // 640)
            out.append(f"drawbox=x={x:.1f}:y={y:.1f}:w='{esc(grow)}':h={h:.1f}:"
                       f"color={hex_to_ff(col, 0.9)}:t={th}:{en}")
            if a.get("label"):
                out.append(self.text_filter({"text": a["label"], "style": "label", "start": start + draw * 0.6,
                                             "end": end, "pos": [a["x"] + 0.015, a["y"] + 0.01]}, f"{idx}z", seg_dur))
        elif kind == "label":  # HH / HL / LH / LL, BOS, CHoCH ...
            out.append(self.text_filter({"text": a["text"], "style": a.get("style", "label"), "start": start,
                                         "end": end, "pos": [a["x"], a["y"]], "color": a.get("color", "text")},
                                        f"{idx}t", seg_dur))
        elif kind == "arrow":
            glyph = {"up": "▲", "down": "▼", "right": "➜", "left": "←"}[a.get("dir", "up")]
            out.append(self.text_filter({"text": glyph, "style": "emphasis", "start": start, "end": end,
                                         "pos": [a["x"], a["y"]], "color": a.get("color", "primary"),
                                         "font": "fallback"}, f"{idx}a", seg_dur))
        else:
            raise SystemExit(f"[ramz] Jenis annotation tidak dikenali: {kind}")
        return out

    # ---------------------------------------------------------------- segment
    def zoom_filter(self, z, seg_dur):
        z0, z1 = float(z.get("from", 1.0)), float(z.get("to", 1.12))
        start = float(z.get("start", 0.0))
        dur = float(z.get("dur", seg_dur - start))
        cx, cy = float(z.get("cx", 0.5)), float(z.get("cy", 0.5))
        T = f"(in/{self.fps})"
        zoom = f"({z0}+({z1 - z0})*{ease(T, start, dur, z.get('ease', 'smooth'))})"
        L = f"({cx}*W*(1-1/{zoom}))"
        Tp = f"({cy}*H*(1-1/{zoom}))"
        R = f"({L}+W/{zoom})"
        B = f"({Tp}+H/{zoom})"
        pts = [L, Tp, R, Tp, L, B, R, B]
        names = ["x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"]
        # perspective gives sub-pixel sampling -> no stair-step jitter on slow push-ins
        return "perspective=" + ":".join(f"{n}='{esc(p)}'" for n, p in zip(names, pts)) + ":interpolation=linear:eval=frame"

    def fit_filters(self, fit, cx=0.5, cy=0.5):
        W, H = self.W, self.H
        if fit == "crop":
            return (f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,"
                    f"crop={W}:{H}:'(iw-{W})*{cx}':'(ih-{H})*{cy}'")
        if fit == "blur_pad":
            return (f"split[fa][fb];[fa]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                    f"boxblur=28:2,eq=brightness=-0.14:saturation=0.8[bg];"
                    f"[fb]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                    f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
        if fit == "pad":
            return (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                    f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color={hex_to_ff(self.color('panel'))}")
        raise SystemExit(f"[ramz] fit tidak dikenali: {fit}")

    def render_segment(self, idx, seg):
        # whole frames only: a segment whose video is a frame shorter than its
        # audio leaves a timestamp gap at every join, and the concat then
        # plays slow (e.g. 29.83 fps) and drifts out of sync
        nframes = max(1, int(round((seg["out"] - seg["in"]) * self.fps)))
        dur = nframes / self.fps
        grade = seg.get("grade", self.plan.get("grade", "neutral"))
        if grade not in GRADES:
            raise SystemExit(f"[ramz] grade tidak dikenali: {grade}")
        fit = seg.get("fit", self.plan.get("fit", "crop"))

        v = [f"setpts=PTS-STARTPTS,fps={self.fps},setsar=1,format=yuv420p"]
        v.append(self.fit_filters(fit, seg.get("crop_cx", 0.5), seg.get("crop_cy", 0.5)))
        if seg.get("zoom"):
            v.append(self.zoom_filter(seg["zoom"], dur))
        # technical fix (e.g. white balance) before the creative grade; a segment
        # may add its own (e.g. a different light source in one shot)
        v += self.plan.get("correction", []) + seg.get("correction", [])
        v += GRADES[grade]
        vig = seg.get("vignette", self.plan.get("vignette", 0.0))
        if vig:
            v.append(f"vignette=angle={0.785 * float(vig):.3f}")
        for fx in seg.get("fx", []):
            v.append(self.fx_filter(fx, dur))
        for i, a in enumerate(seg.get("annotations", [])):
            v += self.annotation_filters(a, f"{idx}_{i}", dur)
        for i, t in enumerate(seg.get("text", [])):
            v.append(self.text_filter(t, f"{idx}_{i}", dur))
        tin, tout = seg.get("transition_in", "cut"), seg.get("transition_out", "cut")
        if tin == "dip":
            v.append("fade=t=in:st=0:d=0.18")
        if tout == "dip":
            v.append(f"fade=t=out:st={max(dur - 0.18, 0):.3f}:d=0.18")
        v.append(f"tpad=stop_mode=clone:stop_duration=0.2,trim=end_frame={nframes},format=yuv420p")

        a = ["asetpts=PTS-STARTPTS", "aresample=48000", "aformat=channel_layouts=stereo"]
        ac = self.plan.get("audio", {})
        if ac.get("highpass", 80):
            a.append(f"highpass=f={ac.get('highpass', 80)}")
        if ac.get("denoise", False):
            a.append(f"afftdn=nf={ac.get('denoise_nf', -45)}:tn=1")
        if ac.get("voice_eq", True):
            a.append("equalizer=f=250:t=q:w=1.2:g=-1.5,equalizer=f=3200:t=q:w=1:g=1.5")
        if seg.get("gain_db"):
            a.append(f"volume={seg['gain_db']}dB")
        # micro fades kill clicks on every cut
        a.append(f"afade=t=in:st=0:d=0.012,afade=t=out:st={max(dur - 0.02, 0):.3f}:d=0.02")
        a.append(f"apad,atrim=0:{dur:.6f}")

        out = os.path.join(self.tmp, f"seg_{idx:03d}.mov")
        has_audio = self.src_info["audio"] is not None
        cmd = ["ffmpeg", "-y", "-hide_banner", "-ss", f"{seg['in']:.3f}", "-t", f"{dur + 0.2:.3f}", "-i", self.source]
        if not has_audio:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        graph = f"[0:v]{','.join(v)}[vout];[{0 if has_audio else 1}:a]{','.join(a)}[aout]"
        cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
                "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-r", f"{self.fps}",
                "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", "48000", "-t", f"{dur:.6f}",
                out]
        run(cmd, quiet=not self.verbose)
        return out

    def fx_filter(self, fx, seg_dur):
        kind = fx["type"]
        at = float(fx.get("at", 0.0))
        d = float(fx.get("dur", 0.15))
        if kind == "flash":  # soft exposure pop on impact
            amt = float(fx.get("amount", 0.18))
            return f"eq=brightness='{esc(f'{amt}*between(t,{at},{at + d})*(1-(t-{at})/{d})')}':eval=frame"
        if kind == "glow":  # restrained bloom on highlights
            amt = float(fx.get("amount", 0.25))
            return (f"split[g0][g1];[g1]curves=master='0/0 0.6/0 1/1',gblur=sigma={0.012 * self.W:.1f}[gb];"
                    f"[g0][gb]blend=all_mode=screen:all_opacity={amt}")
        if kind == "rgb_split":  # very short chromatic hit, only inside [at, at+d]
            px = int(fx.get("px", 6))
            return (f"split[r0][r1];[r1]rgbashift=rh={px}:bh=-{px}[rs];"
                    f"[r0][rs]overlay=enable='between(t,{at},{at + d})'")
        if kind == "blur_in":  # focus pull from soft to sharp
            return f"gblur=sigma=12:enable='between(t,{at},{at + d})'"
        raise SystemExit(f"[ramz] fx tidak dikenali: {kind}")

    # ------------------------------------------------------------------ audio
    def mix_audio(self, master, offsets, total):
        """Voice (+ducked music) (+SFX) -> mixed WAV."""
        inputs = ["-i", master]
        chains = ["[0:a]aformat=channel_layouts=stereo,asplit=2[voice][key]"]
        mix = ["[voice]"]
        n = 1
        ac = self.plan.get("audio", {})
        music = self.plan.get("music")
        if music and music.get("file"):
            inputs = inputs + ["-stream_loop", "-1", "-i", resolve(music["file"], self.base)]
            mdb = music.get("gain_db", self.brand["audio"]["music_db"])
            fin, fout = music.get("fade_in", 0.8), music.get("fade_out", 1.8)
            chain = (f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{total:.3f},"
                     f"asetpts=PTS-STARTPTS,volume={mdb}dB,afade=t=in:d={fin},"
                     f"afade=t=out:st={max(total - fout, 0):.3f}:d={fout}[mus]")
            chains.append(chain)
            if music.get("duck", True):
                chains.append("[mus][key]sidechaincompress=threshold=0.02:ratio=6:attack=25:release=350:makeup=1[musd]")
                mix.append("[musd]")
            else:
                chains.append("[key]anullsink")
                mix.append("[mus]")
            n += 1
        else:
            chains.append("[key]anullsink")
        for si, (seg_idx, cue) in enumerate(self.sfx_cues()):
            path = resolve(cue["file"], self.base)
            if not os.path.exists(path):
                raise SystemExit(f"[ramz] SFX tidak wujud: {path}")
            inputs += ["-i", path]
            at = offsets[seg_idx] + float(cue.get("at", 0.0))
            ms = max(int(at * 1000), 0)
            gdb = cue.get("gain_db", self.brand["audio"]["sfx_db"])
            chains.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,volume={gdb}dB,"
                          f"adelay={ms}|{ms},apad[s{si}]")
            mix.append(f"[s{si}]")
            n += 1
        if len(mix) > 1:
            chains.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first,"
                          f"atrim=0:{total:.3f}[mix]")
        else:
            chains.append(f"[voice]atrim=0:{total:.3f}[mix]")
        out = os.path.join(self.tmp, "mix.wav")
        run(["ffmpeg", "-y", "-hide_banner", *inputs, "-filter_complex", ";".join(chains),
             "-map", "[mix]", "-c:a", "pcm_s24le", "-ar", "48000", out], quiet=not self.verbose)
        return out

    def sfx_cues(self):
        for idx, seg in enumerate(self.plan["segments"]):
            for cue in seg.get("sfx", []):
                yield idx, cue

    def limiter(self):
        """Brick-wall safety after loudnorm: loudnorm can't always hold the
        true-peak ceiling on sharp transients (crowds, claps, SFX)."""
        tp = self.brand["audio"]["true_peak_db"] - 0.3
        return f"alimiter=limit={10 ** (tp / 20):.4f}:attack=2:release=60:level=false,aresample=48000"

    def loudnorm(self, wav):
        target = self.plan.get("audio", {}).get("target_lufs", self.brand["audio"]["target_lufs"])
        tp = self.brand["audio"]["true_peak_db"]
        base = f"loudnorm=I={target}:TP={tp}:LRA=9"
        err = ffmpeg_stderr(["-i", wav, "-af", base + ":print_format=json", "-f", "null", "-"])
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", err, re.S)
        if not m:
            return base
        d = json.loads(m.group(0))
        if float(d["input_i"]) < -70:  # silence: nothing to normalise
            return "anull"
        return (f"{base}:measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:"
                f"measured_thresh={d['input_thresh']}:offset={d['target_offset']}:linear=true")

    # ------------------------------------------------------------------- main
    def render(self):
        require_tools()
        self.preflight()
        segs = self.plan["segments"]
        clips = []
        for i, seg in enumerate(segs):
            print(f"[ramz] segment {i + 1}/{len(segs)}: {seg.get('id', i)} ({seg['out'] - seg['in']:.2f}s)")
            clips.append(self.render_segment(i, seg))
        durs = [probe(c)["duration"] for c in clips]
        offsets = [sum(durs[:i]) for i in range(len(durs))]
        total = sum(durs)

        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        lst = os.path.join(self.tmp, "concat.txt")
        with open(lst, "w") as fh:
            fh.writelines(f"file '{c}'\n" for c in clips)
        master = os.path.join(self.tmp, "master.mov")
        run(["ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", master],
            quiet=not self.verbose)

        print("[ramz] audio mix + loudness")
        mix = self.mix_audio(master, offsets, total)
        norm = self.loudnorm(mix)

        stem = os.path.splitext(self.out_path)[0]
        gfx_inputs, gfx_chain, last = [], [], "0:v"
        for gi, g in enumerate(self.plan.get("graphics", [])):
            seg_idx = next(i for i, sg in enumerate(segs) if sg.get("id") == g["seg"])
            start = offsets[seg_idx] + float(g.get("at", 0.0))
            dur = float(g["dur"])
            print(f"[ramz] motion graphic {gi + 1}: {g['type']} @ {start:.2f}s ({dur:.2f}s)")
            gdir = os.path.join(self.tmp, f"gfx_{gi:02d}")
            x, y = motion.render(g["type"], g, dur, self.fps, gdir, self.plan.get("motion_brand", {}))
            gfx_inputs += ["-framerate", f"{self.fps}", "-i", os.path.join(gdir, "f%04d.png")]
            n = len(gfx_inputs) // 4 + 1  # input index (0=master, 1=mix)
            gfx_chain.append(f"[{n}:v]format=rgba,setpts=PTS-STARTPTS+{start:.4f}/TB[g{gi}]")
            gfx_chain.append(f"[{last}][g{gi}]overlay={x}:{y}:eof_action=pass:format=auto[v{gi}]")
            last = f"v{gi}"

        ass_filter = None
        cap = self.plan.get("captions")
        if cap:
            from .captions import build as build_captions
            ass, review = build_captions(cap, resolve(cap["transcript"], self.base), segs, offsets, durs,
                                         self.W, self.H, cap.get("font", "Inter"), stem)
            if review:
                self.warnings.append(f"Caption: {len(review)} perkataan perlu disemak ({stem}.captions_review.md).")
            p = ass.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
            ass_filter = f"ass='{p}':fontsdir='{os.path.dirname(self.font('headline'))}'"
        elif self.plan.get("subtitles"):
            subs = self.plan["subtitles"]
            ass_filter = self.subtitle_filter(resolve(subs if isinstance(subs, str) else subs["file"], self.base))

        chain = list(gfx_chain)
        outputs = []
        enc = ["-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-t", f"{total:.3f}"]
        base = f"[{last}]" if last != "0:v" else "[0:v]"
        if ass_filter:
            clean_path = stem + "_clean.mp4"
            chain.append(f"{base}split=2[vclean][vsub0]")
            chain.append(f"[vsub0]{ass_filter}[vsub]")
            chain.append("[1:a]" + norm + "," + self.limiter() + ",asplit=2[a0][a1]")
            outputs += ["-map", "[vsub]", "-map", "[a0]", *enc, self.out_path,
                        "-map", "[vclean]", "-map", "[a1]", *enc, clean_path]
        else:
            chain.append(f"{base}null[vout]")
            chain.append("[1:a]" + norm + "," + self.limiter() + "[a0]")
            outputs += ["-map", "[vout]", "-map", "[a0]", *enc, self.out_path]
        cmd = ["ffmpeg", "-y", "-hide_banner", "-i", master, "-i", mix, *gfx_inputs,
               "-filter_complex", ";".join(chain), *outputs]
        print("[ramz] final export" + (" (dengan caption + versi bersih)" if ass_filter else ""))
        run(cmd, quiet=not self.verbose)
        self.write_reports(offsets, durs)
        if not self.keep_temp:
            shutil.rmtree(self.tmp, ignore_errors=True)
        print(f"[ramz] SIAP: {self.out_path} ({total:.2f}s)")
        if ass_filter:
            print(f"[ramz] SIAP: {stem}_clean.mp4 (tanpa caption)")
        for w in self.warnings:
            print(f"[ramz] AMARAN: {w}")
        return self.out_path

    def write_cmx_edl(self, path, offsets, durs):
        """CMX3600 EDL of the cuts — imports into DaVinci Resolve / Premiere / FCP."""
        fps = int(round(self.fps))

        def tc(sec):
            f = int(round(sec * fps))
            return f"{f // (3600 * fps):02d}:{f // (60 * fps) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"
        reel = os.path.splitext(os.path.basename(self.source))[0][:8].upper().replace(" ", "_") or "AX"
        lines = [f"TITLE: {os.path.splitext(os.path.basename(self.out_path))[0]}", "FCM: NON-DROP FRAME", ""]
        for i, (seg, off, d) in enumerate(zip(self.plan["segments"], offsets, durs), 1):
            lines.append(f"{i:03d}  {reel:<8} AA/V  C        {tc(seg['in'])} {tc(seg['in'] + d)} "
                         f"{tc(off)} {tc(off + d)}")
            lines.append(f"* FROM CLIP NAME: {os.path.basename(self.source)}")
            lines.append(f"* COMMENT: {seg.get('id', '')} ({seg.get('role', '')})")
            lines.append("")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def subtitle_filter(self, path):
        style = self.plan.get("subtitle_style", {})
        font = os.path.basename(self.font("body")).split("-")[0]
        size = int(style.get("size", 13))
        force = (f"FontName={style.get('font', font)},FontSize={size},Bold=1,"
                 f"PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=1,Outline=2,Shadow=0,"
                 f"Alignment=2,MarginV={int(style.get('margin_v', 70))}")
        fontsdir = os.path.dirname(self.font("body"))
        p = path.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        return f"subtitles='{p}':fontsdir='{fontsdir}':force_style='{force}'"

    def write_reports(self, offsets, durs):
        stem = os.path.splitext(self.out_path)[0]
        edl = []
        lines = [f"# Timeline — {os.path.basename(self.out_path)}", "",
                 f"Format: {self.W}x{self.H} @ {self.fps:g}fps · Grade: {self.plan.get('grade', 'neutral')}", ""]
        for seg, off, d in zip(self.plan["segments"], offsets, durs):
            edl.append({"id": seg.get("id"), "role": seg.get("role"), "final_start": round(off, 3),
                        "final_end": round(off + d, 3), "source_in": seg["in"], "source_out": seg["out"]})
            lines.append(f"## {fmt_ts(off)}–{fmt_ts(off + d)} — {str(seg.get('role', 'segment')).upper()} "
                         f"(`{seg.get('id', '')}`)")
            lines.append(f"- **Sumber:** {fmt_ts(seg['in'])}–{fmt_ts(seg['out'])} dalam rakaman asal")
            texts = "; ".join(t["text"].replace("\n", " ") for t in seg.get("text", [])) or "—"
            lines.append(f"- **Teks:** {texts}")
            ann = ", ".join(f"{a['type']}" + (f" '{a.get('label') or a.get('text')}'" if a.get('label') or a.get('text') else "")
                            for a in seg.get("annotations", [])) or "—"
            lines.append(f"- **Animasi chart:** {ann}")
            z = seg.get("zoom")
            lines.append(f"- **Motion:** " + (f"zoom {z.get('from', 1.0)}→{z.get('to', 1.12)} fokus "
                                              f"({z.get('cx', 0.5)}, {z.get('cy', 0.5)})" if z else "statik"))
            lines.append(f"- **FX:** " + (", ".join(f["type"] for f in seg.get("fx", [])) or "—"))
            lines.append(f"- **Audio/SFX:** " + (", ".join(f"{os.path.basename(c['file'])} @+{c.get('at', 0)}s"
                                                           for c in seg.get("sfx", [])) or "suara sahaja"))
            lines.append(f"- **Warna:** {seg.get('grade', self.plan.get('grade', 'neutral'))}")
            lines.append(f"- **Transisi:** masuk {seg.get('transition_in', 'cut')}, keluar {seg.get('transition_out', 'cut')}")
            lines.append(f"- **Tujuan kreatif:** {seg.get('purpose', '—')}")
            lines.append("")
        self.write_cmx_edl(stem + ".edl", offsets, durs)
        with open(stem + ".edl.json", "w", encoding="utf-8") as fh:
            json.dump(edl, fh, indent=2, ensure_ascii=False)
        with open(stem + ".timeline.md", "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Render RamzlayerG edit plan")
    ap.add_argument("plan")
    ap.add_argument("--keep-temp", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    Renderer(args.plan, keep_temp=args.keep_temp, verbose=args.verbose).render()


if __name__ == "__main__":
    main()
