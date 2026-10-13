"""Shared helpers: running ffmpeg/ffprobe, probing media, loading config."""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require_tools():
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"[ramz] '{tool}' tidak dijumpai. Install ffmpeg dahulu.")


def run(cmd, quiet=True, capture=False):
    """Run a command; abort with the tail of stderr on failure."""
    if not quiet:
        print("[ramz] $", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                          stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
        sys.exit(f"[ramz] Command gagal ({proc.returncode}):\n{' '.join(cmd)}\n{tail}")
    return proc


def ffmpeg_stderr(args):
    """Run ffmpeg for analysis filters and return stderr text (output discarded)."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args],
                          stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return proc.stderr


def probe(path):
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
                "-show_streams", path], capture=True)
    data = json.loads(proc.stdout)
    info = {"path": path, "duration": float(data["format"].get("duration", 0.0)),
            "video": None, "audio": None}
    for s in data["streams"]:
        if s["codec_type"] == "video" and info["video"] is None:
            info["video"] = {
                "width": int(s["width"]), "height": int(s["height"]),
                "fps": _ratio(s.get("avg_frame_rate", "0/1")),
                "r_fps": _ratio(s.get("r_frame_rate", "0/1")),
                "codec": s.get("codec_name"), "pix_fmt": s.get("pix_fmt"),
                "nb_frames": int(s["nb_frames"]) if s.get("nb_frames", "").isdigit() else None,
            }
        elif s["codec_type"] == "audio" and info["audio"] is None:
            info["audio"] = {"codec": s.get("codec_name"),
                             "sample_rate": int(s.get("sample_rate", 0)),
                             "channels": int(s.get("channels", 0))}
    return info


def _ratio(text):
    num, _, den = text.partition("/")
    try:
        return float(num) / float(den or 1) if float(den or 1) else 0.0
    except ValueError:
        return 0.0


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve(path, base):
    """Resolve a plan-relative path."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base, path))


def hex_to_ff(color, alpha=None):
    """'#D4AF37' -> '0xD4AF37' (+ '@alpha') for ffmpeg colour options."""
    c = "0x" + color.lstrip("#")
    return f"{c}@{alpha}" if alpha is not None else c


def fmt_ts(seconds):
    m, s = divmod(max(seconds, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"
