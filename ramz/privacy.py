"""Face privacy pass — run on RAW footage before any editing.

Doing privacy first means later zooms, crops and transitions can never expose
an unprotected frame.

Detection: OpenCV Haar (frontal + both profiles) every frame, then a tracker
that smooths the box (no jumping mask), pads it generously and holds it for a
while after detection is lost (turned heads, motion blur). Manual keyframed
regions from a JSON file are always applied on top, for the cases automatic
detection misses.

Methods:
  soft    — heavy feathered gaussian blur (default, looks natural)
  pixel   — mosaic + blur, stronger identity protection
  solid   — brand-coloured feathered disc, maximum protection

Writes <output>.privacy.json with coverage stats and the frame ranges that
need a human check. ramz.render refuses privacy_required plans without it.

Usage:
  python3 -m ramz.privacy raw.mp4 -o protected.mp4 [--method soft] [--regions manual.json]
  manual.json: [{"t": 0.0, "x": 0.4, "y": 0.1, "w": 0.2, "h": 0.25}, {"t": 4.0, ...}]
"""
import argparse
import json
import os
import subprocess
import sys

from .common import probe, require_tools


def _cv():
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        sys.exit("[ramz] Perlukan OpenCV: pip install 'opencv-python-headless<5' numpy")
    import cv2
    import numpy as np
    return cv2, np


class FaceTracker:
    """Smoothed single/multi-face box tracker with hold-after-loss."""

    def __init__(self, fps, hold_s=1.2, smooth=0.35):
        self.tracks = []  # each: dict(box=[cx,cy,w,h], miss=int)
        self.hold = int(hold_s * fps)
        self.smooth = smooth

    @staticmethod
    def _iou_center(a, b):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return dx < max(a[2], b[2]) * 0.8 and dy < max(a[3], b[3]) * 0.8

    def update(self, detections):
        used = set()
        for tr in self.tracks:
            match = None
            for i, d in enumerate(detections):
                if i not in used and self._iou_center(tr["box"], d):
                    match = i
                    break
            if match is None:
                tr["miss"] += 1
                # grow a little while lost so a moving head stays covered
                tr["box"][2] *= 1.01
                tr["box"][3] *= 1.01
            else:
                used.add(match)
                a = self.smooth
                tr["box"] = [tr["box"][k] * (1 - a) + detections[match][k] * a for k in range(4)]
                # never shrink faster than it grows: protection over aesthetics
                tr["box"][2] = max(tr["box"][2], detections[match][2])
                tr["box"][3] = max(tr["box"][3], detections[match][3])
                tr["miss"] = 0
        for i, d in enumerate(detections):
            if i not in used:
                self.tracks.append({"box": list(d), "miss": 0})
        self.tracks = [t for t in self.tracks if t["miss"] <= self.hold]
        return [t["box"] for t in self.tracks], any(t["miss"] > 0 for t in self.tracks)


def interp_regions(regions, t):
    """Linear interpolation of manual keyframed regions at time t."""
    if not regions:
        return None
    if t <= regions[0]["t"]:
        r = regions[0]
    elif t >= regions[-1]["t"]:
        r = regions[-1]
    else:
        for a, b in zip(regions, regions[1:]):
            if a["t"] <= t <= b["t"]:
                k = (t - a["t"]) / max(b["t"] - a["t"], 1e-6)
                r = {key: a[key] + (b[key] - a[key]) * k for key in ("x", "y", "w", "h")}
                r["enabled"] = a.get("enabled", True)
                break
    if not r.get("enabled", True):
        return None
    return r


def apply_masks(cv2, np, frame, masks, method, col):
    """Blend the cover into the elliptical masks, working only inside their ROI."""
    H, W = frame.shape[:2]
    feather = int(max(9, min(W, H) * 0.025)) | 1
    m = feather * 2
    x0 = max(0, int(min(cx - ax for cx, _, ax, _ in masks)) - m)
    y0 = max(0, int(min(cy - ay for _, cy, _, ay in masks)) - m)
    x1 = min(W, int(max(cx + ax for cx, _, ax, _ in masks)) + m)
    y1 = min(H, int(max(cy + ay for _, cy, _, ay in masks)) + m)
    if x1 <= x0 or y1 <= y0:
        return frame
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    alpha = np.zeros((rh, rw), np.float32)
    for (cx, cy, ax, ay) in masks:
        cv2.ellipse(alpha, (int(cx - x0), int(cy - y0)), (max(1, int(ax)), max(1, int(ay))), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (feather * 2 + 1, feather * 2 + 1), 0)
    alpha = np.clip(alpha * 1.6, 0, 1)[..., None]  # solid core, soft edge
    if method == "solid":
        cover = np.empty_like(roi)
        cover[:] = col
    elif method == "pixel":
        blk = max(8, int(min(W, H) * 0.035))
        mosaic = cv2.resize(roi, (max(1, rw // blk), max(1, rh // blk)), interpolation=cv2.INTER_AREA)
        cover = cv2.GaussianBlur(cv2.resize(mosaic, (rw, rh), interpolation=cv2.INTER_NEAREST), (blk | 1, blk | 1), 0)
    else:
        # downscale -> heavy blur -> upscale: strong, smooth and cheap
        k = max(3, int(min(W, H) * 0.09) // 4) | 1
        small = cv2.resize(roi, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
        cover = cv2.resize(cv2.GaussianBlur(small, (k, k), 0), (rw, rh), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = (roi * (1 - alpha) + cover * alpha).astype(np.uint8)
    return frame


def main():
    ap = argparse.ArgumentParser(description="RamzlayerG face privacy pass")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--method", choices=["soft", "pixel", "solid"], default="soft")
    ap.add_argument("--pad", type=float, default=0.45, help="padding around face box (fraction)")
    ap.add_argument("--regions", help="manual keyframed regions JSON (normalised coords)")
    ap.add_argument("--hold", type=float, default=1.2, help="seconds to keep mask after detection lost")
    ap.add_argument("--no-detect", action="store_true", help="manual --regions only (fixed desk setups, fastest)")
    ap.add_argument("--detect-every", type=int, default=3, help="run detector every N frames (tracker fills gaps)")
    ap.add_argument("--detect-width", type=int, default=640, help="detection runs at this width for speed")
    ap.add_argument("--color", default="#0B0F14", help="disc colour for --method solid")
    ap.add_argument("--crf", type=int, default=14, help="near-lossless intermediate")
    args = ap.parse_args()

    require_tools()
    cv2, np = _cv()
    info = probe(args.input)
    v = info["video"]
    W, H, fps = v["width"], v["height"], v["fps"] or 30.0

    cascades = [cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))]
    profile = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml"))
    regions = None
    if args.no_detect and not args.regions:
        sys.exit("[ramz] --no-detect perlukan --regions.")
    if args.regions:
        with open(args.regions, encoding="utf-8") as fh:
            regions = sorted(json.load(fh), key=lambda r: r["t"])

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", args.input, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                           stdout=subprocess.PIPE)
    enc_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
               "-r", f"{fps}", "-i", "-", "-i", args.input, "-map", "0:v", "-map", "1:a?",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
               "-c:a", "copy", args.output]
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE)

    tracker = FaceTracker(fps / max(1, args.detect_every), hold_s=args.hold)
    holding = False
    frame_bytes = W * H * 3
    s = min(1.0, args.detect_width / W)
    min_face = max(24, int(min(W, H) * s * 0.06))
    col = tuple(int(args.color.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))  # BGR
    n = covered = detected = runs = 0
    lost_runs, run_start = [], None

    while True:
        buf = dec.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        t = n / fps
        if not args.no_detect and n % max(1, args.detect_every) == 0:
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            dets = []
            for c in cascades:
                dets += list(c.detectMultiScale(gray, 1.12, 5, minSize=(min_face, min_face)))
            dets += list(profile.detectMultiScale(gray, 1.12, 5, minSize=(min_face, min_face)))
            flipped = cv2.flip(gray, 1)
            for (x, y, w, h) in profile.detectMultiScale(flipped, 1.12, 5, minSize=(min_face, min_face)):
                dets.append((gray.shape[1] - x - w, y, w, h))
            # merge overlapping detections into centre boxes in full-res pixels
            boxes = []
            for (x, y, w, h) in dets:
                b = [(x + w / 2) / s, (y + h / 2) / s, w / s, h / s]
                if not any(FaceTracker._iou_center(b, o) for o in boxes):
                    boxes.append(b)
            runs += 1
            if boxes:
                detected += 1
            tracked, holding = tracker.update(boxes)
        else:
            tracked = [tr["box"] for tr in tracker.tracks]

        masks = []
        for (cx, cy, w, h) in tracked:
            pw, ph = w * (1 + args.pad), h * (1 + args.pad) * 1.15  # taller: hair/chin
            masks.append((cx, cy - h * 0.05, pw / 2, ph / 2))
        r = interp_regions(regions, t)
        if r:
            masks.append(((r["x"] + r["w"] / 2) * W, (r["y"] + r["h"] / 2) * H, r["w"] * W / 2, r["h"] * H / 2))

        if masks:
            covered += 1
            frame = apply_masks(cv2, np, frame, masks, args.method, col)

        # flag stretches where a face was being tracked but detection dropped out
        risky = holding
        if risky and run_start is None:
            run_start = t
        elif not risky and run_start is not None:
            if t - run_start > 0.2:
                lost_runs.append([round(run_start, 2), round(t, 2)])
            run_start = None
        enc.stdin.write(frame.tobytes())
        n += 1
        if n % int(fps * 5) == 0:
            print(f"[ramz] privacy {t:6.1f}s / {info['duration']:.1f}s", flush=True)

    if run_start is not None:
        lost_runs.append([round(run_start, 2), round(n / fps, 2)])
    enc.stdin.close()
    enc.wait()
    dec.wait()
    if enc.returncode != 0:
        sys.exit("[ramz] Encode privacy gagal.")

    report = {
        "source": os.path.abspath(args.input), "method": args.method, "frames": n,
        "detector_runs": runs, "detector_hits": detected, "frames_masked": covered,
        "manual_regions": bool(regions),
        "review_needed": lost_runs,
        "note": ("Semak setiap julat dalam review_needed secara manual (kepala berpaling, gerakan pantas, "
                 "pencahayaan). Tambah --regions untuk tutup jurang. Detection automatik bukan jaminan 100%."),
    }
    side = os.path.splitext(args.output)[0] + ".privacy.json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[ramz] SIAP: {args.output}")
    print(f"[ramz] wajah dikesan {detected}/{runs} detection, mask aktif {covered}/{n} frame; "
          f"{len(lost_runs)} julat perlu semakan -> {side}")


if __name__ == "__main__":
    main()
