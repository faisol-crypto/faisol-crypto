"""Face privacy pass — run on RAW footage before any editing.

Doing privacy first means later zooms, crops and transitions can never expose
an unprotected frame.

  1. Detect  — OpenCV Haar (frontal + both profiles) on every Nth frame.
  2. Track   — gaps are filled by interpolating between detections in BOTH
               directions (no lag when the head moves, no hole when a turned
               head is missed), then smoothed with a centred window so the
               mask never jumps. Size uses a rolling max: grows at once,
               shrinks slowly, and widens while the head is moving.
  3. Render  — feathered blur / mosaic / solid disc into the mask. HDR phone
               footage (HLG/PQ) is tone-mapped to SDR and VFR becomes CFR.
  4. Audit   — for every frame with a detection: (a) coverage — the face
               box must lie fully inside the mask; (b) detail leak — the
               Laplacian (edge) energy left inside the face box after blurring,
               relative to the original. Eyes/nose/mouth = high detail; a
               well-blurred face keeps only a few percent. (A plain face
               detector is NOT used as audit: it still finds the head shape in
               any blur, so it cannot tell protected from exposed.)

Manual keyframed regions (--regions) are always applied on top.

Writes <output>.privacy.json. ramz.render refuses privacy_required plans
without it, and warns if it lists anything for review.

Usage:
  python3 -m ramz.privacy raw.mov -o protected.mp4 [--method soft|pixel|solid] [--regions manual.json]
  manual.json: [{"t": 0.0, "x": 0.4, "y": 0.1, "w": 0.2, "h": 0.25}, {"t": 4.0, ...}]
"""
import argparse
import json
import os
import subprocess
import sys

from .common import probe, require_tools, sdr_filter


def _cv():
    try:
        import cv2
        import numpy as np
    except ImportError:
        sys.exit("[ramz] Perlukan OpenCV: pip install 'opencv-python-headless<5' numpy")
    return cv2, np


class Detector:
    def __init__(self, cv2, width, detect_width=640):
        self.cv2 = cv2
        self.s = min(1.0, detect_width / width)
        hc = cv2.data.haarcascades
        self.frontal = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_frontalface_default.xml"))
        self.profile = cv2.CascadeClassifier(os.path.join(hc, "haarcascade_profileface.xml"))

    def __call__(self, frame, min_frac=0.06, neighbors=5):
        cv2, s = self.cv2, self.s
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        mn = max(24, int(min(gray.shape) * min_frac))
        dets = list(self.frontal.detectMultiScale(gray, 1.12, neighbors, minSize=(mn, mn)))
        dets += list(self.profile.detectMultiScale(gray, 1.12, neighbors, minSize=(mn, mn)))
        gw = gray.shape[1]
        for (x, y, w, h) in self.profile.detectMultiScale(cv2.flip(gray, 1), 1.12, neighbors, minSize=(mn, mn)):
            dets.append((gw - x - w, y, w, h))
        # centre boxes (cx, cy, w, h) in full-res pixels
        return [((x + w / 2) / s, (y + h / 2) / s, w / s, h / s) for (x, y, w, h) in dets]


def frames(path, video, fps):
    """Decode to BGR frames at constant fps, tone-mapping HDR to SDR."""
    import numpy as np
    W, H = video["width"], video["height"]
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"{sdr_filter(video)},fps={fps}",
                             "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    size = W * H * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
    proc.wait()


def pick_primary(dets, prev):
    """Keep the detection nearest the previous face (else the largest)."""
    if not dets:
        return None
    if prev is not None:
        return min(dets, key=lambda d: (d[0] - prev[0]) ** 2 + (d[1] - prev[1]) ** 2)
    return max(dets, key=lambda d: d[2] * d[3])


def build_track(raw, fps, max_gap_s, smooth_s):
    """Interpolate gaps between detections both ways, then smooth.

    Returns per-frame boxes and the time ranges filled over gaps longer than
    max_gap_s (flagged for review).
    """
    import numpy as np
    n = len(raw)
    idx = [i for i in range(n) if raw[i] is not None]
    if not idx:
        return [None] * n, [[0.0, round(n / fps, 2)]] if n else []
    arr = np.full((n, 4), np.nan)
    for i in idx:
        arr[i] = raw[i]
    # drop isolated outliers: a detection far from both neighbours that agree
    for k in range(1, len(idx) - 1):
        a, b, c = arr[idx[k - 1]], arr[idx[k]], arr[idx[k + 1]]
        mid = (a + c) / 2
        if np.hypot(*(b[:2] - mid[:2])) > max(a[2], c[2]) * 1.2 and np.hypot(*(a[:2] - c[:2])) < a[2] * 0.6:
            arr[idx[k]] = np.nan
    idx = [i for i in range(n) if not np.isnan(arr[i, 0])]
    t = np.arange(n)
    out = np.empty((n, 4))
    for c in range(4):
        out[:, c] = np.interp(t, idx, arr[idx, c])  # edges hold first/last value
    long_gaps = [[round(a / fps, 2), round(b / fps, 2)] for a, b in zip(idx, idx[1:]) if (b - a) / fps > max_gap_s]
    if idx[0] / fps > max_gap_s:
        long_gaps.insert(0, [0.0, round(idx[0] / fps, 2)])
    if (n - 1 - idx[-1]) / fps > max_gap_s:
        long_gaps.append([round(idx[-1] / fps, 2), round((n - 1) / fps, 2)])

    k = max(1, int(smooth_s * fps)) | 1
    pad = k // 2
    pos = np.pad(out[:, :2], ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    sm = np.stack([np.convolve(pos[:, c], ker, mode="valid") for c in range(2)], axis=1)
    wpad = pad * 2
    size = np.pad(out[:, 2:], ((wpad, wpad), (0, 0)), mode="edge")
    sz = np.stack([np.array([size[i:i + 2 * wpad + 1, c].max() for i in range(n)]) for c in range(2)], axis=1)
    padded = np.pad(out[:, :2], ((pad, pad), (0, 0)), mode="edge")
    travel = np.array([np.ptp(padded[i:i + k, 0]) + np.ptp(padded[i:i + k, 1]) for i in range(n)])
    sz = sz + travel[:, None] * 0.5  # moving head -> wider mask
    return [tuple(map(float, (sm[i, 0], sm[i, 1], sz[i, 0], sz[i, 1]))) for i in range(n)], long_gaps


def detail(cv2, np, frame, box):
    """Edge energy (Laplacian variance) inside the central face box."""
    cx, cy, w, h = box
    H, W = frame.shape[:2]
    x0, x1 = int(max(0, cx - w * 0.35)), int(min(W, cx + w * 0.35))
    y0, y1 = int(max(0, cy - h * 0.35)), int(min(H, cy + h * 0.4))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()) + 1e-6


def covered_by(box, masks):
    """True if the face box (eyes-to-chin core) is inside the solid part of some mask."""
    cx, cy, w, h = box
    pts = [(cx + dx * w * 0.42, cy + dy * h * 0.45) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for (mx, my, ax, ay) in masks:
        if all(((px - mx) / ax) ** 2 + ((py - my) / ay) ** 2 <= 0.9 for px, py in pts):
            return True
    return False


def to_ranges(times, join=0.5):
    out = []
    for t in times:
        if out and t - out[-1][1] <= join:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


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
    return r if r.get("enabled", True) else None


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
        # downscale -> heavy blur twice -> upscale: strong, smooth and cheap
        k = max(3, int(min(W, H) * 0.09) // 4) | 1
        small = cv2.resize(roi, (max(1, rw // 4), max(1, rh // 4)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(cv2.GaussianBlur(small, (k, k), 0), (k, k), 0)
        cover = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    frame[y0:y1, x0:x1] = (roi * (1 - alpha) + cover * alpha).astype(np.uint8)
    return frame


def main():
    ap = argparse.ArgumentParser(description="RamzlayerG face privacy pass")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--method", choices=["soft", "pixel", "solid"], default="soft")
    ap.add_argument("--pad", type=float, default=0.55, help="padding around face box (fraction)")
    ap.add_argument("--regions", help="manual keyframed regions JSON (normalised coords)")
    ap.add_argument("--no-detect", action="store_true", help="manual --regions only (fixed setups, fastest)")
    ap.add_argument("--detect-every", type=int, default=2, help="run detector every N frames")
    ap.add_argument("--detect-width", type=int, default=640, help="detection runs at this width for speed")
    ap.add_argument("--max-gap", type=float, default=0.8, help="gaps longer than this (s) are flagged for review")
    ap.add_argument("--smooth", type=float, default=0.25, help="position smoothing window (s)")
    ap.add_argument("--color", default="#0B0F14", help="disc colour for --method solid")
    ap.add_argument("--fps", type=float, default=30.0, help="output constant frame rate")
    ap.add_argument("--crf", type=int, default=14, help="near-lossless intermediate")
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--leak-ratio", type=float, default=0.15,
                    help="max fraction of facial edge detail allowed to survive the blur")
    args = ap.parse_args()

    require_tools()
    cv2, np = _cv()
    v = probe(args.input)["video"]
    W, H = v["width"], v["height"]
    fps = float(args.fps)  # constant frame rate from here on (phones record VFR)
    if args.no_detect and not args.regions:
        sys.exit("[ramz] --no-detect perlukan --regions.")
    regions = None
    if args.regions:
        with open(args.regions, encoding="utf-8") as fh:
            regions = sorted(json.load(fh), key=lambda r: r["t"])

    track, long_gaps, runs, hits, raw = None, [], 0, 0, []
    if not args.no_detect:
        print("[ramz] privacy 1/3: detect", flush=True)
        detect = Detector(cv2, W, args.detect_width)
        prev = None
        for i, frame in enumerate(frames(args.input, v, fps)):
            box = None
            if i % max(1, args.detect_every) == 0:
                runs += 1
                box = pick_primary(detect(frame), prev)
                if box is not None:
                    hits += 1
                    prev = box
            raw.append(box)
        track, long_gaps = build_track(raw, fps, args.max_gap, args.smooth)

    print("[ramz] privacy 2/3: render", flush=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", f"{fps}", "-i", "-", "-i", args.input, "-map", "0:v", "-map", "1:a:0?",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
         "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
         "-c:a", "aac", "-b:a", "256k", "-shortest", args.output], stdin=subprocess.PIPE)
    col = tuple(int(args.color.lstrip("#")[i:i + 2], 16) for i in (4, 2, 0))  # BGR
    covered = total = 0
    bad_cover, leaks, ratios = [], [], []
    for i, frame in enumerate(frames(args.input, v, fps)):
        masks = []
        if track and i < len(track) and track[i] is not None:
            cx, cy, w, h = track[i]
            masks.append((cx, cy - h * 0.06, w * (1 + args.pad) / 2, h * (1 + args.pad) * 1.2 / 2))
        r = interp_regions(regions, i / fps)
        if r:
            masks.append(((r["x"] + r["w"] / 2) * W, (r["y"] + r["h"] / 2) * H, r["w"] * W / 2, r["h"] * H / 2))
        det = raw[i] if i < len(raw) else None
        before = detail(cv2, np, frame, det) if det and not args.no_audit else None
        if masks:
            covered += 1
            frame = apply_masks(cv2, np, frame, masks, args.method, col)
        if before:
            if not covered_by(det, masks):
                bad_cover.append(round(i / fps, 2))
            ratio = detail(cv2, np, frame, det) / before
            ratios.append(ratio)
            if ratio > args.leak_ratio:
                leaks.append(round(i / fps, 2))
        enc.stdin.write(frame.tobytes())
        total += 1
    enc.stdin.close()
    enc.wait()
    if enc.returncode != 0:
        sys.exit("[ramz] Encode privacy gagal.")

    leak_ranges, cover_ranges = to_ranges(leaks), to_ranges(bad_cover)

    report = {
        "source": os.path.abspath(args.input), "method": args.method, "fps": fps, "frames": total,
        "hdr_tonemapped": bool(v.get("hdr")), "detector_runs": runs, "detector_hits": hits,
        "frames_masked": covered, "manual_regions": bool(regions),
        "long_gaps_interpolated": long_gaps,
        "audit": {
            "frames_checked": len(ratios),
            "detail_left_median": round(float(np.median(ratios)), 4) if ratios else None,
            "detail_left_max": round(float(np.max(ratios)), 4) if ratios else None,
            "leak_threshold": args.leak_ratio,
            "detail_leaks": leak_ranges,
            "coverage_failures": cover_ranges,
        },
        "review_needed": long_gaps + leak_ranges + cover_ranges,
        "note": ("Audit: setiap frame yang ada wajah dikesan disemak — kotak wajah mesti di dalam mask, dan "
                 "butiran (edge) yang tinggal mesti di bawah ambang. Julat dalam review_needed mesti ditonton "
                 "manual. Pengesanan automatik bukan jaminan 100% — wajah yang tidak pernah dikesan tidak "
                 "dapat diaudit."),
    }
    side = os.path.splitext(args.output)[0] + ".privacy.json"
    with open(side, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[ramz] SIAP: {args.output}")
    med = f"{np.median(ratios) * 100:.1f}%" if ratios else "-"
    print(f"[ramz] wajah dikesan {hits}/{runs} detection, mask {covered}/{total} frame, "
          f"{len(long_gaps)} jurang panjang; audit {len(ratios)} frame: butiran tinggal median {med}, "
          f"{len(leaks)} leak, {len(bad_cover)} liputan gagal -> {side}")


if __name__ == "__main__":
    main()
