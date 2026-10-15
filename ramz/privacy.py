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

Identity mode (--only me.jpg): blur ONE person and leave everyone else as is.
YuNet detects every face on every frame, SFace embeds it, faces are linked
into tracks, and each track is judged on its mean embedding (one bad angle
cannot flip it). The reference is bootstrapped: the photo finds the person's
most certain frames, and those frames (same day, same light) become the
reference. A track is the person if it is close to that reference, or close
enough and clearly closer to it than to every other confident identity in the
video. Gaps up to --join-gap inside the person's appearances are bridged.
Models: footage/models/ (opencv_zoo YuNet 2023mar + SFace 2021dec).

Writes <output>.privacy.json. ramz.render refuses privacy_required plans
without it, and warns if it lists anything for review.

Usage:
  python3 -m ramz.privacy raw.mov -o protected.mp4 [--method soft|pixel|solid] [--regions manual.json]
  python3 -m ramz.privacy raw.mov -o protected.mp4 --only footage/ref/me.jpg   # only this person
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


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "footage", "models")
YUNET = "face_detection_yunet_2023mar.onnx"
SFACE = "face_recognition_sface_2021dec.onnx"


class IdentityDetector:
    """YuNet faces + SFace embeddings (unit vectors)."""

    def __init__(self, cv2, model_dir=MODEL_DIR):
        paths = [os.path.join(model_dir, m) for m in (YUNET, SFACE)]
        for pth in paths:
            if not os.path.exists(pth) or os.path.getsize(pth) < 100000:
                sys.exit(f"[ramz] Model tiada: {pth}\n       Muat turun dari "
                         "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
                         "{face_detection_yunet,face_recognition_sface}/<fail>")
        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(paths[0], "", (320, 320), 0.55, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(paths[1], "")

    def __call__(self, img, min_embed=24):
        import numpy as np
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        out = []
        for f in ([] if faces is None else faces):
            e = None
            if f[2] >= min_embed:
                e = self.rec.feature(self.rec.alignCrop(img, f)).flatten()
                e = e / np.linalg.norm(e)
            x, y, w, h = map(float, f[:4])
            out.append((uncrop(x, y, w, h, img.shape[1], img.shape[0]), e))
        return out


def uncrop(x, y, w, h, W, H, edge=3, aspect=0.8):
    """A face cut by the frame edge is detected narrow/short: restore its
    likely full size from the other side and push the centre outwards, so the
    mask covers the visible part with full margin. Returns (cx, cy, w, h)."""
    if x <= edge or x + w >= W - edge:
        full = max(w, h * aspect)
        x = x + w - full if x <= edge else x
        w = full
    if y <= edge or y + h >= H - edge:
        full = max(h, w / aspect)
        y = y + h - full if y <= edge else y
        h = full
    return (x + w / 2, y + h / 2, w, h)


def link_tracks(dets, max_skip=8):
    """dets: per-frame list of (box, emb). Returns tracks: lists of (frame, box, emb)."""
    import numpy as np
    tracks = []
    for i, row in enumerate(dets):
        taken = set()
        for box, emb in sorted(row, key=lambda d: -d[0][2]):
            best, score = None, 0.0
            for k, t in enumerate(tracks):
                fi, lb, _ = t[-1]
                if k in taken or i - fi > max_skip:
                    continue
                dist = np.hypot(lb[0] - box[0], lb[1] - box[1]) / max(lb[2], box[2])
                sc = (1 - dist) * min(lb[2], box[2]) / max(lb[2], box[2]) if dist < 0.7 else 0
                if sc > score:
                    best, score = k, sc
            if best is None:
                tracks.append([])
                best = len(tracks) - 1
            tracks[best].append((i, box, emb))
            taken.add(best)
    return tracks


def unit_mean(vs):
    import numpy as np
    m = np.mean(vs, 0)
    return m / np.linalg.norm(m)


def select_person(tracks, ref, accept=0.52, maybe=0.40, margin=0.10, seed=0.55):
    """Decide which tracks are the reference person. Returns (flags, report rows)."""
    import numpy as np
    embs = [[e for _, _, e in t if e is not None] for t in tracks]
    # bootstrap the reference from the video's own most certain frames
    sure = [e for es in embs for e in es if float(e @ ref) >= seed]
    U = unit_mean(sure) if len(sure) >= 15 else ref
    means = [unit_mean(es) if es else None for es in embs]
    # other confident identities: sizeable tracks clearly not the person
    others = [m for m, t in zip(means, tracks) if m is not None and len(t) >= 15 and float(m @ U) < maybe]
    flags, rows = [], []
    for t, m in zip(tracks, means):
        if m is None:
            flags.append(False)
            continue
        su = float(m @ U)
        so = max([float(m @ o) for o in others if o is not m] or [0.0])
        yes = su >= accept or (su >= maybe and su > so + margin)
        flags.append(yes)
        if len(t) >= 10:
            rows.append({"t0": t[0][0], "t1": t[-1][0], "frames": len(t), "sim_person": round(su, 3),
                         "sim_other": round(so, 3), "person": yes})
    return flags, rows, len(sure)


def person_raw(tracks, flags, n):
    """Per-frame box of the person (largest if several)."""
    raw = [None] * n
    for t, f in zip(tracks, flags):
        if not f:
            continue
        for i, box, _ in t:
            if raw[i] is None or box[2] > raw[i][2]:
                raw[i] = box
    return raw


def build_segments(raw, fps, join_gap_s, smooth_s, extend_s=0.3, gap_grow=1.6):
    """Like build_track, but only across the person's appearances: gaps longer
    than join_gap_s split segments, and nothing is masked between segments.
    Each segment is extended by extend_s (holding its edge box) both ways."""
    n = len(raw)
    idx = [i for i in range(n) if raw[i] is not None]
    track, gaps = [None] * n, []
    if not idx:
        return track, gaps, []
    runs, cur = [], [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if (b - a) / fps > join_gap_s:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    ext = int(round(extend_s * fps))
    spans, extra = [], [[] for _ in range(n)]
    for run in runs:
        a, b = max(0, run[0] - ext), min(n - 1, run[-1] + ext)
        sub, g = build_track(raw[a:b + 1], fps, 0.8, smooth_s)
        for k, box in enumerate(sub):
            if box is not None and (track[a + k] is None or box[2] > track[a + k][2]):
                track[a + k] = box
        gaps += [[round(x + a / fps, 2), round(y + a / fps, 2)] for x, y in g]
        spans.append([round(a / fps, 2), round(b / fps, 2)])
        # inside a gap the head may leave the straight line between the two
        # detections (e.g. out past the frame edge and back): also hold both
        # end boxes over the gap, enlarged: undetected means the head may be
        # anywhere near its last/next position (turning, entering from an edge)
        for p, q in zip(run, run[1:]):
            if q - p >= 4:
                for k in range(p + 1, q):
                    extra[k] += [(b[0], b[1], b[2] * gap_grow, b[3] * gap_grow) for b in (raw[p], raw[q])]
    return track, gaps, spans, extra


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
    # strength follows the face: a big close-up face (wide selfie lens) needs
    # coarser blocks / a wider blur than a small one to lose its features
    face = max(2 * max(ax, ay) for _, _, ax, ay in masks)
    if method == "solid":
        cover = np.empty_like(roi)
        cover[:] = col
    elif method == "pixel":
        blk = max(8, int(min(W, H) * 0.035), int(face / 7))
        mosaic = cv2.resize(roi, (max(1, rw // blk), max(1, rh // blk)), interpolation=cv2.INTER_AREA)
        cover = cv2.GaussianBlur(cv2.resize(mosaic, (rw, rh), interpolation=cv2.INTER_NEAREST), (blk | 1, blk | 1), 0)
    else:
        # downscale -> heavy blur twice -> upscale: strong, smooth and cheap
        k = max(3, int(min(W, H) * 0.09) // 4, int(face * 0.16) // 4) | 1
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
    ap.add_argument("--only", help="reference photo: blur ONLY this person (YuNet + SFace identity tracking)")
    ap.add_argument("--join-gap", type=float, default=1.2,
                    help="--only: bridge gaps up to this long (s) inside the person's appearances")
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
    identity, hold = None, None
    if args.only and not args.no_detect:
        print("[ramz] privacy 1/3: detect + identiti (hanya orang dalam foto rujukan)", flush=True)
        idd = IdentityDetector(cv2)
        ref_img = cv2.imread(args.only)
        if ref_img is None:
            sys.exit(f"[ramz] Foto rujukan tidak boleh dibaca: {args.only}")
        ref_faces = [d for d in idd(ref_img) if d[1] is not None]
        if not ref_faces:
            sys.exit("[ramz] Tiada wajah dikesan dalam foto rujukan.")
        ref = max(ref_faces, key=lambda d: d[0][2])[1]
        dets = []
        for frame in frames(args.input, v, fps):
            dets.append(idd(frame))
        runs = len(dets)
        tracks = link_tracks(dets)
        flags, rows, n_sure = select_person(tracks, ref)
        raw = person_raw(tracks, flags, len(dets))
        hits = sum(r is not None for r in raw)
        track, long_gaps, spans, hold = build_segments(raw, fps, args.join_gap, args.smooth)
        others = sum(len(t) for t, f in zip(tracks, flags) if not f)
        identity = {"reference": os.path.abspath(args.only), "bootstrap_frames": n_sure,
                    "person_segments": spans, "tracks": rows, "other_faces_left_unblurred": others}
        for r in rows:
            print(f"[ramz]   track {r['t0'] / fps:6.2f}-{r['t1'] / fps:6.2f}s  orang={r['sim_person']:.2f} "
                  f"lain={r['sim_other']:.2f}  -> {'BLUR' if r['person'] else 'biar'}")
    elif not args.no_detect:
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
        if hold and i < len(hold):
            for (hx, hy, hw, hh) in hold[i]:
                masks.append((hx, hy - hh * 0.06, hw * (1 + args.pad) / 2, hh * (1 + args.pad) * 1.2 / 2))
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
        "identity": identity,
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
