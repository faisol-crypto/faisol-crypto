"""Original, royalty-free ambient bed composed procedurally (no samples).

A warm minor-key pad (Am7 - Fmaj7 - Cmaj7 - G6) with a soft sub, slow
attacks, gentle movement and a light reverb tail. Meant to sit far under the
voice (the renderer ducks it further whenever speech is present).

Usage:
  python3 -m ramz.music -o assets/music/sirramz_bed.wav --seconds 40
"""
import argparse
import os
import subprocess

import numpy as np

SR = 48000
CHORDS = [  # MIDI note numbers
    [45, 52, 55, 60, 64],  # Am7
    [41, 48, 52, 57, 60],  # Fmaj7
    [48, 55, 59, 64, 67],  # Cmaj7
    [43, 50, 55, 59, 64],  # G6
]


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def voice(freq, n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for det in (-0.0018, 0.0, 0.0021):
        ph = rng.uniform(0, 2 * np.pi)
        f = freq * (1 + det)
        vib = 0.0012 * np.sin(2 * np.pi * rng.uniform(0.15, 0.3) * t + ph)
        for k, amp in ((1, 1.0), (2, 0.28), (3, 0.09), (4, 0.03)):
            out += amp * np.sin(2 * np.pi * f * k * t * (1 + vib) + ph * k)
    return out / 3


def envelope(n, attack, release):
    env = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    env[:a] = np.linspace(0, 1, a) ** 2
    env[-r:] *= np.linspace(1, 0, r) ** 1.5
    return env


def compose(seconds, chord_len=4.8, seed=7):
    rng = np.random.default_rng(seed)
    total = int(seconds * SR)
    L = np.zeros(total + SR * 3)
    R = np.zeros_like(L)
    step = int(chord_len * SR)
    hold = int((chord_len + 1.6) * SR)  # overlap -> legato chord changes
    i = 0
    pos = 0
    while pos < total:
        chord = CHORDS[i % len(CHORDS)]
        env = envelope(hold, 1.4, 1.8)
        for j, m in enumerate(chord):
            v = voice(hz(m), hold, rng) * env * (0.55 if j == 0 else 0.32)
            pan = 0.5 + (j - 2) * 0.12
            L[pos:pos + hold] += v * (1 - pan)
            R[pos:pos + hold] += v * pan
        sub = np.sin(2 * np.pi * hz(chord[0] - 12) * np.arange(hold) / SR) * env * 0.35
        L[pos:pos + hold] += sub
        R[pos:pos + hold] += sub
        pos += step
        i += 1
    # slow "breathing" swell
    t = np.arange(len(L)) / SR
    swell = 0.85 + 0.15 * np.sin(2 * np.pi * t / 9.6)
    L, R = L * swell, R * swell
    st = np.stack([L, R], axis=1)[:total]
    st /= np.abs(st).max() + 1e-9
    return (st * 0.6).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Compose an original ambient bed")
    ap.add_argument("-o", "--out", default="assets/music/sirramz_bed.wav")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    audio = compose(args.seconds + 2, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # warm it up: low-pass, gentle reverb tail, fades, peak-safe
    post = ("lowpass=f=3200,highpass=f=45,aecho=0.8:0.55:60|97|151:0.28|0.2|0.14,"
            f"afade=t=in:d=1.5,afade=t=out:st={args.seconds - 2.5}:d=2.5,alimiter=limit=0.7:level=0")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                    "-af", post, "-t", str(args.seconds), "-c:a", "pcm_s16le", args.out],
                   input=audio.tobytes(), check=True)
    print(f"[ramz] SIAP: {args.out}")


if __name__ == "__main__":
    main()
