#!/usr/bin/env bash
# Build a 40s 1920x1080 synthetic "screen recording" with speech-like audio
# (tone bursts + gaps) to smoke-test the whole pipeline without real footage.
set -euo pipefail
OUT=${1:-tests/out/synthetic_raw.mp4}
mkdir -p "$(dirname "$OUT")"
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=s=1920x1080:r=30:d=40" \
  -f lavfi -i "aevalsrc='0.3*sin(2*PI*(180+40*sin(3*t))*t)*gt(sin(2*PI*0.45*t),-0.3)*(0.6+0.4*sin(2*PI*5*t))':s=48000:d=40:c=stereo" \
  -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k -shortest "$OUT"
echo "$OUT"
