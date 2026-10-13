"""Generate a small, royalty-free starter SFX kit procedurally with ffmpeg.

These are clean, restrained utility sounds (whoosh, impact, riser, click ...).
For flagship videos a licensed library will still sound richer — swap the
files in assets/sfx/ and keep the same names.

Usage:
  python3 -m ramz.sfx [-o assets/sfx]
"""
import argparse
import os

from .common import ROOT, require_tools, run

SR = 48000

# name: (duration, aevalsrc expression, post-filter chain)
KIT = {
    # band-passed noise that sweeps up then down, bell-shaped envelope
    "whoosh": (0.55, "(random(0)*2-1)*sin(PI*t/0.55)^2",
               "bandpass=f=1400:width_type=o:w=2.2,highpass=f=180,volume=3.0,"
               "afade=t=out:st=0.4:d=0.15"),
    "reverse_whoosh": (0.6, "(random(0)*2-1)*(t/0.6)^3",
                       "bandpass=f=1600:width_type=o:w=2,highpass=f=200,volume=3.2,afade=t=in:d=0.05"),
    "soft_impact": (0.9, "0.9*sin(2*PI*(58+60*exp(-t*18))*t)*exp(-t*5.5)+0.25*(random(0)*2-1)*exp(-t*28)",
                    "lowpass=f=2400,volume=1.2"),
    "bass_hit": (1.2, "sin(2*PI*(42+80*exp(-t*25))*t)*exp(-t*3.2)",
                 "lowpass=f=400,volume=1.5"),
    "riser": (1.6, "(0.5*(random(0)*2-1)+0.5*sin(2*PI*(200+900*(t/1.6)^2)*t))*(t/1.6)^2",
              "highpass=f=250,lowpass=f=6000,volume=1.4,afade=t=out:st=1.55:d=0.05"),
    "click": (0.06, "(random(0)*2-1)*exp(-t*220)+0.5*sin(2*PI*2200*t)*exp(-t*160)",
              "highpass=f=900,volume=0.9"),
    "ui_blip": (0.18, "0.6*sin(2*PI*1320*t)*exp(-t*28)+0.3*sin(2*PI*1980*t)*exp(-t*40)",
                "volume=0.9"),
    "typing": (1.2, "(random(0)*2-1)*exp(-mod(t,0.11+0.02*sin(9*t))*150)",
               "highpass=f=1500,lowpass=f=7000,volume=0.6"),
    "glitch": (0.25, "(random(0)*2-1)*lt(mod(t,0.05),0.025)+0.4*sin(2*PI*90*t)*lt(mod(t,0.08),0.03)",
               "highpass=f=300,acrusher=bits=6:mode=lin:mix=0.6,volume=0.6"),
    "ambient_bed": (20.0, "0.6*(random(0)*2-1)",
                    "lowpass=f=420,highpass=f=60,volume=1.2,afade=t=in:d=2,afade=t=out:st=18:d=2"),
}


def generate(out_dir):
    require_tools()
    os.makedirs(out_dir, exist_ok=True)
    for name, (dur, expr, post) in KIT.items():
        path = os.path.join(out_dir, f"{name}.wav")
        run(["ffmpeg", "-y", "-hide_banner", "-f", "lavfi",
             "-i", f"aevalsrc='{expr}':s={SR}:d={dur}:c=stereo",
             "-af", f"{post},alimiter=limit=0.7:level=0", "-c:a", "pcm_s16le", path])
        print(f"[ramz] {path}")


def main():
    ap = argparse.ArgumentParser(description="Generate starter SFX kit")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "assets", "sfx"))
    generate(ap.parse_args().out)


if __name__ == "__main__":
    main()
