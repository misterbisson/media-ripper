#!/usr/bin/env python3
"""Make a video QuickTime plays, at its own size, beside the original.

For video in codecs macOS no longer decodes (Sorenson, Cinepak, Apple Video,
DivX, and QDesign or ADPCM sound, from QuickTime's early days), or in containers
it never opened (AVI, WMV), and for video that's merely old. It writes
"<name>.m4v" beside the original, which it leaves alone:

- H.264 and HEVC video is copied; anything else becomes H.264 (x264 CRF 18),
  deinterlaced if it's interlaced, and back at 24 frames a second if it's film
  with 3:2 pulldown, both found by looking. --fields says which instead, and
  with it H.264 and HEVC are re-encoded too.
- Audio QuickTime plays is copied; the rest becomes AAC, or AC3 above stereo.
- Text subtitles, chapters, cover art, and the title, show, season and episode
  an iTunes file carries, come along. DVD subtitles become text (vobsub.py).

Copy-protected iTunes Store purchases can't be read, and are refused. Needs
`brew install ffmpeg`, and the Xcode command-line tools for DVD subtitles.

Usage:
    ./transcode.py "Old Trailer.mov"
    ./transcode.py ~/Movies/Clips/*.avi
    ./transcode.py film.mp4 --crf 20
    ./transcode.py "TV recording.mp4" --fields interlaced
"""

import argparse
import shutil
import sys
from pathlib import Path

import quicktime


def transcode(src, args):
    dst = src.with_suffix(".m4v")
    if dst == src:  # already an .m4v, perhaps with sound QuickTime drops
        dst = src.with_name(f"{src.stem} - QuickTime.m4v")
    if dst.exists() and not args.force:
        print(f"{dst.name} already exists; skipping {src.name} (--force to replace it)")
        return
    if why := quicktime.problem(src):
        print(f"Skipping {why}")
        return
    print(f"{src.name} → {dst.name}", flush=True)
    srts = quicktime.read_subtitles(src, dst)
    part = dst.with_name(dst.stem + ".part" + dst.suffix)
    try:
        quicktime.make_mp4(src, part, srts, crf=args.crf, preset=args.preset, fields=args.fields)
        part.rename(dst)
    finally:
        for f in (part, *srts.values()):
            f.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", type=Path, nargs="+", metavar="video", help="videos to make playable")
    ap.add_argument("--crf", type=float, default=18,
                    help="x264 quality for video that's re-encoded, lower is better (default: 18)")
    ap.add_argument("--preset", default="slow", help="x264 preset (default: slow)")
    ap.add_argument("--fields", choices=("auto", *quicktime.FIELDS), default="auto",
                    help="whether the video is interlaced, or film with 3:2 pulldown "
                         "(default: auto, found by looking at video that's re-encoded)")
    ap.add_argument("-f", "--force", action="store_true", help="replace an .m4v made before")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found: brew install ffmpeg")
    for src in args.inputs:
        src = src.expanduser().resolve()
        if not src.exists():
            sys.exit(f"{src}: not found")
        transcode(src, args)


if __name__ == "__main__":
    main()
