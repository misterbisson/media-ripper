#!/usr/bin/env python3
"""Find where a file that holds several episodes joins them, and cut it there.

A disc's "play all" title, or a download of a whole series, puts episodes end to
end in one file. Where one ends and the next begins, the picture goes black and the
sound goes quiet together. So does an ad break, so the joins listed are candidates:
look at the frames either side in a player before cutting. --sheet draws them.

Cutting copies the video and sound without re-encoding, so each cut lands on the
first keyframe at or after the time given, which can be a second or two late. It
says where each part really begins. The parts are written beside the file (or in
--out) as "<name> - part 1.mp4" and so on, and the file itself is left alone.

Usage:
    ./episodes.py "Series.mp4"                         # list the candidate joins
    ./episodes.py "Series.mp4" --sheet joins.jpg       # and the frames either side
    ./episodes.py "Series.mp4" --cut 1320,2640,3960    # write the parts
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BLACK = 0.2     # seconds of black, at least, for a join
QUIET = -42     # dB, quieter than which is silence
LONG = 0.6      # seconds black and silent together, at least, to be listed


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout)


def joins(path):
    """(start, end) of each stretch where the picture is black and the sound silent"""
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
         "-map", "0:V:0", "-vf", f"blackdetect=d={BLACK}:pic_th=0.9:pix_th=0.1", "-f", "null", "-",
         "-map", "0:a:0", "-af", f"silencedetect=n={QUIET}dB:d=0.25", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", log)]
    quiet = list(zip(map(float, re.findall(r"silence_start: ([\d.]+)", log)),
                     map(float, re.findall(r"silence_end: ([\d.]+)", log))))
    # A join is black for a while, with silence starting or ending within a second of it
    return [(round(a, 2), round(b, 2)) for a, b in black
            if b - a >= LONG and any(x - 1 <= b and a - 1 <= y for x, y in quiet)]


def clock(s):
    s = int(s)
    return f"{s // 3600}:{s // 60 % 60:02}:{s % 60:02}"


def sheet(path, times, out):
    """Five frames around each join, a row each: 6 s before, then 4, 30, 60 and 100 s after"""
    with tempfile.TemporaryDirectory() as work:
        n = 0
        for t in times:
            for d in (-6, 4, 30, 60, 100):
                n += 1
                frame = Path(work) / f"{n:04}.png"
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0, t + d):.2f}",
                                "-i", str(path), "-map", "0:V:0", "-frames:v", "1", "-vf",
                                "scale=240:-2", str(frame)])
                if not frame.exists():  # past the end: a blank keeps the row in step
                    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                    "color=c=0x222222:s=240x136", "-frames:v", "1", str(frame)])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{work}/%04d.png", "-vf",
                        f"scale=240:136:force_original_aspect_ratio=decrease,pad=240:136:-1:-1:color=0x222222,tile=5x{len(times)}",
                        "-frames:v", "1", str(out)], check=True)


def cut(path, at, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / f"{path.stem} - part %d.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:V:0",
                    "-map", "0:a", "-c", "copy", "-f", "segment", "-segment_start_number", "1",
                    "-segment_times", ",".join(f"{t:.2f}" for t in at), "-reset_timestamps", "1",
                    "-segment_format", "mp4", "-segment_format_options", "movie_timescale=1000",
                    str(pattern)], check=True)
    begun = 0.0
    for n in range(1, len(at) + 2):
        part = Path(str(pattern) % n)
        if not part.exists():
            break
        length = duration(part)
        asked = 0.0 if n == 1 else at[n - 2]
        late = f", {begun - asked:.1f} s after the time asked" if n > 1 and begun - asked > 0.05 else ""
        print(f"{part.name}: from {clock(begun)} for {clock(length)}{late}")
        begun += length


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("file", type=Path)
    p.add_argument("--sheet", type=Path, help="draw the frames around each join into this image")
    p.add_argument("--cut", help="times to cut at, in seconds, comma-separated")
    p.add_argument("--out", type=Path, help="the folder to write parts into, if not beside the file")
    args = p.parse_args()

    if args.cut:
        cut(args.file, [float(t) for t in args.cut.split(",")], args.out or args.file.parent)
        return
    print(f"Looking through {args.file.name} …", flush=True)
    found = joins(args.file)
    if not found:
        print("No stretch is black and silent together.")
        return
    for a, b in found:
        print(f"  {clock(b)}  black from {a:.2f} to {b:.2f} s")
    if args.sheet:
        sheet(args.file, [b for _, b in found], args.sheet)
        print(f"{args.sheet}: a row for each, 6 s before the join, then 4, 30, 60 and 100 s after")


if __name__ == "__main__":
    main()
