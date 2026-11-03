#!/usr/bin/env python3
"""Read subtitles burned into a video's picture, as text, when a clean copy exists.

Two files of one film often differ only in their subtitles: one in the original
language with English burned into the picture, one dubbed with a clean picture.
Decoding both together, what differs in the lower part of the frame is the text and
nothing else, so each subtitle can be found, timed and read without guessing where
it is. macOS's own OCR (Vision, through vobsub_ocr.swift) reads each one twice: once
from the difference, once from the subtitled frames with everything else blanked.

It writes "<subtitled>.<lang>.srt" beside the subtitled file (or in --out), and
"<subtitled>.<lang>.review.txt" listing each line where the two readings disagree,
with both, which is where a person should look. --pictures writes those lines'
frames as PNGs too.

The two files must be the same picture on the same timeline: the same cut, the same
size, starting together. Nothing is changed in either. Needs numpy (it uses the
upscaler's .venv if there is one), ffmpeg, and the Xcode command-line tools.

Usage:
    ./burnedsubs.py "Film (subtitled).m4v" "Film.m4v"
    ./burnedsubs.py "Film (subtitled).m4v" "Film.m4v" --start 600 --length 300
    ./burnedsubs.py "Film (subtitled).m4v" "Film.m4v" --pictures review/
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import numpy as np
except ImportError:
    venv = HERE / ".venv" / "bin" / "python"
    if venv.exists() and Path(sys.executable) != venv:
        os.execv(str(venv), [str(venv), str(HERE / "burnedsubs.py"), *sys.argv[1:]])
    sys.exit("burnedsubs.py needs numpy: python3 -m pip install numpy")

import vobsub

FPS = 5          # frames looked at a second: a subtitle's times are good to 0.2 s
BAND = 0.34      # the share of the picture, from the bottom, where subtitles are looked for
DIFF = 38        # how far two copies' grey must differ for a pixel to be text
MIN_PIXELS = 70  # fewer differing pixels than this is noise, not a subtitle
SAME = 0.55      # overlap with the frame before, below which a new subtitle has begun
WORDS = Path("/usr/share/dict/words")


def size(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return s["width"], s["height"]


def frames(path, start, length):
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(path)]
    if length:
        cmd += ["-t", str(length)]
    cmd += ["-map", "0:V:0", "-vf", f"fps={FPS},format=gray", "-f", "rawvideo", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def find(subtitled, clean, start=0, length=None):
    """Each subtitle as (start, end, mean difference, mean subtitled picture)."""
    w, h = size(subtitled)
    if size(clean) != (w, h):
        sys.exit(f"{subtitled.name} is {w}×{h} and {clean.name} is "
                 f"{'×'.join(map(str, size(clean)))}: they must be the same picture")
    top = int(h * (1 - BAND))
    a, b = frames(subtitled, start, length), frames(clean, start, length)
    n, i, found, cue = w * h, 0, [], None

    def close():
        # Under two frames, 0.4 s, is a difference passing through, not a subtitle
        if cue and cue["n"] >= 2:
            found.append((cue["start"], cue["end"], cue["diff"] / cue["n"], cue["pict"] / cue["n"]))

    while True:
        ra, rb = a.stdout.read(n), b.stdout.read(n)
        if len(ra) < n or len(rb) < n:
            break
        fa = np.frombuffer(ra, np.uint8).reshape(h, w)[top:].astype(np.int16)
        fb = np.frombuffer(rb, np.uint8).reshape(h, w)[top:].astype(np.int16)
        d = np.abs(fa - fb)
        m = d > DIFF
        t = start + i / FPS
        i += 1
        # Text sits in rows. Differences scattered over the band are the two
        # encodes disagreeing, as they do at a cut
        if m.sum() < MIN_PIXELS or ((m.sum(1) >= 6).sum() < 5):
            close()
            cue = None
            continue
        # The same subtitle has nearly the same pixels frame to frame. One that
        # follows straight on from another doesn't, even with as much ink in the
        # same rows, and averaging the two makes nonsense of both
        if cue is not None:
            if (m & cue["last"]).sum() / max((m | cue["last"]).sum(), 1) < SAME:
                close()
                cue = None
        if cue is None:
            cue = dict(start=t, end=t + 1 / FPS, n=1, diff=d.astype(np.float32),
                       pict=fa.astype(np.float32), last=m)
        else:
            cue.update(end=t + 1 / FPS, n=cue["n"] + 1, last=m)
            cue["diff"] += d
            cue["pict"] += fa
        if i % (FPS * 600) == 0:
            print(f"  {int(t) // 60} min: {len(found)} subtitles", flush=True)
    close()
    a.kill()
    b.kill()
    return found


def from_difference(diff):
    """Dark text on white, as the reader expects: the difference is bright where the text is"""
    k = 255.0 / max(np.percentile(diff, 99.5), 40)
    return (255 - np.clip(diff * k, 0, 255)).astype(np.uint8)


def from_picture(diff, pict):
    """The subtitled frames themselves, with everything that isn't the subtitle blanked"""
    text = diff > 18
    grown = text.copy()
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        grown |= np.roll(text, shift, axis)
    return np.where(grown, 255 - pict, 255).astype(np.uint8)


def read(pictures, language):
    data = b"".join(struct.pack(">II", p.shape[1], p.shape[0]) + p.tobytes() for p in pictures)
    out = subprocess.run([str(vobsub.ocr_tool()), language], input=data, capture_output=True,
                         check=True)
    return [json.loads(line).strip() for line in out.stdout.decode().splitlines()]


def words():
    return {w.strip().lower() for w in WORDS.open()} if WORDS.exists() else set()


def score(text, known):
    """How much of a reading looks like words: the share in the word list, less any
    stray characters OCR invents (®, ạ, |)"""
    found = re.findall(r"[A-Za-z']+", text)
    if not found:
        return 0.0
    ok = sum(1 for w in found
             if w[0].isupper() or w.lower().strip("'").removesuffix("'s") in known
             or w.lower().rstrip("s") in known)
    stray = len(re.findall(r"[^A-Za-z0-9\s,.'!?\-\"…:;()]", text))
    return ok / len(found) - stray * 0.2


def same(a, b):
    return re.sub(r"\W", "", a).lower() == re.sub(r"\W", "", b).lower()


def write_png(gray, path):
    h, w = gray.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s",
                    f"{w}x{h}", "-i", "-", "-vf", "scale=iw*2:ih*2", str(path)],
                   input=gray.tobytes(), check=True)


def stamp(s):
    ms = round(s * 1000)
    return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("subtitled", type=Path, help="the copy with subtitles in its picture")
    p.add_argument("clean", type=Path, help="a copy of the same picture without them")
    p.add_argument("--lang", default="eng", help="the subtitles' language, ISO 639-2 (eng)")
    p.add_argument("--start", type=float, default=0, help="seconds in to begin, for a trial")
    p.add_argument("--length", type=float, help="seconds to read, for a trial")
    p.add_argument("--out", type=Path, help="the folder to write into, if not beside the file")
    p.add_argument("--pictures", type=Path, help="write the lines to review as PNGs here")
    args = p.parse_args()

    print(f"Finding the subtitles in {args.subtitled.name} …", flush=True)
    found = find(args.subtitled, args.clean, args.start, args.length)
    print(f"Reading {len(found)} …", flush=True)
    one = read([from_difference(d) for _, _, d, _ in found], args.lang)
    two = read([from_picture(d, q) for _, _, d, q in found], args.lang)
    known = words()

    cues = []  # (start, end, text, agreed, index)
    for k, ((s, e, _, _), a, b) in enumerate(zip(found, one, two)):
        text = a if score(a, known) >= score(b, known) else b
        if not text:
            continue
        if cues and same(cues[-1][2], text) and s - cues[-1][1] < 0.5:
            cues[-1] = (cues[-1][0], e, cues[-1][2], cues[-1][3] and same(a, b), cues[-1][4])
            continue
        cues.append((s, e, text, same(a, b), k))

    base = (args.out / args.subtitled.stem) if args.out else args.subtitled.with_suffix("")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    srt = Path(f"{base}.{args.lang}.srt")
    srt.write_text("".join(f"{n}\n{stamp(s)} --> {stamp(e)}\n{t}\n\n"
                           for n, (s, e, t, _, _) in enumerate(cues, 1)))
    review = [(n, c) for n, c in enumerate(cues, 1) if not c[3]]
    lines = []
    for n, (s, e, t, _, k) in review:
        lines.append(f"{n}  {stamp(s)}\n  kept: {t!r}\n  one:  {one[k]!r}\n  two:  {two[k]!r}\n")
        if args.pictures:
            args.pictures.mkdir(parents=True, exist_ok=True)
            write_png(found[k][3].astype(np.uint8), args.pictures / f"{n:04}.png")
    Path(f"{base}.{args.lang}.review.txt").write_text("\n".join(lines))
    print(f"{srt.name}: {len(cues)} lines, {len(review)} where the two readings disagree "
          f"({Path(f'{base}.{args.lang}.review.txt').name})")


if __name__ == "__main__":
    main()
