#!/usr/bin/env python3
"""Put sound and subtitle tracks from other files into one MP4 with the picture.

For two or more files of one film that differ only in what you hear: the film and a
copy with a commentary in place of its sound, or a dub and the original language.
One file keeps them all, chosen in QuickTime's Audio and Subtitles menus. Nothing is
re-encoded, and no input is changed: it writes --out, which must not exist yet.

Each --sound is the first sound track of the file named, in the order given. The
first is the one a player starts with. Before writing, it checks that each added
track keeps time with the picture file's own sound, at three places through the
film, since a commentary sits over the film's own sound and a dub keeps its music
and effects. A track more than --tolerance seconds off at any of them stops it;
--no-check skips that, for sound with nothing in common, such as a silent picture.

--subtitles adds an .srt as a text track (burnedsubs.py makes one from subtitles in
a picture). Chapters and cover art come from --picture, and tags from --tags-from,
since the better picture is often the copy whose tags say least.

Usage:
    ./addtracks.py --picture "Film.m4v" \\
        --sound "Film.m4v" eng "English" \\
        --sound "Film (commentary).m4v" eng "Director's commentary" \\
        --out "Film - with commentary.m4v"
    ./addtracks.py --picture "Film (dubbed).m4v" \\
        --sound "Film (subtitled).m4v" fra "French" \\
        --sound "Film (dubbed).m4v" eng "English dub" \\
        --subtitles "Film (subtitled).eng.srt" eng "English" \\
        --tags-from "Film (subtitled).m4v" --out "Film - French and English.m4v"
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    import numpy as np
except ImportError:
    venv = HERE / ".venv" / "bin" / "python"
    if venv.exists() and Path(sys.executable) != venv:
        os.execv(str(venv), [str(venv), str(HERE / "addtracks.py"), *sys.argv[1:]])
    sys.exit("addtracks.py needs numpy to check the sound keeps time: python3 -m pip "
             "install numpy, or --no-check")

import quicktime

RATE = 4000     # samples a second the sound is compared at: 0.25 ms apart
WINDOW = 90     # seconds of sound compared at each place
SEARCH = 20     # seconds either way an offset is looked for


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(out.stdout)


def sound(path, at, seconds):
    out = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-ss", str(max(at, 0)), "-t",
                          str(seconds), "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar",
                          str(RATE), "-f", "s16le", "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, np.int16).astype(np.float32)


def offset(reference, other, at):
    """Where `other`'s sound at `at` falls in `reference`'s, in seconds, and how
    clearly: the peak of a whitened cross-correlation, so a loud bass doesn't lead,
    over its spread. Below about 5 the two have little in common there."""
    a = sound(reference, at - SEARCH, WINDOW + 2 * SEARCH)
    b = sound(other, at, WINDOW)
    n = len(a) + len(b)
    x = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n))
    c = np.fft.irfft(x / (np.abs(x) + 1e-6), n)
    i = int(np.argmax(c))
    return (i if i < n / 2 else i - n) / RATE - SEARCH, float(c[i] / (np.std(c) + 1e-9))


def check(picture, sounds, tolerance):
    length = duration(picture)
    for path, _, title in sounds:
        if Path(path).resolve() == Path(picture).resolve():
            continue
        found = [(round(at), *offset(picture, path, at)) for at in
                 (length * 0.12, length * 0.5, length * 0.88)]
        said = ", ".join(f"{s:+.2f} s at {at // 60} min (clarity {c:.0f})" for at, s, c in found)
        print(f"  {title}: {said}")
        if any(abs(s) > tolerance for _, s, _ in found):
            sys.exit(f"{Path(path).name}'s sound doesn't keep time with {Path(picture).name}'s "
                     f"within {tolerance} s, so it isn't written. --no-check writes it anyway.")


def streams(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=index,codec_type,codec_name:stream_disposition=attached_pic",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--picture", required=True, type=Path, help="the file whose picture is kept")
    p.add_argument("--sound", nargs=3, action="append", required=True,
                   metavar=("FILE", "LANG", "TITLE"), help="a sound track: file, ISO 639-2, name")
    p.add_argument("--subtitles", nargs=3, action="append", default=[],
                   metavar=("SRT", "LANG", "TITLE"), help="a text subtitle track")
    p.add_argument("--tags-from", type=Path, help="the file whose title, show and so on to keep")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tolerance", type=float, default=0.2,
                   help="seconds a sound track may be off before it's refused (0.2)")
    p.add_argument("--no-check", action="store_true", help="don't check the sound keeps time")
    args = p.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists, and nothing is written over")

    if not args.no_check:
        print("Checking the sound tracks keep time …", flush=True)
        check(args.picture, args.sound, args.tolerance)

    inputs = [args.picture]

    def number(path):
        path = Path(path)
        if path not in inputs:
            inputs.append(path)
        return inputs.index(path)

    sounds = [(number(f), lang, title) for f, lang, title in args.sound]
    subtitles = [(number(f), lang, title) for f, lang, title in args.subtitles]
    tags = number(args.tags_from) if args.tags_from else 0

    found = streams(args.picture)
    picture = next(s for s in found if s["codec_type"] == "video"
                   and not s.get("disposition", {}).get("attached_pic"))
    covers = [s["index"] for s in found if s.get("disposition", {}).get("attached_pic")]
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    for path in inputs:
        cmd += ["-i", str(path)]
    cmd += ["-map_metadata", str(tags), "-map_chapters", "0", "-map", f"0:{picture['index']}"]
    for c in covers[:1]:
        cmd += ["-map", f"0:{c}"]
    for i, _, _ in sounds:
        cmd += ["-map", f"{i}:a:0"]
    for i, _, _ in subtitles:
        cmd += ["-map", f"{i}:0"]
    cmd += ["-c", "copy", "-c:s", "mov_text"]
    tag = {"h264": "avc1", "hevc": "hvc1"}.get(picture["codec_name"])
    if tag:
        cmd += ["-tag:v:0", tag]
    if covers:
        cmd += ["-disposition:v:1", "attached_pic"]
    for n, (_, lang, title) in enumerate(sounds):
        cmd += [f"-disposition:a:{n}", "default" if n == 0 else "0",
                f"-metadata:s:a:{n}", f"language={lang}", f"-metadata:s:a:{n}", f"title={title}",
                f"-metadata:s:a:{n}", f"handler_name={title}"]
    for n, (_, lang, title) in enumerate(subtitles):
        cmd += [f"-metadata:s:s:{n}", f"language={lang}", f"-metadata:s:s:{n}", f"title={title}",
                f"-metadata:s:s:{n}", f"handler_name={title}"]
    # Milliseconds, as quicktime.make_mp4 does, so a long chapter isn't dropped
    cmd += ["-movie_timescale", "1000", "-f", "mp4", str(args.out)]
    print(f"Writing {args.out.name} …", flush=True)
    if subprocess.run(cmd).returncode:
        sys.exit(f"ffmpeg couldn't make {args.out.name}")
    quicktime.title_tracks(args.out)
    print(f"{args.out.name}: {args.out.stat().st_size / 1e9:.2f} GB, "
          f"{len(sounds)} sound and {len(subtitles)} subtitle tracks")


if __name__ == "__main__":
    main()
