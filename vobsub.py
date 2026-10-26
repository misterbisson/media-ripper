#!/usr/bin/env python3
"""Turn DVD subtitles (VobSub) into text, which QuickTime shows and VobSub it doesn't.

VobSub subtitles are pictures of text. ffprobe hands over each picture's bytes;
this decodes them, draws each one on black as a player would, and Apple's Vision
framework reads the text (vobsub_ocr.swift, compiled on first use with the
Xcode command-line tools).

    cues = vobsub.read_track("film.mkv", 3, "eng")   # stream 3 of the file
    vobsub.write_srt(cues, "film.eng.srt")

Or from the command line, which writes an .srt beside the file for each VobSub track:

    ./vobsub.py film.mkv
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "vobsub_ocr.swift"
BINARY = HERE / ".build" / "vobsub-ocr"
TICK = 1024 / 90000  # the unit of a subpicture's display and hide delays
# For a track without a palette in its header: background, text, outline, shade
FALLBACK_PALETTE = [(0, 0, 0), (255, 255, 255), (0, 0, 0), (128, 128, 128)] * 4


def ocr_tool():
    """The compiled OCR helper, built (again) if vobsub_ocr.swift is newer."""
    if BINARY.exists() and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime:
        return BINARY
    if not shutil.which("swiftc"):
        sys.exit("Reading DVD subtitles needs the Xcode command-line tools: xcode-select --install")
    print("Compiling the subtitle reader …", flush=True)
    BINARY.parent.mkdir(exist_ok=True)
    built = subprocess.run(["swiftc", "-O", str(SOURCE), "-o", str(BINARY)],
                           capture_output=True, text=True)
    if built.returncode:
        sys.exit(f"Couldn't compile {SOURCE.name}:\n{built.stderr}")
    return BINARY


def unhex(dump):
    """Bytes from ffprobe's hexdump ("00000000: 004a 002c …  .J.,")."""
    return bytes.fromhex("".join(line[10:50] for line in dump.splitlines()).replace(" ", ""))


def palette(extradata):
    """The 16 colors that subpictures pick their 4 from, out of the track's .idx header."""
    for line in extradata.decode("latin-1").splitlines():
        if line.startswith("palette:"):
            colors = [bytes.fromhex(c.strip().zfill(6)) for c in line[8:].split(",")]
            if len(colors) == 16:
                return [tuple(c) for c in colors]
    return FALLBACK_PALETTE


def nibbles(b0, b1):
    """A color or alpha command's four 4-bit values, for pixel values 0 to 3."""
    return [b1 & 15, b1 >> 4, b0 & 15, b0 >> 4]


def rle_rows(spu, offset, width, count):
    """`count` rows of 2-bit pixels, run-length coded in nibbles from byte `offset`."""
    n = offset * 2
    nib = lambda: spu[n // 2] >> 4 if n % 2 == 0 else spu[n // 2] & 15
    rows = []
    for _ in range(count):
        row = bytearray()
        while len(row) < width:
            # 4, 8, 12 or 16 bits: the run length, then 2 bits of color
            v = nib(); n += 1
            for limit in (0x4, 0x10, 0x40):
                if v >= limit:
                    break
                v = v << 4 | nib(); n += 1
            run, color = v >> 2, v & 3
            room = width - len(row)
            row += bytes([color]) * (min(run, room) if run else room)
        n += n % 2  # each row starts on a byte
        rows.append(row)
    return rows


def decode(spu, colors):
    """One subpicture → (show, hide, width, height, gray), the times in seconds
    after the packet's own (hide None when it doesn't say), and gray drawn as
    dark text on white: what Vision reads best. None if there's nothing to show."""
    show = hide = rect = fields = None
    index, alpha = [0, 1, 2, 3], [0, 15, 15, 15]
    pos = struct.unpack_from(">H", spu, 2)[0]
    while True:
        delay, following = struct.unpack_from(">HH", spu, pos)
        i = pos + 4
        while i < len(spu) and spu[i] != 0xFF:
            cmd = spu[i]
            if cmd in (0, 1):  # 0 is the same, but forced on even with subtitles off
                show = delay * TICK if show is None else show
            elif cmd == 2:
                hide = delay * TICK
            elif cmd == 3:
                index = nibbles(spu[i + 1], spu[i + 2])
            elif cmd == 4:
                alpha = nibbles(spu[i + 1], spu[i + 2])
            elif cmd == 5:
                b = spu[i + 1:i + 7]
                rect = (b[0] << 4 | b[1] >> 4, (b[1] & 15) << 8 | b[2],
                        b[3] << 4 | b[4] >> 4, (b[4] & 15) << 8 | b[5])
            elif cmd == 6:
                fields = struct.unpack_from(">HH", spu, i + 1)
            elif cmd == 7:  # color and contrast changes mid-picture: rare, and skipped
                i += 1 + struct.unpack_from(">H", spu, i + 1)[0]  # a size that counts itself
                continue
            else:
                break
            i += 1 + {0: 0, 1: 0, 2: 0, 3: 2, 4: 2, 5: 6, 6: 4}[cmd]
        if following == pos:
            break
        pos = following
    if rect is None or fields is None or show is None:
        return None
    w, h = rect[1] - rect[0] + 1, rect[3] - rect[2] + 1
    if w <= 0 or h <= 0:
        return None
    # The picture is interlaced: even rows, then odd rows
    top = rle_rows(spu, fields[0], w, (h + 1) // 2)
    bottom = rle_rows(spu, fields[1], w, h // 2)
    rows = [r for pair in zip(top, bottom) for r in pair] + top[len(bottom):]
    # Over black, then inverted: light text with a dark outline becomes dark
    # text on white, and the outline melts into the background
    table = bytearray(range(256))
    for v in range(4):
        r, g, b = colors[index[v]]
        table[v] = 255 - round((0.299 * r + 0.587 * g + 0.114 * b) * alpha[v] / 15)
    gray = b"".join(rows).translate(table)
    if len(set(gray)) < 2:
        return None
    return show, hide, w, h, gray


def read_track(path, stream, language="und"):
    """The text of VobSub stream `stream` (its index among all the file's streams),
    as a list of (start, end, text) in seconds."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", str(stream), "-show_data",
         "-show_entries", "stream=extradata:packet=pts_time,duration_time,data",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    colors = palette(unhex(d["streams"][0].get("extradata", "")))
    pictures = []
    for p in d["packets"]:
        try:
            picture = decode(unhex(p["data"]), colors)
        except (IndexError, struct.error):  # a damaged packet
            continue
        if picture:
            t = float(p["pts_time"])
            length = p.get("duration_time")
            pictures.append((t, float(length) if length not in (None, "N/A") else None, picture))
    if not pictures:
        return []
    feed = b"".join(struct.pack(">II", w, h) + gray for *_, (_, _, w, h, gray) in pictures)
    result = subprocess.run([str(ocr_tool()), language], input=feed, capture_output=True)
    if result.returncode:
        sys.exit(f"Subtitle OCR failed:\n{result.stderr.decode(errors='replace')}")
    texts = [json.loads(line) for line in result.stdout.decode().splitlines()]

    cues = []
    for n, ((t, length, (show, hide, *_)), text) in enumerate(zip(pictures, texts)):
        start = t + show
        # Until it says to hide, else for the packet's duration, else 5 s, and
        # never past the next one
        end = t + hide if hide else t + length if length else start + 5
        if n + 1 < len(pictures):
            end = min(end, pictures[n + 1][0] + pictures[n + 1][2][0])
        if text.strip() and end > start:
            cues.append((start, end, text.strip()))
    return cues


def write_srt(cues, path):
    def stamp(s):
        ms = round(s * 1000)
        return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
    Path(path).write_text("".join(f"{n}\n{stamp(a)} --> {stamp(b)}\n{text}\n\n"
                                  for n, (a, b, text) in enumerate(cues, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path, help="a video file with DVD subtitles, such as a rip")
    path = ap.parse_args().input.expanduser()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries",
         "stream=index,codec_name:stream_tags=language", "-of", "json", str(path)],
        capture_output=True, text=True)
    if probe.returncode:
        sys.exit(f"{path}: {probe.stderr.strip() or 'not a video ffprobe can read'}")
    tracks = [s for s in json.loads(probe.stdout)["streams"] if s["codec_name"] == "dvd_subtitle"]
    if not tracks:
        sys.exit(f"{path.name} has no DVD subtitles")
    for n, s in enumerate(tracks, 1):
        lang = s.get("tags", {}).get("language", "und")
        srt = path.with_name(f"{path.stem}.{lang}{f'.{n}' if len(tracks) > 1 else ''}.srt")
        cues = read_track(path, s["index"], lang)
        write_srt(cues, srt)
        print(f"{srt.name}: {len(cues)} lines")


if __name__ == "__main__":
    main()
