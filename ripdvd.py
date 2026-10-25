#!/usr/bin/env python3
"""Rip a DVD to MKV with HandBrakeCLI.

- Finds the inserted disc under /Volumes and names the output after its label
- Looks the film up on Wikidata and writes "Title (Year)/Title (Year).mkv"
  with a Kodi/Jellyfin/Plex .nfo and poster beside it (see metadata.py)
- Picks the main feature (HandBrake's detection, falling back to the longest title)
- Keeps every audio track in the original language and English (incl. commentary),
  passed through untouched (AC3/DTS), with chapters
- Keeps English subtitles as soft (selectable, not burned-in) VOBSUB tracks
- Optionally upscales to 720p/1080p (plain Lanczos scaling, no AI)

Usage:
    ./ripdvd.py                 # rip whatever disc is inserted
    ./ripdvd.py --list          # show titles/tracks and exit
    ./ripdvd.py --loop --eject  # rip, eject, wait for the next disc, repeat
    ./ripdvd.py --sample --upscale 1080   # one chapter, to compare settings
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import metadata

HANDBRAKE = shutil.which("HandBrakeCLI") or "HandBrakeCLI"
PRESET_NAME = "ripdvd"
SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "nor",
               "of", "on", "or", "the", "to", "vs", "with"}
# Volume-label noise that isn't part of the movie title
LABEL_NOISE = re.compile(
    r"\b(DIS[CK] ?\d+|D\d|WS|FS|16X9|WIDESCREEN|FULLSCREEN|NTSC|PAL|R[1-6]|SE|DVD)\b", re.I)


# ---------------------------------------------------------------- disc discovery

def find_dvds():
    vols = Path("/Volumes")
    return sorted(p for p in vols.iterdir()
                  if (p / "VIDEO_TS").is_dir() or (p / "video_ts").is_dir())


def wait_for_dvd():
    print("Waiting for a DVD to be inserted…", flush=True)
    while True:
        discs = find_dvds()
        if discs:
            time.sleep(3)  # let the mount settle
            return discs[0]
        time.sleep(2)


def wait_for_removal(disc):
    while disc.exists():
        time.sleep(2)


def eject(disc):
    subprocess.run(["diskutil", "eject", str(disc)], check=False)


def name_from_label(label):
    s = re.sub(r"[_.]+", " ", label)
    words = LABEL_NOISE.sub("", s).lower().split()
    out = []
    for i, w in enumerate(words):
        if 0 < i < len(words) - 1 and w in SMALL_WORDS:
            out.append(w)
        elif re.fullmatch(r"[ivx]{2,}", w):
            out.append(w.upper())  # roman numerals: Rocky II
        else:
            out.append(w[:1].upper() + w[1:])
    return " ".join(out) or label


def safe_filename(name):
    return re.sub(r'[/:\\?*"<>|]', "-", name).strip()


# ---------------------------------------------------------------- scanning

def disc_regions(disc):
    """Regions the disc plays in. Byte 0x23 of VIDEO_TS.IFO has a bit set for each
    region it's barred from."""
    try:
        mask = (Path(disc) / "VIDEO_TS" / "VIDEO_TS.IFO").read_bytes()[0x23]
    except (OSError, IndexError):
        return []
    return [r for r in range(1, 9) if not mask & 1 << (r - 1)]


def scan(disc):
    print(f"Scanning {disc} …", flush=True)
    # libdvdcss logs the drive's region only when verbose.
    proc = subprocess.run(
        [HANDBRAKE, "-i", str(disc), "-t", "0", "--scan", "--json"],
        capture_output=True, text=True, env={**os.environ, "DVDCSS_VERBOSE": "2"})
    marker = "JSON Title Set:"
    idx = proc.stdout.find(marker)
    if idx < 0:
        drive = re.search(r"drive region\(s\) (\d)", proc.stderr)
        regions = disc_regions(disc)
        if drive and regions and int(drive[1]) not in regions:
            # libdvdcss cached blank keys for this disc; they'd spoil a later rip.
            cache = re.search(r"key cache dir: (\S+\.dvdcss/[^/\s]+)/", proc.stderr)
            if cache:
                shutil.rmtree(cache[1], ignore_errors=True)
            sys.exit(f"This disc is for region {' and '.join(map(str, regions))} and the "
                     f"drive is set to region {drive[1]}, so the drive won't read it. Use a "
                     "drive set to the disc's region, or change this one's (macOS's DVD "
                     "Player offers to; drives allow only 5 changes).")
        errors = [l for l in proc.stderr.splitlines() if "dvdcss debug" not in l]
        sys.exit("Scan failed (is libdvdcss installed?):\n" + "\n".join(errors[-30:]))
    data, _ = json.JSONDecoder().raw_decode(proc.stdout[idx + len(marker):].lstrip())
    return data


def seconds(title):
    d = title["Duration"]
    return d["Hours"] * 3600 + d["Minutes"] * 60 + d["Seconds"]


def hms(secs):
    return f"{secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}"


def pick_title(data, explicit=None):
    titles = {t["Index"]: t for t in data["TitleList"]}
    if explicit is not None:
        if explicit not in titles:
            sys.exit(f"Title {explicit} not found; choices: {sorted(titles)}")
        return titles[explicit]
    longest = max(titles.values(), key=seconds)
    main = titles.get(data.get("MainFeature"), longest)
    if main is not longest and seconds(longest) > seconds(main) * 1.1:
        print(f"Note: HandBrake's main feature is title {main['Index']} "
              f"({hms(seconds(main))}) but title {longest['Index']} is longer "
              f"({hms(seconds(longest))}). Using {main['Index']}; pass --title to override.")
    return main


def print_titles(data, chosen=None):
    main = data.get("MainFeature")
    longest = max(data["TitleList"], key=seconds)["Index"]
    print(f"\n{'':2}{'Title':>5}  {'Length':>8}  {'Chap':>4}  {'Audio':>5}  {'Subs':>4}")
    for t in sorted(data["TitleList"], key=lambda t: t["Index"]):
        tags = []
        if t["Index"] == main:
            tags.append("main feature")
        if t["Index"] == longest:
            tags.append("longest")
        is_chosen = chosen and t["Index"] == chosen["Index"]
        print(f"{'→ ' if is_chosen else '  '}{t['Index']:>5}  {hms(seconds(t)):>8}  "
              f"{len(t['ChapterList']):>4}  {len(t['AudioList']):>5}  "
              f"{len(t['SubtitleList']):>4}  {', '.join(tags)}")
    if chosen:
        print(f"\nTitle {chosen['Index']} tracks:")
        for a in chosen["AudioList"]:
            print(f"  audio {a['TrackNumber']}: {a['Description']}{commentary_tag(a)}")
        for s in chosen["SubtitleList"]:
            print(f"  subs  {s['TrackNumber']}: {s['Language']}")
    print()


# ---------------------------------------------------------------- track selection

def commentary_tag(track):
    attrs = track.get("Attributes", {})
    return " [commentary]" if attrs.get("Commentary") or attrs.get("AltCommentary") else ""


def select_audio(title, langs):
    tracks = title["AudioList"]
    if not tracks:
        return []
    # DVDs almost always list the original-language track first
    wanted = set(langs) | {tracks[0]["LanguageCode"], "und"}
    return [a for a in tracks if a["LanguageCode"] in wanted]


def audio_name(a):
    channels = {1: "Mono", 2: "Stereo", 6: "5.1", 7: "6.1", 8: "7.1"}.get(
        a["ChannelCount"], f"{a['ChannelCount']}ch")
    label = "Commentary" if commentary_tag(a) else f"{a['Language']} {channels}"
    return label.replace(",", " ")  # -A uses commas as separators


def select_subtitles(title, langs):
    subs = [s for s in title["SubtitleList"] if s["LanguageCode"] in langs]
    if not subs:  # many discs leave language unset; better than nothing
        subs = [s for s in title["SubtitleList"] if s["LanguageCode"] == "und"]
    return subs


# ---------------------------------------------------------------- encoding

def build_preset(title, args):
    """Start from HandBrake's 'H.264 MKV' preset and adjust it.

    Upscaling can only be enabled via a preset (there's no CLI flag), so we
    generate one rather than splitting settings between flags and presets.
    """
    pal = title["Geometry"]["Height"] > 480
    base = "H.264 MKV 576p25" if pal else "H.264 MKV 480p30"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "base.json"
        subprocess.run([HANDBRAKE, "--preset-export", base,
                        "--preset-export-file", str(path)],
                       capture_output=True, check=True)
        doc = json.loads(path.read_text())
    p = doc["PresetList"][0]
    p.update({
        "PresetName": PRESET_NAME,
        "Type": 1,
        "FileFormat": "av_mkv",
        "ChapterMarkers": True,
        "VideoEncoder": args.encoder,
        "VideoPreset": "slow",
        "VideoQualityType": 2,
        "VideoQualitySlider": args.quality,
        # Only touches frames that are actually interlaced (combed)
        "PictureCombDetectPreset": "default",
        "PictureDeinterlaceFilter": "decomb",
        "PictureDeinterlacePreset": "default",
    })
    if args.upscale:
        height = int(args.upscale)
        p.update({
            "PictureAllowUpscaling": True,
            "PictureUseMaximumSize": True,
            "PictureWidth": height * 16 // 9,
            "PictureHeight": height,
            # Square pixels at the target size instead of anamorphic DVD pixels
            "PicturePAR": "off",
            "VideoScaler": "zscale",
            "PictureSharpenFilter": "lapsharp",
            "PictureSharpenPreset": "light",
            "PictureSharpenTune": "film",
        })
    else:
        # Keep the DVD's native resolution and anamorphic aspect
        p.update({"PictureWidth": 720, "PictureHeight": 576 if pal else 480,
                  "PictureUseMaximumSize": True, "PicturePAR": "auto"})
    return doc


def encode(disc, title, out_path, args):
    audio = select_audio(title, args.audio_langs)
    subs = select_subtitles(title, args.sub_langs)

    print(f"Title {title['Index']} ({hms(seconds(title))}) → {out_path}")
    for a in audio:
        print(f"  audio: {a['Description']}  as '{audio_name(a)}'")
    for s in subs:
        print(f"  subs:  {s['Language']}")
    if not subs:
        print("  subs:  none matching", ",".join(args.sub_langs))

    preset_doc = build_preset(title, args)
    tmp_out = out_path.with_name(out_path.stem + ".part" + out_path.suffix)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(preset_doc, f)
        preset_file = f.name

    cmd = [HANDBRAKE, "-i", str(disc), "-t", str(title["Index"]),
           "-o", str(tmp_out), "--preset-import-file", preset_file,
           "-Z", PRESET_NAME, "--markers"]
    if audio:
        cmd += ["-a", ",".join(str(a["TrackNumber"]) for a in audio),
                "-E", ",".join("copy" for _ in audio),
                "--audio-copy-mask", "ac3,eac3,dts,dtshd,mp2,mp3,aac,flac",
                "--audio-fallback", "ca_aac",
                "-A", ",".join(audio_name(a) for a in audio)]
    if subs:
        cmd += ["-s", ",".join(str(s["TrackNumber"]) for s in subs),
                "--subtitle-burned=none"]
        foreign = audio and audio[0]["LanguageCode"] not in args.sub_langs
        # Foreign-language film: turn the first English track on by default
        cmd += ["--subtitle-default=1" if foreign else "--subtitle-default=none"]
    else:
        cmd += ["-s", "none"]
    if args.sample:
        # One whole chapter from the middle. HandBrake ignores -c when given
        # --start-at or --stop-at, and seeking by time reads the disc from the top.
        cmd += ["-c", str(len(title["ChapterList"]) // 2 + 1)]

    log_path = out_path.with_suffix(".log")
    if args.verbose:
        print(" ".join(cmd))
    try:
        with open(log_path, "w") as log:
            proc = subprocess.run(cmd, stderr=log)
    finally:
        os.unlink(preset_file)
    if proc.returncode != 0 or not tmp_out.exists():
        print(f"\nHandBrake failed (exit {proc.returncode}); see {log_path}")
        return False
    tmp_out.rename(out_path)
    log_path.unlink()
    print(f"\nDone: {out_path}")
    return True


def output_path(name, args):
    """Movies go in their own folder, Kodi/Jellyfin/Plex style; samples don't."""
    stem = safe_filename(name)
    if args.sample:
        folder, stem = args.out_dir, f"{stem} [sample {args.upscale or 'native'}]"
    else:
        folder = args.out_dir / stem
    path = folder / f"{stem}.mkv"
    n = 2
    while path.exists():
        path = folder / f"{stem} ({n}).mkv"
        n += 1
    return path


# ---------------------------------------------------------------- metadata

def choose_film(name, runtime, interactive):
    try:
        candidates = metadata.search(name, runtime)
    except Exception as e:
        print(f"Wikidata search failed: {e}")
        return None
    pick = metadata.confident(candidates, name, runtime)
    if interactive and candidates:
        print("\nWikidata matches:")
        for i, f in enumerate(candidates[:9], 1):
            print(f"  {i}. {f.summary()}  [{f.qid}]")
        print("  0. none of these")
        default = candidates.index(pick) + 1 if pick else 0
        reply = input(f"Film [{default}]: ").strip()
        choice = int(reply) if reply.isdigit() else default
        pick = candidates[choice - 1] if 0 < choice <= len(candidates) else None
    elif not candidates:
        print(f"No Wikidata film found for '{name}'.")
    elif not pick:
        print(f"No confident Wikidata match for '{name}'; skipping metadata.")
    if not pick:
        return None
    try:
        film = metadata.details(pick.qid)
    except Exception as e:
        print(f"Wikidata lookup failed: {e}")
        return None
    film.minutes = film.minutes or runtime / 60
    print(f"Matched {film.summary()}  [{film.qid}]")
    return film


def write_metadata(film, mkv):
    nfo = mkv.with_suffix(".nfo")
    metadata.write_nfo(film, nfo)
    print(f"  wrote {nfo.name}")
    for path in metadata.save_artwork(film, mkv.parent):
        print(f"  wrote {path.name}")


# ---------------------------------------------------------------- main

def rip_one(disc, args):
    data = scan(disc)
    if not data.get("TitleList"):
        print("No titles found on disc.")
        return False
    title = pick_title(data, args.title)
    print_titles(data, chosen=title)
    if args.list:
        return True

    interactive = not args.yes and sys.stdin.isatty()
    name = args.name or name_from_label(disc.name)
    if interactive:
        reply = input(f"Title [{title['Index']}]: ").strip()
        if reply:
            title = pick_title(data, int(reply))
        name = input(f"Name [{name}]: ").strip() or name

    film = None
    if not args.no_metadata and not args.sample:
        film = choose_film(name, seconds(title), interactive)
    if film:
        name = film.folder_name
        # Keep the film's original language even if the disc lists a dub first
        args = argparse.Namespace(**vars(args))
        args.audio_langs = args.audio_langs + film.languages

    out = output_path(name, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = encode(disc, title, out, args)
    if ok and film:
        write_metadata(film, out)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("disc", nargs="?", type=Path, help="disc path (default: auto-detect)")
    ap.add_argument("-o", "--out-dir", type=Path, default=Path.home() / "Movies",
                    help="output folder (default: ~/Movies)")
    ap.add_argument("-n", "--name", help="output name (default: from the disc label)")
    ap.add_argument("-t", "--title", type=int, help="title number (default: main feature)")
    ap.add_argument("-l", "--list", action="store_true", help="list titles and exit")
    ap.add_argument("-y", "--yes", action="store_true", help="don't prompt for name/title")
    ap.add_argument("--audio-langs", default="eng",
                    help="audio languages to keep besides the original (default: eng)")
    ap.add_argument("--sub-langs", default="eng", help="subtitle languages (default: eng)")
    ap.add_argument("--upscale", choices=["720", "1080"],
                    help="scale up to 720p/1080p (default: keep DVD resolution)")
    ap.add_argument("--encoder", default="x264", choices=["x264", "x265", "x265_10bit"],
                    help="video encoder (default: x264)")
    ap.add_argument("-q", "--quality", type=float, default=19,
                    help="constant quality RF, lower is better (default: 19)")
    ap.add_argument("--sample", action="store_true",
                    help="encode only one chapter from the middle, for testing")
    ap.add_argument("--no-metadata", action="store_true",
                    help="skip the Wikidata lookup, .nfo and poster")
    ap.add_argument("--eject", action="store_true", help="eject the disc when done")
    ap.add_argument("--loop", action="store_true",
                    help="after each disc, wait for the next one (implies --yes)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print the HandBrake command")
    args = ap.parse_args()
    args.audio_langs = args.audio_langs.split(",")
    args.sub_langs = args.sub_langs.split(",")
    args.out_dir = args.out_dir.expanduser()
    if args.loop:
        args.yes = True

    if not shutil.which(HANDBRAKE):
        sys.exit("HandBrakeCLI not found: brew install handbrake")

    if args.loop:
        while True:
            disc = wait_for_dvd()
            rip_one(disc, args)
            eject(disc)
            wait_for_removal(disc)
        return

    disc = args.disc
    if disc is None:
        discs = find_dvds()
        if len(discs) > 1:
            sys.exit("Several discs mounted; pass one: " + ", ".join(map(str, discs)))
        disc = discs[0] if discs else wait_for_dvd()
    ok = rip_one(disc, args)
    if ok and args.eject and not args.list:
        eject(disc)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
