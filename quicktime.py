"""Make an MP4 that QuickTime plays, from any video ffmpeg reads.

upscale.py and transcode.py both end here. What QuickTime can't play is converted;
the rest is copied:

- Video: HEVC is tagged hvc1 (QuickTime won't open ffmpeg's default, hev1), H.264
  is copied, and anything older becomes H.264 at its own size.
- Audio: AAC, AC3, E-AC3 and ALAC are copied. DTS, MP2, MP3 (which QuickTime
  drops from an MP4 without a word) and the QuickTime-era codecs that macOS no
  longer decodes (QDesign, ADPCM) become AAC, or AC3 above stereo.
- Subtitles: text stays text, and DVD subtitles, which are pictures, become text
  through macOS's own OCR (vobsub.py).
- Chapters, cover art, track names and flags, and the title, show, season,
  episode and description an iTunes file carries, all as they were.
"""

import json
import os
import struct
import subprocess
import sys

import vobsub

# Audio QuickTime plays from an MP4
QUICKTIME_AUDIO = {"aac", "ac3", "eac3", "alac"}
TEXT_SUBTITLES = {"subrip", "ass", "ssa", "mov_text", "webvtt", "text"}
# iTunes Store purchases, encrypted with FairPlay: only Apple's apps can decode them
PROTECTED = {"drmi", "drms", "drma", "drmn", "paac", "pac3", "penc"}


def ffprobe_streams(path):
    return subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=index,codec_type,codec_name,codec_tag_string,channels,height"
         ":stream_tags=language,title:stream_disposition=default,forced,attached_pic",
         "-of", "json", str(path)], capture_output=True, text=True)


def streams(path):
    out = ffprobe_streams(path)
    if out.returncode:
        sys.exit(f"{path}: {out.stderr.strip() or 'not a video ffmpeg can read'}")
    return json.loads(out.stdout)["streams"]


def problem(path):
    """Why `path` isn't a video this can work from, or None if it is."""
    out = ffprobe_streams(path)
    if out.returncode:
        why = (out.stderr.strip().splitlines() or ["not a video ffmpeg can read"])[-1]
        return f"{path.name}: {why.removeprefix(f'{path}: ')}"
    found = json.loads(out.stdout)["streams"]
    if any(s.get("codec_tag_string") in PROTECTED for s in found):
        return (f"{path.name} is copy-protected (an iTunes Store purchase): only "
                "Apple's apps can play it")
    if not any(s["codec_type"] == "video" and not attached(s) for s in found):
        return f"{path.name} has no video"
    return None


def attached(stream):
    """A cover picture, which ffmpeg lists as a video stream"""
    return stream.get("disposition", {}).get("attached_pic")


def read_subtitles(src, dst):
    """OCR the source's DVD subtitle tracks to .srt files beside `dst`, keyed by
    each one's place among its subtitle tracks. A track that reads as nothing is
    left out. Run this first, so a problem shows up before hours of work, not after."""
    srts = {}
    subs = [s for s in streams(src) if s["codec_type"] == "subtitle"]
    for k, s in enumerate(subs):
        if s["codec_name"] != "dvd_subtitle":
            continue
        lang = s.get("tags", {}).get("language", "und")
        cues = vobsub.read_track(src, s["index"], lang)
        print(f"  subtitles: read {len(cues)} lines of track {k + 1} ({lang}) as text")
        if cues:
            srts[k] = dst.with_name(f"{dst.stem}.part-subs-{k}.srt")
            vobsub.write_srt(cues, srts[k])
    return srts


def make_mp4(src, dst, srts=None, video=None, video_title=None, hd=None, crf=18, preset="slow"):
    """Write `src` to `dst` as an MP4 QuickTime plays.

    `video` replaces the source's own picture with the first video track of that
    file (an upscale), which must share the source's timeline. `srts` are the
    source's DVD subtitles as text, from read_subtitles(). `hd` sets the iTunes
    HD flag: 1 for 720p, 2 for 1080p. Old video is encoded as x264 at `crf`."""
    srts = srts or {}
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if video:
        # ffmpeg starts each input at 0, so the new picture goes in at the
        # source's own gap between its first picture and the file's start (an
        # MPEG stream's audio can lead its video), wherever mkvmerge started it
        starts = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
             "stream=start_time:format=start_time", "-of", "json", str(src)],
            capture_output=True, text=True, check=True).stdout)
        lead = (float(starts["streams"][0].get("start_time", 0))
                - float(starts["format"].get("start_time", 0)))
        cmd += ["-itsoffset", f"{lead:.6f}", "-i", str(video)]
    cmd += ["-i", str(src)]
    s_in = 1 if video else 0  # the source's input number
    for path in srts.values():
        cmd += ["-i", str(path)]
    srt_input = {k: n for n, k in enumerate(srts, s_in + 1)}
    cmd += ["-map_metadata", str(s_in), "-map_chapters", str(s_in)]
    if hd is not None:
        cmd += ["-metadata", f"hd_video={hd}"]

    found = streams(src)
    pictures = [s for s in found if s["codec_type"] == "video" and not attached(s)]
    if video:
        cmd += ["-map", "0:v:0", "-c:v:0", "copy", "-tag:v:0", "hvc1"]
        if video_title:
            cmd += ["-metadata:s:v:0", f"title={video_title}"]
    else:
        cmd += ["-map", f"{s_in}:{pictures[0]['index']}"]
        codec = pictures[0]["codec_name"]
        if codec in ("h264", "hevc"):
            cmd += ["-c:v:0", "copy", "-tag:v:0", "avc1" if codec == "h264" else "hvc1"]
        else:
            # Deinterlaces only the frames marked interlaced (MPEG-2 from TV or a
            # DVD); even dimensions, as x264 needs, keeping the display shape
            cmd += ["-filter:v:0", "bwdif=mode=send_frame:deint=interlaced,scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v:0", "libx264", "-crf", str(crf), "-preset", preset,
                    "-pix_fmt", "yuv420p", "-fps_mode", "passthrough"]
            print(f"  video: {codec} to H.264")
    v = 1
    for st in found:
        if attached(st) and st["codec_name"] in ("mjpeg", "png"):
            cmd += ["-map", f"{s_in}:{st['index']}", f"-c:v:{v}", "copy",
                    f"-disposition:v:{v}", "attached_pic"]
            v += 1

    a = s = k = 0
    for st in found:
        kind, codec, tags = st["codec_type"], st["codec_name"], st.get("tags", {})
        if kind == "audio":
            cmd += ["-map", f"{s_in}:{st['index']}"]
            if codec in QUICKTIME_AUDIO:
                cmd += [f"-c:a:{a}", "copy"]
            else:
                # Up to stereo, AAC. Above it AC3, which holds at most 5.1, so
                # DTS-ES 6.1 folds down; 5.1 gets AC3's top rate, as DTS starts
                # out richer than a DVD's own AC3.
                channels = min(st.get("channels") or 2, 6)
                to = "ac3" if channels > 2 else "aac"
                cmd += [f"-c:a:{a}", to, f"-ac:a:{a}", str(channels),
                        f"-b:a:{a}", "640k" if channels > 2 else f"{96 * channels}k"]
                # A named track is named for where it came from too, so it's told
                # apart from a track of the same language and channels the source
                # already had, as a DVD's DTS 5.1 from its AC3 5.1
                if "title" in tags:
                    cmd += [f"-metadata:s:a:{a}", f"title={tags['title']} (from {codec.upper()})"]
                print(f"  audio: {tags.get('title', f'track {a + 1}')}, {codec.upper()} to {to.upper()}")
            a += 1
        elif kind == "subtitle":
            k += 1
            if k - 1 in srt_input:  # DVD subtitles, read as text: the .srt in their place
                disposition = st.get("disposition", {})
                flags = "+".join(f for f in ("default", "forced") if disposition.get(f))
                cmd += ["-map", f"{srt_input[k - 1]}:0", f"-disposition:s:{s}", flags or "0"]
                for key in ("language", "title"):
                    if key in tags:
                        cmd += [f"-metadata:s:s:{s}", f"{key}={tags[key]}"]
            elif codec in TEXT_SUBTITLES:
                cmd += ["-map", f"{s_in}:{st['index']}"]
            else:
                continue
            cmd += [f"-c:s:{s}", "mov_text"]
            s += 1
    # No +faststart: title_tracks() needs the index (moov) left at the end. ffmpeg
    # times the chapter track in the movie timescale, which it otherwise derives
    # from the tracks' (10 MHz with text subtitles), and a chapter over 2^31 ticks
    # (3.6 minutes at 10 MHz) is silently dropped. Milliseconds, as ffmpeg once did.
    cmd += ["-movie_timescale", "1000", "-f", "mp4", str(dst)]
    if subprocess.run(cmd).returncode:
        sys.exit(f"ffmpeg couldn't make {dst.name}")
    title_tracks(dst)


def mp4_children(data):
    """(type, contents) of each box in an MP4 box's contents"""
    at = 0
    while at + 8 <= len(data):
        size, kind = struct.unpack_from(">I4s", data, at)
        if size < 8:
            break
        yield kind, data[at + 8:at + size]
        at += size


def mp4_box(kind, contents):
    return struct.pack(">I4s", 8 + len(contents), kind) + contents


def title_tracks(path):
    """Give each track a 3GPP title (titl) beside the QuickTime name ffmpeg writes,
    since QuickTime's menus show only the former: it's what tells "Commentary"
    from "English 5.1". ffmpeg leaves the index (moov) last in the file, so it can
    grow without moving anything else."""
    def udta(contents):
        parts = dict(mp4_children(contents))
        if b"name" not in parts or b"titl" in parts:
            return contents
        # Version and flags, the language ("und", packed in 15 bits), the title
        return contents + mp4_box(b"titl", bytes(4) + b"\x55\xc4" + parts[b"name"] + b"\0")

    def trak(contents):
        return b"".join(mp4_box(k, udta(c) if k == b"udta" else c)
                        for k, c in mp4_children(contents))

    with open(path, "r+b") as f:
        size = f.seek(0, os.SEEK_END)
        at, kind = 0, None
        while at + 8 <= size:  # the top-level boxes: ftyp, free, mdat, moov
            f.seek(at)
            n, kind = struct.unpack(">I4s", f.read(8))
            if n == 1:  # mdat's 64-bit size
                n = struct.unpack(">Q", f.read(8))[0]
            if kind == b"moov" or n < 8:
                break
            at += n
        if kind != b"moov" or at + n != size:
            return  # the index isn't last after all: leave the file be
        moov = f.read(n - 8)
        f.seek(at)
        f.write(mp4_box(b"moov", b"".join(mp4_box(k, trak(c) if k == b"trak" else c)
                                          for k, c in mp4_children(moov))))
        f.truncate()
