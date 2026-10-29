# media-ripper

Rip a DVD into a media-library folder with one command:

```
~/Movies/Manufactured Landscapes (2006)/
  Manufactured Landscapes (2006).mkv   ← main feature, all chapters
  Manufactured Landscapes (2006).nfo   ← title, year, director, cast, plot, IMDb/TMDb/Wikidata ids
  poster.jpg                           ← when one is available (see below)
```

That layout is what Kodi, Jellyfin, Plex and Infuse expect, and the `.nfo`'s ids let
any of them (or [tesserae](https://github.com/misterbisson/tesserae)) fetch whatever it
leaves out.

## Setup (macOS)

```
brew install handbrake libdvdcss ffmpeg mkvtoolnix python@3.12
```

`ripdvd.py` needs only Python 3.10+'s standard library. The ML upscaler also needs
PyTorch and coremltools. The first time you use `--upscale`, it offers to install them
into `.venv/` (about 800 MB, a couple of minutes). It also builds a small subtitle
reader, which needs the Xcode command-line tools (`xcode-select --install`).

DVD drives are region-locked. A drive set to region 1 won't read a region 2 disc, and
nothing in software gets around it. `ripdvd.py` says so when that's the problem. Use
a second drive set to the other region, or change this one's in macOS's DVD Player.
Drives allow 5 changes, and the last one sticks.

## Use

```
./ripdvd.py                  # rip the inserted disc; asks to confirm title, name and film
./ripdvd.py --list           # show the disc's titles and tracks, then stop
./ripdvd.py -y --eject       # no questions; eject when done
./ripdvd.py --loop           # rip, eject, wait for the next disc, repeat
./ripdvd.py --upscale        # rip, then make a 1080p version in the background
./ripdvd.py --loop --upscale # feed it discs all evening; the upscales run overnight
./upscale.py --status        # what's upscaling, what's waiting, what failed
```

`./ripdvd.py --help` lists the rest (`--title`, `--name`, `--out-dir`, `--quality`,
`--encoder`, `--audio-langs`, `--sub-langs`, `--no-metadata`, `--sample`).

## What it picks

**Title.** HandBrake's main-feature detection, which copes with discs that pad
themselves with decoy titles better than "longest" does. If the longest title is
noticeably longer, it says so. `--list` shows every title with its length, chapter
count and tracks.

**Name.** From the volume label (`THE_GODFATHER_PART_III_WS` → *The Godfather Part
III*), then matched on Wikidata. When several films share a name, the one whose
runtime matches the disc's wins (the 1972 *Solaris* vs. the 2002 one). Without
`--yes` you confirm or pick from the list. With `--yes`, a doubtful match writes no
metadata instead of a wrong one.

**Audio.** Every track in the original language and in English, copied bit-for-bit
(AC3/DTS), so 5.1 stays 5.1 and commentary tracks come along. The original language
comes from Wikidata, so a foreign film whose disc lists an English dub first still
keeps its own soundtrack. Tracks are named "English 5.1", "English Stereo", or
"Commentary" when the disc flags one (most don't).

**Subtitles.** English tracks are kept as selectable tracks, never burned in. DVD
subtitles are VobSub: pictures of text, not text. MKV holds them as-is and VLC, IINA,
Infuse, Kodi, Jellyfin and Plex all show them. MP4 effectively can't, which is the
main reason the rip is MKV. The upscaler's MP4 gets them as text instead (see
below). For text subtitles from any rip, `./vobsub.py film.mkv` writes an `.srt`
for each track. When
the film's original language isn't English, the first English track is turned on by
default.

**Video.** x264 at RF 19 (`slow` preset), native DVD resolution, anamorphic aspect kept,
deinterlacing applied only to frames that are actually interlaced. PAL discs stay
576 lines. HandBrake's stock "480p30" presets would squash them to 480.

## Posters

Wikidata only carries freely licensed posters, which few in-copyright films have.
For posters and fan art from TMDb, get a free API key at themoviedb.org and

```
export TMDB_API_KEY=...
```

Without one, the `.nfo`'s ids are enough for Kodi/Jellyfin/Plex to fetch artwork
themselves.

## Upscaling

Plain scaling (Lanczos and the like) adds no detail: your TV or player does the same
at playback, for free. So `ripdvd.py` always keeps the DVD's native resolution, and
the native MKV is the master to keep. Adding detail takes an ML upscaler, which is
slow.

`--upscale` hands each finished rip to `upscale.py` and goes on to the next disc. A
background worker upscales queued films one at a time, at low priority so rips
come first. It writes `Film (2006) - 1080p.m4v` beside the rip. That plays in
QuickTime, and Jellyfin and Plex show it as a second version of the same film. Check
the look first on one chapter:
`./ripdvd.py --sample --upscale` upscales about 30 seconds of it.

```
./upscale.py --status                  # running, queued and failed, with progress
./upscale.py film.mkv                  # upscale a rip now, in this terminal
./upscale.py film.mkv --sample         # ~30 s from the middle
./upscale.py --enqueue film.mkv        # add a rip to the background queue
./upscale.py --enqueue ~/"Movies/TV Shows/Show/Season 1/"*.m4v
```

It isn't only for rips: give it any standard-definition video ffmpeg reads, such
as an old iTunes download or a TV recording, and it keeps that file's cover art,
chapters, and its title, show, season and episode. Copy-protected iTunes Store
purchases can't be read, and `--enqueue` says so and skips them. At 9 to 10 hours
a film (below), choose what's worth it: a season of a show is a week of nights.
Check the look on one first with `--sample`. The settings were tuned on DVDs, and
a small, heavily compressed download is a different kind of damaged.

The queue lives in `~/Library/Application Support/media-ripper/`, and the log is
`~/Library/Logs/media-ripper/upscale.log`. If the Mac restarts mid-film, that film
stays queued: the next `--enqueue` starts a worker again, or run
`./upscale.py --worker` to start one yourself. It begins that film from the start.

### How `upscale.py` works

The pipeline:

1. ffmpeg decodes the rip, and deinterlaces it if it's interlaced (see
   [Interlacing](#interlacing); HandBrake has already done a rip's).
2. Real-ESRGAN general-x4v3 upscales 4× on the Neural Engine. On first use it downloads
   the 4 MB weights. It converts them to Core ML for each new frame size, which varies
   with how HandBrake cropped the letterbox bars. That takes a minute or two once per
   size, and the result is kept in `models/`.
3. The GPU resizes to 1080p at the film's real aspect ratio, with square pixels, and adds
   luma grain.
4. x265 encodes 10-bit HEVC.
5. mkvmerge restores each frame's original timestamp. Rips are often variable frame
   rate: HandBrake returns film to 23.976 fps but leaves video-shot stretches at
   29.97. Forcing one rate would drop or repeat frames there.
6. [`quicktime.py`](quicktime.py) puts the new video in an MP4 that QuickTime plays,
   with everything else from the source: audio, subtitles, chapters, cover art,
   tags, and track names and flags.
   - The HEVC is tagged `hvc1`. QuickTime won't open ffmpeg's default, `hev1`.
   - AAC, AC3 and ALAC audio is copied. The rest becomes AAC, or AC3 (640 kbps)
     above stereo: DTS and MP2, which QuickTime hides without a word, MP3, which it
     drops from an MP4, and old QuickTime codecs macOS no longer decodes. A named
     track is renamed for where it came from, as "English 5.1 (from DTS)".
   - The DVD's subtitles become text. QuickTime can't show VobSub pictures, so
     [`vobsub.py`](vobsub.py) decodes each picture and macOS's own OCR (Vision) reads
     it. That runs before step 1, in a minute or two per film, so a problem shows up
     before the hours of upscaling, not after. Check a film's subtitles once: OCR
     can misread a word, and loses italics.
   - Track names get a second copy in the box QuickTime's menus read, so
     "Commentary" shows there as it does in Plex.

   The MP4's index stays at the end of the file. That's fine for local playback and
   for Plex and Jellyfin. For a rip, give `-o` a name ending `.mkv` to skip this
   step and get an MKV of the new video and the rip's other tracks instead.

Options for `./upscale.py` (`ripdvd.py --upscale` uses the defaults):

- `--grain 3`: grain strength. Use 0 for none.
- `--denoise 1`: Real-ESRGAN's denoise strength.
- `--crf 21` and `--preset medium`: x265 quality and speed.
- `--height`: output height, 1080 by default.
- `--fields`: `interlaced`, `telecined` or `progressive`, when the automatic
  choice ([Interlacing](#interlacing)) gets a video wrong.

The Mac stays awake while it runs, and progress shows frames per second and time left.

A warm M2 MacBook Air manages about 4 frames a second, so 9 to 10 hours for a
90-minute film. The finished-run summary shows where each frame's time went. On the
Air, the Neural Engine, GPU and x265 all share one fanless power budget, so a Mac with
a fan should do better. The settings come from the comparison below.

For the cleanest input, rip with `./ripdvd.py -q 0`. That's lossless x264, so the
upscaler never sees x264's own compression. It's about 6 GB per film.

Other options, for reference:

- **Upscayl** is built for still images (Real-ESRGAN models). Its default model
  (x4plus) invents texture and takes days per film on a Mac.
- **Real-ESRGAN / video2x** script the same models for video, mostly for Linux/Windows
  GPUs.
- **Topaz Video AI** (paid) uses video models that look across frames, so less
  flicker, with dedicated DVD/deinterlace models.
- **SeedVR2 and FlashVSR** are diffusion models with the most convincing detail. They
  need an NVIDIA GPU, or days per film.

### Comparing ML upscalers on a Mac

[`upscale_compare.py`](upscale_compare.py) runs short lossless clips through several
Real-ESRGAN models on each Apple Silicon backend (PyTorch on the GPU, ncnn/Vulkan as
Upscayl uses, Core ML on the Neural Engine). It builds `upscale-test/results/index.html`,
which shows two methods as a wipe over synchronized video, with frame stepping, zoom and a
speed table. Its docstring covers setup. Everything it downloads and writes stays in
`upscale-test/`, which git ignores.

Results on a fanless M2 MacBook Air (24 GB), with five 5-second clips from *Manufactured
Landscapes* (720×480 → 1920×1080):

| Model | Backend | Frames/s | 86-min film |
|---|---|---|---|
| general x4v3, whole frame (any denoise level) | Core ML | 5.6 | ~6 h |
| general x4v3, 256 px tiles | Core ML | 3.9 | ~9 h |
| general x4v3, denoise 0.5 | PyTorch | 2.8 | ~13 h |
| animevideov3 ×2 | ncnn | 2.7 | ~13 h |
| general x4v3 | PyTorch | 2.5 | ~14 h |
| x2plus | PyTorch | 0.7 | ~2 days |
| x4plus | Core ML | 0.5 | ~3 days |
| x4plus (Upscayl's default) | ncnn | 0.09 | ~16 days |

Numbers moved ±40% between runs as the Air heated up: PyTorch x4v3 ran at 3.6 fps cold
and 2.5 fps warm. Expect a full film to be at the slow end.

What the stills show:

- **general x4v3** (fast) cleans DVD noise and gives crisp edges, but it smooths skin
  towards wax and turns hazy areas painterly. Lower denoise strengths keep more of the
  source's noise. That suits skin, but in haze it brings back blotchy MPEG-2 noise.
- **animevideov3** is the smoothest: fine for animation, flat on live action.
- **x4plus** gives the most convincing faces and edges, and it also invents texture.
  A plain blue object in the e-waste shot comes out speckled.
- **Lanczos** keeps the grain and invents nothing. It's soft, but nothing in it is false.

**Denoise strength.** general-x4v3 has a weak-denoise twin (`wdn`) with the same
architecture, so the two sets of weights can be interpolated into an in-between model.
That's Real-ESRGAN's `-dn` option. The harness converts each blend to Core ML at the full
frame size, which runs about twice as fast on the Neural Engine as 256 px tiles.

**Added grain.** As with upscaled stills, adding grain at the final 1920×1080 (luma only,
softened slightly, strongest in the midtones) takes the wax off. Measured in a flat
patch, the source has about 1.35 code values of grain, and x4v3 leaves 0.67. Adding 3
looked most like film; 5 looks sandy. Full denoise plus added grain beat lower denoise
strengths, because the added grain is even where the source's noise is blotchy.

Grain costs bitrate. On the close-up face clip, grain 3 versus none:

| Encode | Grain kept (of 3.06) | Bitrate with grain vs without |
|---|---|---|
| x264 CRF 19 | 2.80 | 21.4 vs 8.3 Mbps |
| x265 CRF 22 | 2.54 | 8.9 vs 4.3 Mbps |
| SVT-AV1 CRF 30 | 1.43 | 4.2 vs 2.9 Mbps |
| SVT-AV1 + film-grain synthesis | 1.65 | 3.9 Mbps |

x265 keeps the grain for about twice the bits of a clean picture. AV1 smears half of it
away, and SVT-AV1's grain synthesis (`film-grain=8:film-grain-denoise=1`; denoising is off
by default) underestimates grain this fine, so it doesn't bring much back.

## Old videos that no longer play

macOS has dropped the codecs much early QuickTime video used. On macOS 27:

| Plays | Doesn't |
|---|---|
| H.264, HEVC, MPEG-4 Part 2, H.263, MPEG-1 and -2, Motion JPEG | Sorenson Video, Cinepak, Apple Video, DivX, and anything in AVI, WMV or FLV |
| AAC, AC3, ALAC, IMA4 | QDesign (QDM2), Microsoft ADPCM, and MP3 or DTS inside an MP4 |

[`transcode.py`](transcode.py) writes an `.m4v` that plays beside each video it's
given, at the video's own size, and leaves the original alone. It needs only ffmpeg
(and the Xcode command-line tools for DVD subtitles).

```
./transcode.py "Old Trailer.mov"
./transcode.py ~/Movies/Clips/*.avi
```

H.264 and HEVC video is copied, and anything older becomes H.264 (x264 CRF 18, set
with `--crf`), deinterlaced if need be (below). Audio, subtitles, chapters, cover art and tags go through the same
step 6 as an upscale. A file that's already `.m4v` gets `<name> - QuickTime.m4v`,
and one made before is left alone without `--force`. Copy-protected iTunes Store
purchases are skipped, with a note saying so.

To upscale an old video instead, give it to `upscale.py`: the result plays too.
Transcoding is for video too small for an upscale to help, or already HD.

### Interlacing

TV is interlaced: each frame is two fields, the odd lines and the even lines, shot
1/60 s apart. Shown as whole frames, as on a computer or a modern TV, anything
moving gets comb-like stripes along its edges. Film shown on TV was spread from 24
frames a second over 30 interlaced ones by 3:2 pulldown, which shows the stripes in
2 frames of every 5. TV recordings and old downloads seldom say which they are, so
`upscale.py` and `transcode.py` look, with ffmpeg's `idet`, at 8 stretches spread
through the video:

- Interlaced: each frame is rebuilt from its two fields (`bwdif`), at the same
  frame rate.
- Film with pulldown: the film's own frames are matched back up from their fields,
  and the frame in 5 the pulldown added is dropped, so it's 24 frames a second
  again, whole. This needs the pattern of repeated fields to show, which takes
  movement. A film that's mostly still, or so compressed that the repeats are lost,
  is deinterlaced instead, which also removes the stripes.
- Progressive: left alone, apart from any frames marked interlaced.

The log and the terminal say which it chose. If that's wrong, `--fields` says so
instead, and `transcode.py --fields interlaced` re-encodes an H.264 or HEVC video
it would otherwise copy.
