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
brew install handbrake libdvdcss ffmpeg mkvtoolnix
```

`ripdvd.py` needs only Python 3.10+'s standard library. The optional ML upscaler,
`upscale.py`, needs PyTorch and coremltools (see *ML upscaling* below).

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
./ripdvd.py --sample --upscale 1080      # one chapter, to try settings on
```

`./ripdvd.py --help` lists the rest (`--title`, `--name`, `--out-dir`, `--quality`,
`--encoder`, `--audio-langs`, `--sub-langs`, `--no-metadata`).

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
main reason the output is MKV. If you ever need text subtitles (`.srt`), OCR them
afterwards with [Subtitle Edit](https://www.nikse.dk/subtitleedit) or `pgsrip`. When
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

`--upscale 720|1080` scales with a good conventional filter (zscale/Lanczos plus light
sharpening) and stores square pixels at the target size. Be clear about what that
buys: **it adds no detail**. Your TV or player does nearly the same scaling at playback
for free. The file just gets bigger. It's worth it mainly when the player's own scaler
is poor. Try `--sample` with and without it and compare on the screen you watch on.

Adding detail needs ML upscaling, which is a separate, slow step on the finished
rip. The native-resolution MKV is the right master to keep either way.

### ML upscaling: `upscale.py`

```
python3.12 -m venv .venv && .venv/bin/pip install torch coremltools numpy
.venv/bin/python upscale.py "~/Movies/Film (2006)/Film (2006).mkv" --sample   # ~30 s first
.venv/bin/python upscale.py "~/Movies/Film (2006)/Film (2006).mkv"            # overnight
```

This writes `Film (2006) - 1080p.mkv` beside the rip. Jellyfin and Plex show it as a
second version of the same film. The pipeline:

1. ffmpeg decodes the rip.
2. Real-ESRGAN general-x4v3 upscales 4× on the Neural Engine. On first use it downloads
   the 4 MB weights and converts them to Core ML for the rip's frame size, in `models/`.
3. The GPU resizes to 1080p at the film's real aspect ratio, with square pixels, and adds
   luma grain.
4. x265 encodes 10-bit HEVC.
5. mkvmerge puts it together with the rip's audio, subtitles, chapters and track flags,
   unchanged. It also restores each frame's original timestamp. Rips are often
   variable frame rate: HandBrake returns film to 23.976 fps but leaves video-shot
   stretches at 29.97. Forcing one rate would drop or repeat frames there.

Options:

- `--grain 3`: grain strength. Use 0 for none.
- `--denoise 1`: Real-ESRGAN's denoise strength.
- `--crf 21` and `--preset medium`: x265 quality and speed.
- `--height`: output height, 1080 by default.

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
