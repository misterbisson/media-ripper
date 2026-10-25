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
brew install handbrake libdvdcss
```

Python 3.10+ standard library only. Nothing to install.

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
file:

- **Upscayl** is built for still images (Real-ESRGAN models). For video you'd split
  into frames, upscale each and reassemble. That works, but it's slow, and the
  general-purpose models tend to flicker and paint over film grain.
- **Real-ESRGAN / video2x** are the same models scripted for video. Free, but mostly
  aimed at Linux/Windows GPUs, and best on animation.
- **Topaz Video AI** (paid) is the usual answer for live action on a Mac: video models
  that look across frames, so less flicker, with dedicated DVD/deinterlace models.

The native-resolution MKV is the right master to keep either way. Upscale a copy if
you decide you like the result.

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
| general x4v3 | Core ML | 3.9 | ~9 h |
| general x4v3, half denoise | PyTorch | 2.8 | ~13 h |
| animevideov3 ×2 | ncnn | 2.7 | ~13 h |
| general x4v3 | PyTorch | 2.5 | ~14 h |
| x2plus | PyTorch | 0.7 | ~2 days |
| x4plus | Core ML | 0.5 | ~3 days |
| x4plus (Upscayl's default) | ncnn | 0.09 | ~16 days |

Numbers moved ±40% between runs as the Air heated up: PyTorch x4v3 ran at 3.6 fps cold
and 2.5 fps warm. Expect a full film to be at the slow end.

What the stills show:

- **general x4v3** (fast) cleans DVD noise and gives crisp edges, but it smooths skin
  towards wax and turns hazy areas painterly. The half-denoise blend keeps a little more
  texture.
- **animevideov3** is the smoothest: fine for animation, flat on live action.
- **x4plus** gives the most convincing faces and edges, and it also invents texture.
  A plain blue object in the e-waste shot comes out speckled.
- **Lanczos** keeps the grain and invents nothing. It's soft, but nothing in it is false.
