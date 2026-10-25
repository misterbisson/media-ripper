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
./ripdvd.py --sample 60 --upscale 1080   # a 60 s clip to try settings on
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
is poor. Try `--sample 60` with and without it and compare on the screen you watch on.

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
