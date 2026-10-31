#!/usr/bin/env python3
"""Upscale a DVD rip, or any standard-definition video, to 1080p with Real-ESRGAN
on the Neural Engine, plus film grain.

Takes an MKV from ripdvd.py, or anything ffmpeg reads, and writes
"<name> - 1080p.m4v" beside it, which QuickTime plays and Plex and Jellyfin show
as a second version of the same film. The video goes:

    ffmpeg decode → Real-ESRGAN general-x4v3, 4×, Core ML on the Neural Engine
    → resize to 1080p and add luma grain, PyTorch on the GPU → x265 10-bit

and mkvmerge puts the source's frame timing back on it. quicktime.py then puts it
in an MP4 with the source's audio, subtitles, chapters, cover art and tags, in
forms QuickTime plays: audio it can't play becomes AAC, or AC3 above stereo, and
a DVD's subtitles, which are pictures, become text, read by macOS's own OCR
(vobsub.py). Interlaced video, found by looking, is deinterlaced first, and film
put on video with 3:2 pulldown goes back to its own 24 frames a second. A warm M2 MacBook
Air manages about 4 frames a second, so 9 to 10 hours for a 90-minute film. The
Mac is kept awake meanwhile. Copy-protected iTunes Store purchases can't be read,
and are refused.

Needs `brew install ffmpeg mkvtoolnix python@3.12`, and the Xcode command-line
tools (`xcode-select --install`) to build the subtitle reader. On first use it
offers to put PyTorch and coremltools (about 800 MB) in .venv/ beside it, and from
then on runs itself with that. The model weights (4 MB each) download from Real-ESRGAN's GitHub
releases, and are converted to Core ML once per frame size, in models/.

Usage:
    ./upscale.py "~/Movies/Film (2006)/Film (2006).mkv"
    ./upscale.py film.mkv --sample     # 20-30 s from the middle
    ./upscale.py film.mkv --grain 0 --denoise 0.5
    ./upscale.py --enqueue film.mkv    # upscale in the background, one at a time
    ./upscale.py --enqueue ~/"Movies/TV Shows/Show/Season 1/"*.m4v
    ./upscale.py --status              # what's queued, running and failed
"""

import argparse
import fcntl
import hashlib
import json
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from fractions import Fraction
from multiprocessing import shared_memory
from pathlib import Path

import quicktime
import vobsub

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
# The versions this was tested with. coremltools trails PyTorch, so newer ones
# may not convert the model.
PACKAGES = ["torch==2.14.1", "coremltools==9.0", "numpy==2.5.4"]
PYTHONS = ["python3.12", "python3.13", "python3.11"]


def setup(yes=False):
    """Create .venv with PyTorch and coremltools, after asking."""
    ready = VENV / "media-ripper-ready"
    if ready.exists():
        return
    python = next(filter(None, map(shutil.which, PYTHONS)), None)
    if not python:
        sys.exit("ML upscaling needs Python 3.11 to 3.13: brew install python@3.12")
    if not yes:
        if not sys.stdin.isatty():
            sys.exit(f"ML upscaling isn't set up yet: run {HERE / 'upscale.py'} --setup")
        try:
            reply = input(f"ML upscaling needs PyTorch and coremltools (about 800 MB) in "
                          f"{VENV}. Install them now? [Y/n] ")
        except EOFError:
            reply = "n"
        if reply.strip().lower() not in ("", "y", "yes"):
            sys.exit("Not installed.")
    print(f"Installing {', '.join(PACKAGES)} …", flush=True)
    shutil.rmtree(VENV, ignore_errors=True)
    subprocess.run([python, "-m", "venv", str(VENV)], check=True)
    subprocess.run([str(VENV / "bin" / "pip"), "install", "-q", "--disable-pip-version-check",
                    *PACKAGES], check=True)
    ready.touch()


# Run as a script with any python3, this sets up .venv if need be and re-runs
# itself there, before importing anything that only the venv has.
if __name__ == "__main__" and Path(sys.prefix).resolve() != VENV.resolve():
    setup(yes="--yes" in sys.argv or "-y" in sys.argv)
    venv_python = str(VENV / "bin" / "python")
    os.execv(venv_python, [venv_python, str(HERE / "upscale.py"), *sys.argv[1:]])

import numpy as np  # noqa: E402

# Quiet in the log: coremltools warns on every import that it hasn't been tested
# with this PyTorch, and PyTorch that the tracing used for conversion is deprecated
logging.getLogger("coremltools").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

MODELS = HERE / "models"
RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
WEIGHTS = {
    "realesr-general-x4v3.pth":
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    "realesr-general-wdn-x4v3.pth":
        "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
}
SAMPLE_SECS = 30


# ---------------------------------------------------------------- the model

def fetch_weights():
    MODELS.mkdir(exist_ok=True)
    for name, sha256 in WEIGHTS.items():
        path = MODELS / name
        if not path.exists():
            print(f"Downloading {name} …")
            body = urllib.request.urlopen(RELEASES + name, timeout=60).read()
            if hashlib.sha256(body).hexdigest() != sha256:
                sys.exit(f"{name}: checksum mismatch, not saving it")
            path.write_bytes(body)


def general_x4v3(denoise):
    """Real-ESRGAN's SRVGGNetCompact with general-x4v3 weights.

    `denoise` is Real-ESRGAN's -dn: x4v3 and its weak-denoise twin share an
    architecture, so their weights interpolate into an in-between model.
    """
    import torch
    from torch import nn

    class SRVGGNetCompact(nn.Module):
        def __init__(self, feat=64, convs=32, scale=4):
            super().__init__()
            self.scale = scale
            layers = [nn.Conv2d(3, feat, 3, 1, 1), nn.PReLU(feat)]
            for _ in range(convs):
                layers += [nn.Conv2d(feat, feat, 3, 1, 1), nn.PReLU(feat)]
            layers.append(nn.Conv2d(feat, 3 * scale * scale, 3, 1, 1))
            self.body = nn.ModuleList(layers)
            self.upsampler = nn.PixelShuffle(scale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + nn.functional.interpolate(
                x, scale_factor=self.scale, mode="nearest")

    load = lambda n: torch.load(MODELS / n, map_location="cpu", weights_only=True)["params"]
    strong, weak = load("realesr-general-x4v3.pth"), load("realesr-general-wdn-x4v3.pth")
    net = SRVGGNetCompact()
    net.load_state_dict({k: denoise * v + (1 - denoise) * weak[k] for k, v in strong.items()})
    return net.eval()


def model_package(denoise, w, h):
    """The model compiled for whole w×h frames, which the Neural Engine runs about
    twice as fast as tiles. Converted on first use and kept in models/."""
    package = MODELS / f"general-x4v3-dn{round(denoise * 100)}-{w}x{h}.mlpackage"
    if not package.exists():
        import coremltools as ct
        import torch
        fetch_weights()
        print(f"Converting the model for {w}×{h} frames …")
        traced = torch.jit.trace(general_x4v3(denoise), torch.rand(1, 3, h, w))
        ct.convert(traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                   inputs=[ct.TensorType(name="input", shape=(1, 3, h, w), dtype=np.float16)],
                   outputs=[ct.TensorType(name="output", dtype=np.float16)],
                   minimum_deployment_target=ct.target.macOS15).save(str(package))
    return package


def load_coreml(package):
    import coremltools as ct
    # Neural Engine only: it's as fast as letting Core ML also use the GPU, and
    # leaves the GPU free for the resize and grain
    return ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)


class Finisher:
    """Resizes the 4× output to its final size and adds grain, on the GPU.

    Grain is luma only, at the final resolution, as when finishing an upscaled
    still: `amount` is its standard deviation in 8-bit code values at midtones,
    tapering to a quarter in deep shadows and highlights, and it's softened by a
    0.8 px blur so it reads as grain rather than static.
    """

    def __init__(self, out_w, out_h, amount, size=0.8):
        import torch
        self.torch = torch
        self.dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.size, self.amount = (out_h, out_w), amount
        r = int(np.ceil(3 * size))
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * size ** 2))
        self.kernel = (k / k.sum()).to(self.dev)
        self.luma = torch.tensor([0.2126, 0.7152, 0.0722], device=self.dev).view(1, 3, 1, 1)
        self.gen = torch.Generator(device=self.dev)

    def __call__(self, x, frame_no):
        torch, F = self.torch, self.torch.nn.functional
        t = torch.from_numpy(x).to(self.dev).float()
        t = F.interpolate(t, size=self.size, mode="bicubic", antialias=True).clamp_(0, 1)
        out = t * 255
        if self.amount:
            n = torch.randn(1, 1, *self.size, device=self.dev, generator=self.gen.manual_seed(frame_no))
            k = self.kernel
            n = F.conv2d(n, k.view(1, 1, 1, -1), padding=(0, k.numel() // 2))
            n = F.conv2d(n, k.view(1, 1, -1, 1), padding=(k.numel() // 2, 0))
            luma = (t * self.luma).sum(1, keepdim=True)
            out = out + self.amount * (n / n.std()) * (4 * luma * (1 - luma)).clamp(0.25, 1)
        return out.clamp_(0, 255).round_().byte()[0].permute(1, 2, 0).contiguous().cpu().numpy()


# ---------------------------------------------------------------- video

def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "V:0", "-show_entries",
         "stream=width,height,sample_aspect_ratio,r_frame_rate,color_space:format=duration",
         "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    if not d["streams"]:
        sys.exit(f"ffprobe found no video stream in {path}")
    s = d["streams"][0]
    # Counting packets reads the whole file, but only demuxes: seconds, not minutes
    frames = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "V:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout)["streams"][0]["nb_read_packets"]
    sar = s.get("sample_aspect_ratio", "1:1")
    return {"w": s["width"], "h": s["height"],
            "sar": Fraction(*map(int, sar.split(":"))) if sar != "0:1" else Fraction(1),
            "rate": s["r_frame_rate"], "duration": float(d["format"]["duration"]),
            "frames": int(frames),
            "matrix": "bt709" if s.get("color_space") == "bt709" else "bt601"}


def output_size(info, height):
    """Square pixels at the display aspect, fitting within 16:9 at `height`."""
    dar = info["w"] * info["sar"] / info["h"]
    max_w = height * 16 // 9
    w, h = round(height * dar), height
    if w > max_w:
        w, h = max_w, round(max_w / dar)
    return w // 2 * 2, h // 2 * 2


def mkvmerge(*args):
    # Exit status 1 means warnings only
    if subprocess.run(["mkvmerge", "-q", *map(str, args)]).returncode > 1:
        sys.exit("mkvmerge failed")


def upscale(src, dst, args):
    mp4 = dst.suffix.lower() != ".mkv"
    part = dst.with_name(dst.stem + ".part" + dst.suffix)
    video, timed, times = (dst.with_name(dst.stem + s) for s in
                           (".part-video.mkv", ".part-timed.mkv", ".part-times.txt"))
    # MPEG program streams (.mpg) say nothing about their tracks but what's in the
    # packets, and ffmpeg's copy of one can lose the video: ffprobe then takes it
    # for MP3. mkvmerge reads them too, and writes each track's codec in the header.
    by_mkvmerge = src.suffix.lower() in (".mkv", ".mpg", ".mpeg")
    window = dst.with_name(dst.stem + ".part-window" + (".mkv" if by_mkvmerge else src.suffix))
    if args.sample:
        # Cut every track from the middle and treat that as the film. Both start
        # at the next keyframe, so the sample can come out a little short.
        mid = probe(src)["duration"] / 2
        if by_mkvmerge:
            mkvmerge("-o", window, "--split",
                     f"parts:{(mid - SAMPLE_SECS / 2) * 1000:.0f}ms-{(mid + SAMPLE_SECS / 2) * 1000:.0f}ms",
                     src)
        elif subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(mid - SAMPLE_SECS / 2),
                             "-t", str(SAMPLE_SECS), "-i", str(src), "-map", "0:V", "-map", "0:a?",
                             "-map", "0:s?", "-c", "copy", str(window)]).returncode:
            sys.exit(f"ffmpeg couldn't cut a sample from {src.name}")
        src = window

    info = probe(src)
    w, h = info["w"], info["h"]
    out_w, out_h = output_size(info, args.height)
    rate, total = info["rate"], info["frames"]
    kind, order = quicktime.picture(src, args.fields)
    if kind == "telecined":  # 1 frame in 5 goes
        rate, total = str(Fraction(rate) * 4 / 5), total * 4 // 5

    print(f"{src.name}: {w}×{h} → {out_w}×{out_h}, denoise {args.denoise}, "
          f"grain {args.grain}, x265 CRF {args.crf}, {total} frames")
    print(f"  picture: {quicktime.describe(kind, order)}")
    srts = quicktime.read_subtitles(src, dst) if mp4 else {}
    package = model_package(args.denoise, w, h)
    finish = Finisher(out_w, out_h, args.grain)

    # Rips are often variable frame rate: HandBrake turns film back into 23.976 fps
    # but leaves video-shot stretches at 29.97. Raw frames lose their timestamps on
    # the way through the model, so the decoder also writes each frame's timestamp
    # to a file, and mkvmerge puts them back on the encoded video. Interlaced
    # video (HandBrake has already done a rip's) is deinterlaced first, which
    # the model can't do, and the times are those of the frames that come out:
    # pulldown's are 24 a second, which the source's own clock can't hold.
    passthrough = ["-fps_mode", "passthrough",
                   "-enc_time_base", "filter" if kind == "telecined" else "demux"]
    decode_cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                  f"[0:V:0]{quicktime.field_filter(kind, order)},split[pic][time];"
                  f"[pic]scale=in_color_matrix={info['matrix']}[rgb]",
                  "-map", "[rgb]", *passthrough, "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
                  "-map", "[time]", "-c:v", "wrapped_avframe", *passthrough,
                  "-f", "mkvtimestamp_v2", str(times)]
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}", "-r", rate, "-i", "-",
         # The encoder takes its color tags from the frames, so set them here
         "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le,"
                "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
         "-c:v", "libx265", "-preset", args.preset, "-crf", str(args.crf),
         "-x265-params", "log-level=error", str(video)],
        stdin=subprocess.PIPE)

    # The Neural Engine and the GPU work on different frames at the same time, in
    # separate processes, with the 4× frames passed through shared memory
    ctx = multiprocessing.get_context("spawn")
    shm = shared_memory.SharedMemory(create=True, size=SLOTS * 3 * 16 * w * h * 4)
    slots = np.ndarray((SLOTS, 3, 4 * h, 4 * w), np.float32, buffer=shm.buf)
    free, ready = ctx.Queue(), ctx.Queue()
    for i in range(SLOTS):
        free.put(i)
    worker = ctx.Process(target=infer_worker, daemon=True,
                         args=(decode_cmd, str(package), w, h, shm.name, free, ready))
    worker.start()

    spent = {"grain": 0.0, "encode": 0.0}
    t0 = last = time.monotonic()
    tty = sys.stdout.isatty()
    done = 0
    try:
        while True:
            try:
                item = ready.get(timeout=60)
            except queue.Empty:
                if not worker.is_alive():
                    sys.exit(f"\nThe model process stopped; the partial video is {video}")
                continue
            if item[0] == "done":
                _, decode_status, worker_spent = item
                break
            slot, n = item
            t = time.monotonic()
            frame = finish(slots[slot][None], n).tobytes()
            free.put(slot)
            t2 = time.monotonic()
            encoder.stdin.write(frame)
            spent["grain"] += t2 - t
            spent["encode"] += time.monotonic() - t2
            done += 1
            now = time.monotonic()
            # A line that updates in place on a terminal; one a minute in a log
            if now - last >= (5 if tty else 60):
                fps = done / (now - t0)
                eta = max(total - done, 0) / fps if fps else 0
                line = (f"  {done}/{total} frames, {fps:.2f} fps, "
                        f"{eta / 3600:.0f}h{eta % 3600 / 60:02.0f}m left ")
                print("\r" + line if tty else line, end="" if tty else "\n", flush=True)
                last = now
    finally:
        del slots
        shm.close()
        shm.unlink()
    encoder.stdin.close()
    if decode_status or encoder.wait():
        sys.exit(f"\nffmpeg failed; the partial video is {video}")
    worker.join()

    # The new video with the source's timestamps, plus everything but video from
    # the source: audio, subtitles, chapters, tags, and track names and flags
    title = f"Real-ESRGAN {args.height}p" + (f" + grain {args.grain:g}" if args.grain else "")
    if mp4:
        mkvmerge("-o", timed, "--timestamps", f"0:{times}", video)
        quicktime.make_mp4(src, part, srts, video=timed, video_title=title,
                           hd=2 if args.height >= 1080 else 1 if args.height >= 720 else 0)
    else:
        mkvmerge("-o", part, "--timestamps", f"0:{times}", "--track-name", f"0:{title}",
                 "--default-track-flag", "0:yes", video, "--no-video", src)
    part.rename(dst)
    for f in (video, timed, times, window, *srts.values()):
        f.unlink(missing_ok=True)
    took = time.monotonic() - t0
    print(f"\nDone: {dst}  ({done} frames in {took / 60:.0f} min, {done / took:.2f} fps)")
    # Decode and model run in one process, grain and encode in the other; whichever
    # process's total is near the wall-clock time per frame is the bottleneck
    print("  ms per frame: " + ", ".join(f"{k} {v / max(done, 1) * 1000:.0f}"
                                          for k, v in {**worker_spent, **spent}.items()))


SLOTS = 4


def infer_worker(decode_cmd, package, w, h, shm_name, free, ready):
    """Decode and run the model in a child process, handing 4× frames back through
    shared memory. Core ML's predict holds Python's interpreter lock while it
    converts its 66 MB output, which stalled the grain step 6× when this was a
    thread of the same process."""
    model = load_coreml(package)
    shm = shared_memory.SharedMemory(name=shm_name)
    slots = np.ndarray((SLOTS, 3, 4 * h, 4 * w), np.float32, buffer=shm.buf)
    decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    spent = {"decode": 0.0, "model": 0.0}
    n = 0
    try:
        while True:
            t = time.monotonic()
            raw = decoder.stdout.read(frame_bytes)
            spent["decode"] += time.monotonic() - t
            if len(raw) < frame_bytes:
                break
            t = time.monotonic()
            x = np.frombuffer(raw, np.uint8).reshape(h, w, 3).transpose(2, 0, 1)[None]
            y = model.predict({"input": x.astype(np.float16) / 255})["output"]
            spent["model"] += time.monotonic() - t
            slot = free.get()
            slots[slot] = y[0]
            ready.put((slot, n))
            n += 1
        ready.put(("done", decoder.wait(), spent))
    finally:
        del slots
        shm.close()


# ---------------------------------------------------------------- the queue

# One JSON file per job, oldest first. A worker runs them one at a time, since two
# upscales would only share the Neural Engine and GPU. It holds a lock while it
# works, so whoever adds a job can start a worker without checking for one.
STATE = Path.home() / "Library" / "Application Support" / "media-ripper"
QUEUE, FAILED, LOCK = STATE / "queue", STATE / "failed", STATE / "worker.lock"
LOG = Path.home() / "Library" / "Logs" / "media-ripper" / "upscale.log"


def pending():
    return sorted(QUEUE.glob("*.json"))


def enqueue(srcs, options):
    QUEUE.mkdir(parents=True, exist_ok=True)
    queued = {json.loads(p.read_text())["input"] for p in pending()}
    for src in srcs:
        if str(src) in queued:
            print(f"Already queued: {src}")
        elif why := quicktime.problem(src):
            print(f"Not queued: {why}")
        else:
            job = {"input": str(src), "options": options, "queued": datetime.now().isoformat()}
            (QUEUE / f"{time.time_ns()}.json").write_text(json.dumps(job))
            print(f"Queued for upscaling: {src.name}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as log:
        subprocess.Popen([sys.executable, str(HERE / "upscale.py"), "--worker"],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    print(f"Progress: ./upscale.py --status, or the log at {LOG}")


def worker_running():
    STATE.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def run_queue():
    # Below rips and everything else, which get the CPU first
    os.nice(10)
    # A worker outlives the terminal that started it, and torch can't be imported
    # from a directory that's since been deleted. Jobs hold absolute paths.
    os.chdir(HERE)
    STATE.mkdir(parents=True, exist_ok=True)
    while True:
        with open(LOCK, "a") as lock:
            # Retry briefly: --status takes the lock for a moment to test it
            for _ in range(10):
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.2)
            else:
                return  # another worker has it, and will get to our job
            while jobs := pending():
                run_job(jobs[0])
        # A job added while this one was letting go of the lock
        if not pending():
            return


def run_job(path):
    job = json.loads(path.read_text())
    stamp = lambda: datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{stamp()}  Upscaling {job['input']}", flush=True)
    status = subprocess.run([sys.executable, str(HERE / "upscale.py"),
                             job["input"], *job["options"]]).returncode
    if status == 0:
        path.unlink()
    else:
        FAILED.mkdir(parents=True, exist_ok=True)
        path.rename(FAILED / path.name)
        print(f"{stamp()}  Failed (exit {status}); kept in {FAILED}", flush=True)


def show_status():
    jobs, failed = pending(), sorted(FAILED.glob("*.json"))
    running = worker_running()
    name = lambda p: Path(json.loads(p.read_text())["input"]).name
    for i, job in enumerate(jobs):
        print(f"{'running' if i == 0 and running else 'queued ':7}  {name(job)}")
    if jobs and running and LOG.exists():
        current = LOG.read_text().rpartition("  Upscaling ")[2]
        lines = [l for l in current.splitlines() if " fps, " in l]
        print(f"         {lines[-1].strip() if lines else 'starting'}")
    if jobs and not running:
        print("No worker is running. Start one with ./upscale.py --worker")
    for job in failed:
        print(f"failed   {name(job)}  (to retry, move it from {FAILED} to {QUEUE})")
    if not jobs and not failed:
        print("Nothing queued.")
    print(f"Log: {LOG}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", type=Path, nargs="*", metavar="video",
                    help="a rip from ripdvd.py, or any video ffmpeg reads")
    ap.add_argument("--enqueue", action="store_true",
                    help="add the videos to the background queue instead of upscaling now")
    ap.add_argument("--status", action="store_true", help="show the queue and exit")
    ap.add_argument("--worker", action="store_true",
                    help="run the queue in this terminal (one starts by itself on --enqueue)")
    ap.add_argument("--setup", action="store_true", help="install PyTorch and coremltools, then exit")
    ap.add_argument("-y", "--yes", action="store_true", help="install them without asking")
    ap.add_argument("-o", "--output", type=Path,
                    help='output file (default: "<input> - 1080p.m4v" beside it); '
                         'a name ending .mkv skips the MP4 step')
    ap.add_argument("--height", type=int, default=1080, help="output height (default: 1080)")
    ap.add_argument("--grain", type=float, default=3,
                    help="grain strength in 8-bit code values, 0 for none (default: 3)")
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="Real-ESRGAN denoise strength, 0 to 1 (default: 1)")
    ap.add_argument("--crf", type=float, default=21, help="x265 quality, lower is better (default: 21)")
    ap.add_argument("--preset", default="medium", help="x265 preset (default: medium)")
    ap.add_argument("--fields", choices=("auto", *quicktime.FIELDS), default="auto",
                    help="whether the video is interlaced, or film with 3:2 pulldown "
                         "(default: auto, found by looking)")
    ap.add_argument("--sample", action="store_true",
                    help=f"only {SAMPLE_SECS} s from the middle, to check settings")
    args = ap.parse_args()

    if args.setup:
        # setup() ran before the venv took over; this builds the subtitle reader
        vobsub.ocr_tool()
        print("ML upscaling is set up.")
        return
    if args.status:
        return show_status()
    if args.worker:
        return run_queue()
    if not args.inputs:
        ap.error("give a video to upscale")
    if len(args.inputs) > 1 and (args.output or not args.enqueue):
        ap.error("upscale one video at a time, or queue several with --enqueue")
    srcs = [p.expanduser().resolve() for p in args.inputs]
    for src in srcs:
        if not src.exists():
            sys.exit(f"{src}: not found")
    for tool, package in (("ffmpeg", "ffmpeg"), ("mkvmerge", "mkvtoolnix")):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found: brew install {package}")
    if args.enqueue:
        options = ["--height", str(args.height), "--grain", str(args.grain),
                   "--denoise", str(args.denoise), "--crf", str(args.crf), "--preset", args.preset,
                   "--fields", args.fields]
        options += ["--sample"] if args.sample else []
        options += ["-o", str(args.output.expanduser().resolve())] if args.output else []
        return enqueue(srcs, options)
    src = srcs[0]
    if why := quicktime.problem(src):
        sys.exit(why)
    suffix = f" - {args.height}p" + (" sample" if args.sample else "")
    dst = args.output or src.with_name(src.stem + suffix + ".m4v")
    # Keep the Mac awake (but let the display sleep) until this process exits
    subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    upscale(src, dst, args)


if __name__ == "__main__":
    main()
