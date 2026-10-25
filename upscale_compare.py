#!/usr/bin/env python3
"""Compare upscalers on short lossless clips, for speed and for looks.

Each method gets the same decoded 720x480 frames. Whatever it returns is scaled
to 1920x1080 with the same Lanczos filter and HD color matrix, so only the
upscaler differs. Speed is model time (including GPU transfers) per frame.

Setup (Apple Silicon), in upscale-test/:
    python3.12 -m venv venv && venv/bin/pip install torch torchvision spandrel coremltools
    tools/ncnn/      realesrgan-ncnn-vulkan macOS release from xinntao/Real-ESRGAN
    models/*.pth     Real-ESRGAN weights;  models/coreml/*.mlpackage  Core ML conversions

Usage:
    upscale-test/venv/bin/python upscale_compare.py upscale-test/clips/*.mkv
    upscale-test/venv/bin/python upscale_compare.py clips/*.mkv --methods lanczos,torch-general
    upscale-test/venv/bin/python upscale_compare.py --page   # rebuild index.html only
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "upscale-test"
MODELS = ROOT / "models"
NCNN = ROOT / "tools" / "ncnn"
OUT_W, OUT_H = 1920, 1080
STILL_FRAME = 60
FILM_FRAMES = 124_300  # Manufactured Landscapes: 86:25 at 23.976 fps


# ---------------------------------------------------------------- video I/O

def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], s["r_frame_rate"]


def read_frames(path):
    w, h, rate = probe(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", "scale=in_color_matrix=bt601",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), rate


SCALE_TO_HD = (f"scale={OUT_W}:{OUT_H}:flags=lanczos+accurate_rnd+full_chroma_int"
               ":out_color_matrix=bt709")
HD_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


class Writer:
    """Streams frames of any size to a 1080p H.264 file, plus one PNG still."""

    def __init__(self, mp4, still, rate):
        self.mp4, self.still, self.rate = mp4, still, rate
        self.proc = None
        self.n = 0

    def write(self, frame):
        h, w = frame.shape[:2]
        rawin = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", self.rate, "-i", "-"]
        if self.proc is None:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", *rawin, "-vf", SCALE_TO_HD + ",format=yuv420p",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "10", *HD_TAGS,
                 "-movflags", "+faststart", str(self.mp4)], stdin=subprocess.PIPE)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        if self.n == STILL_FRAME:
            subprocess.run(["ffmpeg", "-v", "error", "-y", *rawin, "-vf", SCALE_TO_HD,
                            "-frames:v", "1", str(self.still)],
                           input=frame.tobytes(), check=True)
        self.n += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------- methods
# Each takes the clip's frames and yields (upscaled_frame, model_seconds).

def lanczos(frames):
    for f in frames:
        yield f, 0.0


def torch_model(weights, blend_with=None, blend=0.5):
    def run(frames):
        import spandrel
        import torch
        dev = torch.device("mps")
        desc = spandrel.ModelLoader().load_from_file(MODELS / weights)
        if blend_with:
            # Real-ESRGAN's "denoise strength": interpolate two models' weights
            other = spandrel.ModelLoader().load_from_file(MODELS / blend_with).model.state_dict()
            sd = desc.model.state_dict()
            desc.model.load_state_dict({k: blend * v + (1 - blend) * other[k] for k, v in sd.items()})
        net = desc.model.to(dev).eval().half()
        with torch.inference_mode():
            for f in frames:
                t0 = time.perf_counter()
                x = torch.from_numpy(f).to(dev).permute(2, 0, 1)[None].half() / 255
                y = net(x).clamp_(0, 1).mul_(255).round_().byte()[0].permute(1, 2, 0).cpu().numpy()
                yield y, time.perf_counter() - t0
    return run


def ncnn_model(name, scale):
    def run(frames):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "in"), Path(tmp, "out")
            src.mkdir(), dst.mkdir()
            for i, f in enumerate(frames):
                Image.fromarray(f).save(src / f"{i:05d}.png", compress_level=1)
            t0 = time.perf_counter()
            subprocess.run([str(NCNN / "realesrgan-ncnn-vulkan"), "-i", str(src), "-o", str(dst),
                            "-n", name, "-s", str(scale), "-m", str(NCNN / "models"),
                            "-f", "png"], check=True, capture_output=True)
            per_frame = (time.perf_counter() - t0) / len(frames)
            for i in range(len(frames)):
                yield np.asarray(Image.open(dst / f"{i:05d}.png").convert("RGB")), per_frame
    return run


def tile_spans(size, tile, overlap):
    """Tile starts covering `size`, and the slice of the input each tile owns."""
    starts = sorted(set(list(range(0, size - tile, tile - overlap)) + [size - tile]))
    edges = [0] + [(a + tile + b) // 2 for a, b in zip(starts, starts[1:])] + [size]
    return [(s, edges[i], edges[i + 1]) for i, s in enumerate(starts)]


def coreml_model(package, compute="ALL", tile=256, scale=4, overlap=32):
    def run(frames):
        import coremltools as ct
        model = ct.models.MLModel(str(MODELS / "coreml" / package),
                                  compute_units=getattr(ct.ComputeUnit, compute))
        h, w = frames[0].shape[:2]
        ys, xs = tile_spans(h, tile, overlap), tile_spans(w, tile, overlap)
        for f in frames:
            t0 = time.perf_counter()
            x = (f.astype(np.float16) / 255).transpose(2, 0, 1)[None]
            batch = [{"input": x[:, :, y:y + tile, xx:xx + tile]} for y, _, _ in ys for xx, _, _ in xs]
            outs = iter(model.predict(batch))
            y_out = np.empty((h * scale, w * scale, 3), np.uint8)
            for y, ylo, yhi in ys:
                for xx, xlo, xhi in xs:
                    t = next(outs)["output"][0].transpose(1, 2, 0)
                    t = t[(ylo - y) * scale:(yhi - y) * scale, (xlo - xx) * scale:(xhi - xx) * scale]
                    y_out[ylo * scale:yhi * scale, xlo * scale:xhi * scale] = \
                        (np.clip(t, 0, 1) * 255).round().astype(np.uint8)
            yield y_out, time.perf_counter() - t0
    return run


METHODS = {
    "lanczos": ("Lanczos (no ML)", "ffmpeg", lanczos),
    "torch-general": ("Real-ESRGAN general x4v3", "PyTorch · GPU",
                      torch_model("realesr-general-x4v3.pth")),
    "torch-general-grain": ("Real-ESRGAN general x4v3, half denoise", "PyTorch · GPU",
                            torch_model("realesr-general-x4v3.pth", "realesr-general-wdn-x4v3.pth")),
    "torch-x2plus": ("Real-ESRGAN x2plus", "PyTorch · GPU", torch_model("RealESRGAN_x2plus.pth")),
    "ncnn-animevideo": ("Real-ESRGAN animevideov3 x2", "ncnn · Vulkan",
                        ncnn_model("realesr-animevideov3", 2)),
    "ncnn-x4plus": ("Real-ESRGAN x4plus (Upscayl's default)", "ncnn · Vulkan",
                    ncnn_model("realesrgan-x4plus", 4)),
    "coreml-general": ("Real-ESRGAN general x4v3", "Core ML · Neural Engine",
                       coreml_model("RealESRGAN-x4v3.mlpackage")),
    "coreml-x4plus": ("Real-ESRGAN x4plus", "Core ML · Neural Engine",
                      coreml_model("RealESRGAN-x4plus.mlpackage")),
}


# ---------------------------------------------------------------- running

def run_method(key, clips, out_dir, timings):
    label, engine, fn = METHODS[key]
    print(f"\n== {key}: {label} [{engine}]")
    for clip in clips:
        frames, rate = read_frames(clip)
        dest = out_dir / clip.stem
        dest.mkdir(parents=True, exist_ok=True)
        writer = Writer(dest / f"{key}.mp4", dest / f"{key}.png", rate)
        model_secs, t0 = 0.0, time.perf_counter()
        for frame, secs in fn(frames):
            writer.write(frame)
            model_secs += secs
        writer.close()
        timings.setdefault(key, {})[clip.stem] = {"frames": len(frames), "seconds": model_secs}
        fps = len(frames) / model_secs if model_secs else float("inf")
        print(f"  {clip.stem}: {len(frames)} frames, {fps:.2f} fps "
              f"(wall {time.perf_counter() - t0:.0f}s)")


def summarize(timings):
    rows = []
    for key, clips in timings.items():
        frames = sum(c["frames"] for c in clips.values())
        secs = sum(c["seconds"] for c in clips.values())
        fps = frames / secs if secs else None
        label, engine, _ = METHODS[key]
        rows.append({"key": key, "label": label, "engine": engine, "fps": fps,
                     "film_hours": FILM_FRAMES / fps / 3600 if fps else 0})
    return rows


def build_page(out_dir, timings):
    clips = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    methods = [k for k in METHODS if any((out_dir / c / f"{k}.mp4").exists() for c in clips)]
    data = {"clips": clips, "methods": [{"key": r["key"], "label": r["label"],
                                         "engine": r["engine"], "fps": r["fps"],
                                         "film_hours": r["film_hours"]}
                                        for r in summarize(timings) if r["key"] in methods]}
    template = (Path(__file__).resolve().parent / "upscale_compare.html").read_text()
    (out_dir / "index.html").write_text(template.replace("/*DATA*/null", json.dumps(data)))
    print(f"\nComparison page: {out_dir / 'index.html'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="*", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "results")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help="comma-separated: " + ", ".join(METHODS))
    ap.add_argument("--page", action="store_true", help="only rebuild the comparison page")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    timings_file = args.out / "timings.json"
    timings = json.loads(timings_file.read_text()) if timings_file.exists() else {}
    if not args.page:
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found: brew install ffmpeg")
        for key in args.methods.split(","):
            run_method(key, args.clips, args.out, timings)
            timings_file.write_text(json.dumps(timings, indent=2))
    for r in summarize(timings):
        if r["fps"]:
            print(f"{r['key']:22} {r['fps']:7.2f} fps   {r['film_hours']:6.1f} h for the film")
    build_page(args.out, timings)


if __name__ == "__main__":
    main()
