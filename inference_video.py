#!/usr/bin/env python
# inference_video.py

import os
import cv2
import torch
import argparse
import numpy as np
import time
import subprocess  # For ffmpeg process handling
import warnings
import threading
import shutil
import json
import math
import sys
import re
from fractions import Fraction
from queue import Queue, Empty
from torch.nn import functional as F
import traceback

# Force immediate console output (no buffering)
sys.stdout.reconfigure(line_buffering=True)

# Fix potential numpy attribute issues.
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _natural_name_key(name):
    parts = re.split(r"(\d+)", str(name or "").casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part != ""
    )


def _sorted_image_filenames(folder):
    names = []
    try:
        for item in os.listdir(folder):
            if os.path.splitext(item)[1].lower() in _IMAGE_EXTENSIONS:
                names.append(item)
    except Exception:
        return []

    names.sort(key=lambda item: (_natural_name_key(item), str(item).casefold(), str(item)))
    return names


def _prepare_image_sequence(folder):
    sequence_names = _sorted_image_filenames(folder)
    if not sequence_names:
        print(f"\nError: No supported image files found in folder: {folder}")
        sys.exit(1)

    first_frame = None
    remaining = []
    valid_count = 0

    for frame_name in sequence_names:
        frame_bgr = cv2.imread(os.path.join(folder, frame_name), cv2.IMREAD_UNCHANGED)
        if frame_bgr is None:
            print(f"\n[WARN] Skipping unreadable input frame: {frame_name}")
            continue
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        valid_count += 1
        if first_frame is None:
            first_frame = frame_rgb
        else:
            remaining.append(frame_name)

    if first_frame is None:
        print(f"\nError: No readable image frames found in folder: {folder}")
        sys.exit(1)

    return remaining, first_frame, valid_count

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
cancel_increase = False
# New variable to control ffmpeg output filtering
show_ffmpeg_output = False  # Set to True to see ffmpeg output

# =============================================================================
# CANCEL PROCESS LISTENER
# =============================================================================
# Listen for keystroke "c" to cancel the interpolation process.
try:
    import msvcrt  # Windows-only module for non-blocking keyboard input
except ImportError:
    msvcrt = None

def listen_for_cancel():
    """
    Listen for a key press to cancel the FPS increase process on Windows.
    When the user presses 'c' (or 'C'), set the cancel_increase flag.
    Non-Windows input handling is removed as it causes issues with subprocess.
    """
    global cancel_increase
    if msvcrt: # Only run if msvcrt (Windows) is available
        print("Press 'c' at any time to cancel the FPS increase process (Windows only).")
        try:
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8', errors='ignore') # Added error handling
                    if key.lower() == 'c':
                        cancel_increase = True
                        sys.stdout.write("\nCancellation requested via keypress. Interpolation will be skipped.\n")
                        sys.stdout.flush()
                        break
                time.sleep(0.1) # Reduce CPU usage
        except Exception as e:
            # This might happen if the console closes unexpectedly
            print(f"\nWarning: Error in Windows cancellation listener thread: {e}")
    else:
        # Print info message for non-windows users
        print("[INFO] Keystroke cancellation ('c') is only available on Windows.")
        # Keep thread alive without blocking stdin
        while not cancel_increase: # Check the flag periodically
             time.sleep(0.5)

# =============================================================================
# Parse command line arguments
# =============================================================================
parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=float, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to output png format frames')
parser.add_argument('--sequence-format', dest='sequence_format', type=str, default='png', choices=['png', 'jpg', 'jpeg'], help='sequence image format when --png is enabled')
parser.add_argument('--jpg-quality', dest='jpg_quality', type=int, default=95, help='JPEG quality (1-100) for JPG sequence output')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
# New argument to skip audio transfer if desired.
parser.add_argument('--no-audio', dest='no_audio', action='store_true', help='Skip audio transfer even if source video has audio')
# New argument to show ffmpeg output
parser.add_argument('--show-ffmpeg', dest='show_ffmpeg', action='store_true', help='Show ffmpeg encoding output')
parser.add_argument('--video-codec', dest='video_codec', type=str, default='libx264', help='FFmpeg video codec for output encoding (e.g. libx264, libx265, libsvtav1)')
parser.add_argument('--video-preset', dest='video_preset', type=str, default='slow', help='FFmpeg preset for video encoding')
parser.add_argument('--video-crf', dest='video_crf', type=int, default=10, help='FFmpeg CRF value for video encoding quality')
parser.add_argument('--pixel-format', dest='pixel_format', type=str, default='yuv420p', help='FFmpeg pixel format for output encoding')
parser.add_argument('--h265-tune', dest='h265_tune', type=str, default='none', help='Optional x265 tune (none, grain, psnr, ssim, fastdecode, zerolatency, animation)')
parser.add_argument('--av1-film-grain', dest='av1_film_grain', type=int, default=8, help='SVT-AV1 film grain strength (0-50)')
parser.add_argument('--av1-film-grain-denoise', dest='av1_film_grain_denoise', type=int, default=0, choices=[0, 1], help='SVT-AV1 film-grain-denoise flag (0 or 1)')
# SECourses additions (2026-11): exact-timestep frame-rate conversion, faster I/O and inference.
parser.add_argument('--pad', dest='pad', type=int, default=0, help='padding multiple at scale 1 (0 = 128, the upstream value; RIFE 4.25/4.26 work with 64, which was not faster on an RTX 5090)')
parser.add_argument('--reader', dest='reader', type=str, default='auto', choices=['auto', 'ffmpeg', 'skvideo'], help='video frame reader (auto = ffmpeg pipe when ffmpeg is available)')
parser.add_argument('--legacy-fps', dest='legacy_fps', action='store_true', help='old --fps behaviour: interpolate at fps*multi, then let ffmpeg drop/duplicate frames down to --fps')
parser.add_argument('--scene-detect', dest='scene_detect', type=str, default='content', choices=['ssim', 'content', 'hash', 'none'], help='scene-cut detector: ssim (upstream: 32x32 SSIM of the padded frames), content (SSIM + colour histogram of the real frame area, plus a luma-difference spike against the neighbouring frame pairs), hash (DCT perceptual hash), none')
parser.add_argument('--scene-threshold', dest='scene_threshold', type=float, default=None, help='cut threshold (ssim/content: SSIM below = cut, default 0.2; hash: normalized Hamming distance at or above = cut, default 0.41)')
parser.add_argument('--dedup', dest='dedup', action='store_true', help='drop duplicated (held) frames and interpolate across the gap (animation)')
parser.add_argument('--dedup-threshold', dest='dedup_threshold', type=float, default=0.5, help='mean absolute difference (0-255, 64x64 luma) below which a frame is a duplicate')
parser.add_argument('--dedup-block-threshold', dest='dedup_block_threshold', type=float, default=2.0, help='and the largest 8x8-block mean absolute difference must stay below this (protects small moving objects)')
parser.add_argument('--static-blend', dest='static_blend', action='store_true', help='upstream behaviour: replace a source frame whose 32x32 SSIM to the previous one is > 0.996 by the midpoint of its neighbours (off by default: it replaced real frames of slow shots, -0.8 dB on the robot test clip)')
parser.add_argument('--no-static-blend', dest='no_static_blend', action='store_true', help=argparse.SUPPRESS)  # legacy, now the default
parser.add_argument('--batch-timesteps', dest='batch_timesteps', type=int, default=1, help='run up to N timesteps of one frame pair in one batch')
parser.add_argument('--cuda-graph', dest='cuda_graph', nargs='?', const='on', default='auto', choices=['auto', 'on', 'off'], help='capture the RIFE forward pass in a CUDA graph (auto: on up to 1920x1152 padded frames, i.e. 1080p: 2.1x at 320x180, +32 %% at 1080p x4, +0.1 GB VRAM; at 4K the gain was small and VRAM grew by ~3.7 GB)')
parser.add_argument('--feature-cache', dest='feature_cache', action='store_true', help='reuse encoder features across timesteps/pairs (measured slower than recomputing on RTX 5090; kept for other GPUs)')
parser.add_argument('--no-feature-cache', dest='no_feature_cache', action='store_true', help=argparse.SUPPRESS)  # legacy, now the default
parser.add_argument('--timeline-start', dest='timeline_start', type=int, default=0, help='chunked runs: global index of the first source frame of this chunk; outputs stay on the whole-video frame grid')
parser.add_argument('--timeline-end', dest='timeline_end', type=int, default=-1, help='chunked runs: global source index where the next chunk takes over (exclusive); -1 = render to the last frame')
parser.add_argument('--source-rate', dest='source_rate', type=str, default='', help='exact source frame rate as a fraction (chunked runs pass the rate of the whole video)')
parser.add_argument('--engine', dest='engine', type=str, default='rife', choices=['rife', 'nvfg'], help='rife or nvfg (NVIDIA RTX Video Frame Generation, RTX 40/50 series)')
parser.add_argument('--nvfg-mode', dest='nvfg_mode', type=str, default='HIGH', choices=['LOW', 'MEDIUM', 'HIGH'], help='NVIDIA Frame Generation quality mode')

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (args.video is not None or args.img is not None), "Either --video or --img must be provided."

if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if args.img is not None:
    args.png = True
args.sequence_format = str(args.sequence_format or 'png').strip().lower()
if args.sequence_format == 'jpeg':
    args.sequence_format = 'jpg'
args.jpg_quality = max(1, min(100, int(args.jpg_quality or 95)))
args.multi = max(1, int(args.multi))
args.batch_timesteps = max(1, int(args.batch_timesteps or 1))

# Set ffmpeg output flag
show_ffmpeg_output = args.show_ffmpeg
args.video_codec = str(args.video_codec or "libx264").strip() or "libx264"
args.video_preset = str(args.video_preset or "slow").strip() or "slow"
args.pixel_format = str(args.pixel_format or "yuv420p").strip() or "yuv420p"
args.h265_tune = str(args.h265_tune or "none").strip().lower() or "none"
if args.h265_tune not in {"none", "grain", "psnr", "ssim", "fastdecode", "zerolatency", "animation"}:
    args.h265_tune = "none"
try:
    args.video_crf = int(args.video_crf)
except Exception:
    args.video_crf = 10
args.video_crf = max(0, min(63, int(args.video_crf)))
try:
    args.av1_film_grain = int(args.av1_film_grain)
except Exception:
    args.av1_film_grain = 8
args.av1_film_grain = max(0, min(50, int(args.av1_film_grain)))
args.av1_film_grain_denoise = bool(int(args.av1_film_grain_denoise))


def apply_codec_tuning(ffmpeg_cmd):
    codec_norm = str(args.video_codec or "").strip().lower()
    if codec_norm == "libx265" and args.h265_tune != "none":
        ffmpeg_cmd.extend(["-tune", args.h265_tune])
    if codec_norm in {"libsvtav1", "libaom-av1"}:
        ffmpeg_cmd.extend(
            [
                "-svtav1-params",
                f"film-grain={int(args.av1_film_grain)}:film-grain-denoise={1 if args.av1_film_grain_denoise else 0}",
            ]
        )


def snap_rate(value):
    """Exact rational for a frame rate typed as a float (60 -> 60/1, 59.94 -> 60000/1001)."""
    value = float(value)
    if abs(value - round(value)) < 1e-6:
        return Fraction(int(round(value)), 1)
    ntsc = value * 1.001
    if abs(ntsc - round(ntsc)) < 2e-3:
        return Fraction(int(round(ntsc)) * 1000, 1001)
    return Fraction(value).limit_denominator(1000)


def rate_text(rate):
    return f"{rate.numerator}/{rate.denominator}"


def probe_video(path):
    """Width, height (after rotation) and the frame rate as an exact fraction, via ffprobe."""
    info = {}
    if shutil.which("ffprobe") is None:
        return info
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate,avg_frame_rate,color_space,color_primaries,color_transfer,color_range"
             ":stream_side_data=rotation:stream_tags=rotate",
             "-of", "json", path],
            capture_output=True, text=True, creationflags=_NO_WINDOW,
        ).stdout
        stream = (json.loads(out or "{}").get("streams") or [{}])[0]
    except Exception:
        return info
    try:
        w, h = int(stream.get("width") or 0), int(stream.get("height") or 0)
        rotation = 0
        for side in stream.get("side_data_list") or []:
            if "rotation" in side:
                rotation = int(float(side["rotation"]))
        if not rotation:
            rotation = int(float((stream.get("tags") or {}).get("rotate") or 0))
        if abs(rotation) % 180 == 90:
            w, h = h, w
        if w > 0 and h > 0:
            info["size"] = (w, h)
    except Exception:
        pass

    def _frac(text):
        try:
            value = Fraction(str(text))
            return value if value > 0 else None
        except Exception:
            return None

    for key in ("color_space", "color_primaries", "color_transfer", "color_range"):
        info[key] = str(stream.get(key) or "").lower()
    r_rate, avg_rate = _frac(stream.get("r_frame_rate")), _frac(stream.get("avg_frame_rate"))
    if r_rate and avg_rate and abs(r_rate - avg_rate) <= r_rate * Fraction(2, 1000):
        info["rate"] = r_rate
    elif avg_rate:
        info["rate"] = avg_rate
    elif r_rate:
        info["rate"] = r_rate
    return info


_SWS_MATRIX = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt2020nc": "bt2020",
               "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}


def color_encode_args(info):
    """Encode the RGB frames with the source's YUV matrix and tag the output like the source.

    Frames are decoded with the source's own matrix (ffmpeg honours the tags); encoding them with
    ffmpeg's default BT.601 matrix and no tags made players show BT.709 videos with shifted colours.
    Untagged sources keep ffmpeg's defaults."""
    space = (info or {}).get("color_space", "")
    matrix = _SWS_MATRIX.get(space)
    if not matrix:
        return []
    rng = "pc" if (info or {}).get("color_range") in {"pc", "jpeg", "full"} else "tv"
    # ffmpeg 8/9 encoders take colorimetry from the frames, so label them with setparams.
    params = [f"colorspace={space}", f"range={rng}"]
    prim, trc = info.get("color_primaries", ""), info.get("color_transfer", "")
    if prim and prim not in {"unknown", "reserved"}:
        params.append(f"color_primaries={prim}")
    if trc and trc not in {"unknown", "reserved"}:
        params.append(f"color_trc={trc}")
    return ["-vf", f"scale=out_color_matrix={matrix}:out_range={rng},setparams={':'.join(params)}"]


def ffmpeg_frame_reader(path, width, height):
    """Decoded RGB24 frames straight from an ffmpeg pipe (writable numpy arrays)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=_NO_WINDOW, bufsize=frame_bytes * 2)
    try:
        while True:
            buf = bytearray(frame_bytes)
            view = memoryview(buf)
            got = 0
            while got < frame_bytes:
                n = proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()


# =============================================================================
# Setup device and model
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
model = None
vfg = None
feature_cache = None
if args.engine == 'nvfg':
    print("Initializing NVIDIA RTX Video Frame Generation...", end="")
    sys.stdout.flush()
    if not torch.cuda.is_available():
        print("\nError: NVIDIA Frame Generation needs a CUDA GPU.")
        sys.exit(1)
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 9):
        print(f"\nError: NVIDIA Frame Generation needs an RTX 40 or 50 series GPU (Ada or newer); "
              f"'{torch.cuda.get_device_name(device)}' is compute capability {major}.{minor}. Choose a RIFE model instead.")
        sys.exit(1)
    try:
        import nvvfx
        if not hasattr(nvvfx, "VideoFrameGeneration"):
            raise ImportError(f"nvidia-vfx {getattr(nvvfx, '__version__', '?')} has no VideoFrameGeneration (needs 0.2.0.0 or newer)")
    except Exception as exc:
        print(f"\nError: NVIDIA Frame Generation runtime could not be loaded: {exc}")
        sys.exit(1)
    args.fp16 = False
    print(" done")
else:
    print("Initializing RIFE model...", end="")
    sys.stdout.flush()  # Force immediate output
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if args.fp16:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

    from train_log.RIFE_HDv3 import Model

    model_load_start = time.time()
    model = Model()
    if not hasattr(model, 'version'):
        model.version = 0
    model.load_model(args.modelDir, -1)
    model_load_time = time.time() - model_load_start
    print(f" done in {model_load_time:.2f}s")
    sys.stdout.flush()  # Force immediate output
    model.eval()
    model.device()


class CachedEncoder(torch.nn.Module):
    """Reuse the encoder features of a frame across timesteps and across consecutive pairs.

    IFNet encodes both frames of a pair on every call, so every frame is encoded 2 * (multi - 1)
    times. A frame's features depend only on the frame, so identical inputs (checked with
    torch.equal) return the stored result: the output is unchanged. Batched rows that the caller
    marks identical are encoded once and broadcast.
    """

    def __init__(self, encoder, slots=2):
        super().__init__()
        self.encoder = encoder
        self.slots = slots
        self.entries = []
        self.enabled = True
        self.rows_identical = False
        self.hits = 0
        self.misses = 0

    def _lookup(self, x):
        for ref, out in self.entries:
            if ref.shape == x.shape and ref.dtype == x.dtype and ref.device == x.device and torch.equal(ref, x):
                return out
        return None

    def forward(self, x, *extra, **kw):
        if extra or kw or not self.enabled:
            return self.encoder(x, *extra, **kw)
        batch = int(x.shape[0])
        if batch > 1 and self.rows_identical:
            return self.forward(x[:1]).expand(batch, *([-1] * (x.dim() - 1)))
        out = self._lookup(x)
        if out is not None:
            self.hits += 1
            return out
        self.misses += 1
        out = self.encoder(x)
        # Keep a 3-channel copy, not the view: the view would keep the whole 6-channel pair alive.
        self.entries.append((x.clone(), out))
        if len(self.entries) > self.slots:
            self.entries.pop(0)
        return out


if model is not None and args.feature_cache and isinstance(getattr(model.flownet, "encode", None), torch.nn.Module):
    feature_cache = CachedEncoder(model.flownet.encode)
    model.flownet.encode = feature_cache

supports_timestep = bool(vfg is not None or args.engine == 'nvfg' or (model is not None and model.version >= 3.9))

# =============================================================================
# Start cancellation listener in a separate thread
# =============================================================================
cancel_thread = threading.Thread(target=listen_for_cancel, daemon=True)
cancel_thread.start()

# =============================================================================
# Prepare video/image input/output
# =============================================================================
src_rate = None
if args.video is not None:
    print("\nReading video metadata...", end="")
    sys.stdout.flush()  # Force immediate output
    video_open_start = time.time()
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    probe = probe_video(args.video)
    src_rate = probe.get("rate") or (snap_rate(fps) if fps and fps > 0 else Fraction(30, 1))
    if args.source_rate:
        try:
            src_rate = Fraction(str(args.source_rate))
        except (ValueError, ZeroDivisionError):
            print(f"[WARN] Ignoring invalid --source-rate {args.source_rate!r}")
    fps = float(src_rate)
    video_open_time = time.time() - video_open_start
    print(f" done in {video_open_time:.2f}s")

    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
        print(f"[INFO] Input: {int(tot_frame)} frames at {fps:.3f}fps | Output: {args.fps:.3f}fps ({args.multi}x)")
    else:
        fpsNotAssigned = False
        print(f"[INFO] Input: {int(tot_frame)} frames at {fps:.3f}fps | Output: {args.fps:.3f}fps (user-specified)")

    print("Setting up video reader...", end="")
    reader_kind = args.reader
    if reader_kind == "auto":
        reader_kind = "ffmpeg" if (shutil.which("ffmpeg") and probe.get("size")) else "skvideo"
    if reader_kind == "ffmpeg" and not probe.get("size"):
        reader_kind = "skvideo"
    if reader_kind == "ffmpeg":
        videogen = ffmpeg_frame_reader(args.video, probe["size"][0], probe["size"][1])
    else:
        import skvideo.io
        videogen = skvideo.io.vreader(args.video)

    first_frame_start = time.time()
    try:
        lastframe = next(videogen)
        first_frame_time = time.time() - first_frame_start
        print(f" done in {first_frame_time:.2f}s ({reader_kind})")
    except Exception as e:
        print(f"\nError: Failed to load first frame: {str(e)}")
        sys.exit(1)

    video_path_wo_ext, ext = os.path.splitext(args.video)

    if not args.png and fpsNotAssigned and not args.no_audio:
        print("[INFO] Audio will be transferred from source to output")
    else:
        print("[INFO] Audio transfer will be skipped")
else:
    videogen, lastframe, tot_frame = _prepare_image_sequence(args.img)
    fpsNotAssigned = True
    fps = 1.0
    src_rate = Fraction(1, 1)
    args.fps = float(args.multi)

h, w, _ = lastframe.shape
vid_out_name = None
sequence_output_dir = 'vid_out'
ffmpeg_process = None  # This will be used for video saving

# =============================================================================
# Output timeline
# =============================================================================
# Output frame k shows source time k * src_rate / out_rate (in source-frame units). Exact mode
# renders every output frame at its own timestep, so a 24 -> 60 fps conversion needs no
# intermediate 96/120 fps file, no second lossy encode and no ffmpeg frame dropping (judder).
exact_mode = bool(supports_timestep and not fpsNotAssigned and not args.legacy_fps and args.video is not None)
if exact_mode:
    out_rate = snap_rate(args.fps)
    if out_rate == src_rate * args.multi:
        print(f"[INFO] Output {rate_text(out_rate)} fps = {args.multi}x the source rate.")
    else:
        print(f"[INFO] Exact frame-rate conversion: {rate_text(src_rate)} -> {rate_text(out_rate)} fps, "
              f"each output frame interpolated at its own timestep.")
else:
    if not fpsNotAssigned and args.video is not None and not supports_timestep and args.fps > fps * args.multi + 1e-6:
        needed = 2
        while fps * needed < args.fps - 1e-6 and needed < 64:
            needed *= 2
        print(f"[INFO] This model cannot interpolate arbitrary timesteps; using x{needed} then resampling to {args.fps:g} fps.")
        args.multi = needed
    out_rate = src_rate * args.multi
step_ratio = src_rate / out_rate  # source frames advanced per output frame

# Chunked runs render only their own part of the whole-video output grid: output k shows global
# source time k * step_ratio, and this chunk emits the k whose time lies in
# [timeline_start, timeline_end). The chunk's first frame (the previous chunk's last one) is
# context for the first pair, so the merged chunks equal one whole-video run, with no overlap
# to trim and no grid shift at chunk boundaries.
timeline_start = Fraction(0)
timeline_end = None
if (args.timeline_start > 0 or args.timeline_end >= 0) and args.video is not None:
    if supports_timestep:
        timeline_start = Fraction(max(0, int(args.timeline_start)))
        timeline_end = Fraction(int(args.timeline_end)) if args.timeline_end >= 0 else None
        print(f"[INFO] Timeline segment: global source frames from {args.timeline_start}"
              + (f" up to {args.timeline_end} (exclusive)" if timeline_end is not None else " to the end"))
    else:
        print("[WARN] --timeline-start/--timeline-end need a model with timestep input; ignored.")

# =============================================================================
# Define write and read buffer functions for multithreaded I/O
# =============================================================================
def clear_write_buffer(user_args, write_buffer, ffmpeg_proc=None):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            frame_ext = '.jpg' if user_args.sequence_format == 'jpg' else '.png'
            frame_path = os.path.join(sequence_output_dir, '{:0>7d}{}'.format(cnt, frame_ext))
            frame_bgr = item[:, :, ::-1]
            if user_args.sequence_format == 'jpg':
                cv2.imwrite(frame_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(user_args.jpg_quality)])
            else:
                cv2.imwrite(frame_path, frame_bgr)
            cnt += 1
        else:
            # RGB frames go to an rgb24 pipe as they are.
            try:
                data = item.data if item.flags["C_CONTIGUOUS"] else item.tobytes()
                ffmpeg_proc.stdin.write(data)
            except (OSError, BrokenPipeError) as e:
                print(f"\nError writing frame to ffmpeg: {e}")
                print("FFmpeg process may have terminated unexpectedly.")
                break

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        buffer_count = 0
        loading_start_time = time.time()

        # Display initial loading message
        print("Pre-loading frames into buffer...", end="")
        sys.stdout.flush()

        for frame in videogen:
            if user_args.img is not None:
                frame_bgr = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)
                if frame_bgr is None:
                    print(f"\n[WARN] Skipping unreadable input frame: {frame}")
                    continue
                frame = frame_bgr[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]

            read_buffer.put(frame)
            buffer_count += 1

            # Only update status occasionally to avoid console spam
            if buffer_count % 500 == 0:
                elapsed = time.time() - loading_start_time
                fps_now = buffer_count / elapsed if elapsed > 0 else 0
                print(f"\rPre-loading: {buffer_count}/{int(tot_frame)} frames ({buffer_count/max(1, tot_frame)*100:.1f}%) at {fps_now:.1f} fps", end="")
                sys.stdout.flush()
    except Exception as e:
        print(f"\rError loading frames: {str(e)}")

    # Final loading status
    total_time = time.time() - loading_start_time
    buffer_fps = buffer_count/total_time if total_time > 0 else 0
    print(f"\rPre-loading: Complete - {buffer_count} frames loaded in {total_time:.2f}s ({buffer_fps:.1f} fps)")
    sys.stdout.flush()  # Force immediate output
    read_buffer.put(None)

# =============================================================================
# Define interpolation function
# =============================================================================
def make_inference(I0, I1, n, current_frame):
    """Legacy recursive interpolation for models older than 3.9 (no arbitrary timestep)."""
    global cancel_increase
    middle = model.inference(I0, I1, args.scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, n=n // 2, current_frame=current_frame)
    second_half = make_inference(middle, I1, n=n // 2, current_frame=current_frame)
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def to_frames(batch):
    """(B,3,Hp,Wp) float in [0,1] -> list of (h,w,3) uint8 RGB, rounded (not truncated)."""
    t = batch[:, :, :h, :w].float().mul(255.0).round().clamp(0, 255).to(torch.uint8)
    t = t.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return [t[i] for i in range(t.shape[0])]


class GraphedInference:
    """model.inference captured in a CUDA graph per batch size (static shapes)."""

    def __init__(self):
        self.graphs = {}

    def __call__(self, I0, I1, timesteps):
        b = len(timesteps)
        entry = self.graphs.get(b)
        if entry is None:
            s0 = I0.expand(b, -1, -1, -1).clone()
            s1 = I1.expand(b, -1, -1, -1).clone()
            st = torch.tensor(timesteps, dtype=I0.dtype, device=I0.device).view(b, 1, 1, 1)
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    model.inference(s0, s1, st, args.scale)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = model.inference(s0, s1, st, args.scale)
            entry = (graph, s0, s1, st, out)
            self.graphs[b] = entry
        graph, s0, s1, st, out = entry
        s0.copy_(I0.expand(b, -1, -1, -1))
        s1.copy_(I1.expand(b, -1, -1, -1))
        st.copy_(torch.tensor(timesteps, dtype=st.dtype, device=st.device).view(b, 1, 1, 1))
        graph.replay()
        return out


graphed = None  # decided after padding (needs the frame size)


def rife_frames(I0, I1, timesteps):
    """Interpolated frames (uint8 RGB) for the given timesteps of one pair."""
    frames_out = []
    step = args.batch_timesteps if (graphed is not None or args.batch_timesteps > 1) else 1
    for i in range(0, len(timesteps), step):
        chunk = list(timesteps[i:i + step])
        n = len(chunk)
        if graphed is not None:
            padded_chunk = chunk + [chunk[-1]] * (step - n)
            out = graphed(I0, I1, padded_chunk)[:n]
        elif n == 1:
            out = model.inference(I0, I1, chunk[0], args.scale)
        else:
            st = torch.tensor(chunk, dtype=I0.dtype, device=I0.device).view(n, 1, 1, 1)
            if feature_cache is not None:
                feature_cache.rows_identical = True
            try:
                out = model.inference(I0.expand(n, -1, -1, -1), I1.expand(n, -1, -1, -1), st, args.scale)
            finally:
                if feature_cache is not None:
                    feature_cache.rows_identical = False
        frames_out.extend(to_frames(out))
    return frames_out


def nvfg_frames(I0, I1, timesteps):
    frames_out = []
    uniform = (not args.dedup) and step_ratio == Fraction(1, args.multi) and 2 <= args.multi <= 8
    for t in timesteps:
        if uniform:
            idx = int(round(t * args.multi))
            res = vfg.run(I0[0], I1[0], frame_index=idx)
        else:
            res = vfg.run_at_timestep(I0[0], I1[0], float(t))
        frames_out.extend(to_frames(torch.from_dlpack(res.image).clone().unsqueeze(0)))
    return frames_out


def interpolate_frames(I0, I1, timesteps):
    if not timesteps:
        return []
    if vfg is not None:
        return nvfg_frames(I0, I1, timesteps)
    return rife_frames(I0, I1, timesteps)

# =============================================================================
# Prepare output video settings
# =============================================================================
if args.png:
    if args.output is not None:
        sequence_output_dir = args.output
    if not os.path.exists(sequence_output_dir):
        os.makedirs(sequence_output_dir, exist_ok=True)
    if args.sequence_format == 'jpg':
        print(f"[INFO] Output format: JPG sequence in '{sequence_output_dir}' directory (quality={args.jpg_quality})")
    else:
        print(f"[INFO] Output format: PNG sequence in '{sequence_output_dir}' directory")
else:
    print("Setting up FFmpeg encoder...", end="")
    sys.stdout.flush()
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{:.3f}fps.{}'.format(video_path_wo_ext, args.multi, args.fps, args.ext)

    # Define temporary high-FPS filename
    vid_out_name_temp_highfps = os.path.splitext(vid_out_name)[0] + "_temp_highfps" + os.path.splitext(vid_out_name)[1]

    # Exact mode writes the target rate directly; the legacy path writes fps * multi first.
    actual_high_fps = float(out_rate)
    print(f" Initial encoding to: {os.path.basename(vid_out_name_temp_highfps)} [{w}x{h}@{actual_high_fps:.3f}fps]")
    print(
        f"[INFO] Encoder settings: codec={args.video_codec}, preset={args.video_preset}, "
        f"crf={args.video_crf}, pix_fmt={args.pixel_format}, h265_tune={args.h265_tune}, "
        f"av1_film_grain={args.av1_film_grain}, av1_film_grain_denoise={int(args.av1_film_grain_denoise)}"
    )

    # First ffmpeg command: Encode raw frames to high-FPS intermediate file
    ffmpeg_cmd_1 = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{w}x{h}',
        '-r', rate_text(out_rate),  # exact rational output rate for the input pipe
        '-i', '-',  # input from pipe
        '-an', # No audio for this intermediate step
        *(color_encode_args(probe) if (args.video is not None and not args.montage) else []),
        '-c:v', args.video_codec,
        '-preset', args.video_preset,
        '-crf', str(args.video_crf),
        '-pix_fmt', args.pixel_format,
    ]
    apply_codec_tuning(ffmpeg_cmd_1)
    ffmpeg_cmd_1.append(vid_out_name_temp_highfps) # Output to temporary high-FPS file
    # Use DEVNULL for stdout/stderr if not showing FFmpeg output
    ffmpeg_stdout = None if show_ffmpeg_output else subprocess.DEVNULL
    ffmpeg_stderr = None if show_ffmpeg_output else subprocess.DEVNULL
    # Store the first process
    ffmpeg_process_1 = subprocess.Popen(ffmpeg_cmd_1, stdin=subprocess.PIPE,
                                     stdout=ffmpeg_stdout, stderr=ffmpeg_stderr,
                                     creationflags=_NO_WINDOW)

    # Pass the correct process to the writer thread
    ffmpeg_process = ffmpeg_process_1 # The writer thread writes to this first process

# =============================================================================
# Audio transfer function (unchanged)
# =============================================================================
def transferAudio(sourceVideo, targetVideo):
    import shutil
    import os
    temp_dir = "./temp"
    tempAudioFileName = os.path.join(temp_dir, "audio.mkv")

    audio_check_cmd = f'ffprobe -loglevel error -select_streams a -show_entries stream=index -of csv=p=0 "{sourceVideo}"'
    audio_streams = os.popen(audio_check_cmd).read().strip()
    if not audio_streams:
        print("No audio stream found in source video. Skipping audio transfer.")
        return

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Hide ffmpeg output unless show_ffmpeg_output is True
    redirect = "" if show_ffmpeg_output else " > nul 2>&1" if os.name == 'nt' else " > /dev/null 2>&1"

    print("Extracting audio from source video...", end="")
    sys.stdout.flush()
    os.system(f'ffmpeg -y -i "{sourceVideo}" -map 0:a? -c:a copy -vn "{tempAudioFileName}"{redirect}')
    if not os.path.exists(tempAudioFileName) or os.path.getsize(tempAudioFileName) == 0:
        print(" failed!")
        return
    print(" done")

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)

    print("Merging audio with new video...", end="")
    sys.stdout.flush()
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy -shortest "{targetVideo}"{redirect}')

    if os.path.getsize(targetVideo) == 0:
        print(" failed with copy, trying with transcode...")
        sys.stdout.flush()
        tempAudioFileName_aac = os.path.join(temp_dir, "audio.m4a")
        os.system(f'ffmpeg -y -i "{sourceVideo}" -map 0:a? -c:a aac -b:a 160k -vn "{tempAudioFileName_aac}"{redirect}')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName_aac}" -c copy -shortest "{targetVideo}"{redirect}')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print(" failed! Output will have no audio.")
        else:
            print(" done with AAC audio")
            os.remove(targetNoAudio)
    else:
        print(" done")
        os.remove(targetNoAudio)

    shutil.rmtree(temp_dir)

# =============================================================================
# Calculate padding to ensure dimensions are multiples of a base value.
# =============================================================================
if args.montage:
    left = w // 4
    w = w // 2
if vfg is None and args.engine == 'nvfg':
    tmp = 2  # NVIDIA Frame Generation takes the frame as it is
else:
    pad_base = int(args.pad) if int(args.pad or 0) > 0 else 128
    tmp = max(pad_base, int(pad_base / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)
if args.engine == 'nvfg':
    padding = (0, 0, 0, 0)
    vfg = nvvfx.VideoFrameGeneration(
        mode=nvvfx.VideoFrameGeneration.Mode[args.nvfg_mode],
        automatic_shot_change_detection_enabled=True,
        device=torch.cuda.current_device(),
    )
    vfg.input_width = w
    vfg.input_height = h
    vfg.frame_multiplier = max(2, min(8, args.multi)) if args.multi >= 2 else 2
    vfg.load()
    print(f"[INFO] NVIDIA Frame Generation ready: mode={args.nvfg_mode}, {w}x{h}")
else:
    print(f"[INFO] Padding {w}x{h} to {pw}x{ph} (multiple of {tmp})")
    use_graph = args.cuda_graph == 'on' or (args.cuda_graph == 'auto' and pw * ph <= 1920 * 1152)
    if model is not None and use_graph:
        if not torch.cuda.is_available() or not supports_timestep:
            if args.cuda_graph == 'on':
                print("[INFO] CUDA graph capture needs CUDA and a model with timestep input; disabled.")
        else:
            graphed = GraphedInference()
            if feature_cache is not None:
                feature_cache.enabled = False  # equality checks would synchronize inside the graph
            print("[INFO] CUDA graph capture: on")

# =============================================================================
# Setup buffers and start processing threads
# =============================================================================
print("\nSetting up processing buffers...", end="")
sys.stdout.flush()
if args.montage:
    lastframe = lastframe[:, left: left + w]

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)

print(" done")
print("Starting reader and writer threads...")
reader_thread = threading.Thread(target=build_read_buffer, args=(args, read_buffer, videogen), daemon=True)
writer_thread = threading.Thread(target=clear_write_buffer, args=(args, write_buffer, None if args.png else ffmpeg_process), daemon=True)
reader_thread.start()
writer_thread.start()

print("Preparing first frame...")
def to_tensor(frame_np):
    return torch.from_numpy(np.ascontiguousarray(np.transpose(frame_np, (2, 0, 1)))).to(device, non_blocking=True).unsqueeze(0).float() / 255.

def pad_image(img):
    if args.fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)
I1 = pad_image(to_tensor(lastframe))
temp = None  # Temporarily holds frame when processing static scenes

# =============================================================================
# Scene-cut and duplicate detectors
# =============================================================================
scene_threshold = args.scene_threshold
if scene_threshold is None:
    scene_threshold = 0.41 if args.scene_detect == 'hash' else 0.2


def dct_hash(frame_np, size=16, factor=2):
    gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size * factor, size * factor), interpolation=cv2.INTER_AREA).astype(np.float32)
    small /= max(1.0, float(small.max()))
    low = cv2.dct(small)[:size, :size]
    return low > np.median(low)


min_scene_frames = max(1, int(round(0.5 * fps))) if args.video is not None else 1
last_cut_index = -10 ** 9
prev_hash = dct_hash(lastframe) if args.scene_detect == 'hash' else None


def small_luma(frame_np):
    return cv2.resize(cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)


last_unique_small = small_luma(lastframe) if args.dedup else None
dedup_skipped = 0

# Lookahead: the content detector compares a pair's luma difference with the next two pairs, so a
# hard cut (one isolated spike) is told apart from fast motion or a whip transition (several).
lookahead = []


def next_frame(timeout=None):
    if lookahead:
        return lookahead.pop(0)
    if timeout is None:
        return read_buffer.get()
    return read_buffer.get(timeout=timeout)


def peek_frames(count):
    while len(lookahead) < count and (not lookahead or lookahead[-1] is not None):
        lookahead.append(read_buffer.get())
    return [f for f in lookahead[:count] if f is not None]


class ContentCutDetector:
    """Cut when (a) SSIM of 32x32 thumbnails of the real (unpadded) frames < threshold and the HSV
    colour histograms differ (correlation < 0.75), or (b) the 64x64 luma MAD of the pair is a spike:
    above 30, 3x the mean of the previous four pairs and 2x the mean of the next two."""

    def __init__(self, first_frame, threshold):
        self.threshold = threshold
        self.cache = []  # (frame, features): each frame is analysed once (it is also a lookahead frame)
        self.prev = self._feat(first_frame)
        self.history = []

    @staticmethod
    def _area64(frame_np):
        # Stride-decimate large frames before the area resize: a full 4K INTER_AREA pass per frame
        # cost ~25 % of RIFE's 4K throughput on the main thread.
        h, w = frame_np.shape[:2]
        step = max(1, min(h, w) // 256)
        src = np.ascontiguousarray(frame_np[::step, ::step]) if step > 1 else frame_np
        return cv2.resize(src, (64, 64), interpolation=cv2.INTER_AREA)

    def _feat(self, frame_np):
        for f, feat in self.cache:
            if f is frame_np:
                return feat
        feat = self._compute(frame_np)
        self.cache.append((frame_np, feat))
        if len(self.cache) > 4:
            self.cache.pop(0)
        return feat

    def _compute(self, frame_np):
        small = self._area64(frame_np)
        luma = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_RGB2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
        thumb = torch.from_numpy(cv2.resize(frame_np, (32, 32), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1).copy())
        # Same device and dtype as ssim_matlab's window (FP16 mode makes half the default dtype).
        thumb = (thumb.unsqueeze(0).float() / 255.0).to(device=device, dtype=torch.get_default_dtype())
        return luma, hist, thumb

    def is_cut(self, frame_np, future):
        cur = self._feat(frame_np)
        mad = float(np.abs(cur[0] - self.prev[0]).mean())
        ssim_u = float(ssim_matlab(self.prev[2], cur[2]))
        corr = float(cv2.compareHist(self.prev[1], cur[1], cv2.HISTCMP_CORREL))
        nxt = []
        last = cur[0]
        for f in future[:2]:
            l = self._feat(f)[0]
            nxt.append(float(np.abs(l - last).mean()))
            last = l
        before = self.history[-4:]
        spike = (mad > 30.0
                 and mad > 3.0 * max(float(np.mean(before)) if before else 0.0, 8.0)
                 and mad > 2.0 * max(float(np.mean(nxt)) if nxt else 0.0, 8.0))
        self.history.append(mad)
        self.prev = cur
        return (ssim_u < self.threshold and corr < 0.75) or spike


content_detector = ContentCutDetector(lastframe, scene_threshold) if args.scene_detect == 'content' else None

# =============================================================================
# Main processing loop: read frames, run interpolation if needed,
# update progress, check for cancellation, and write out results.
# =============================================================================
# Initialize frame counter and timing variables
current_frame = 0
start_time = time.time()
last_update_time = start_time
frames_since_update = 0
recent_fps_values = []
out_index = math.ceil(timeline_start / step_ratio)  # next output frame index k (global grid)
pair_start = 0           # source index of I0 / lastframe
src_index = 0            # source index of the newest frame read
written = 0
cuts = 0

# Initialize status line
print("\nProcessing frames: ")
sys.stdout.flush()  # Force immediate output

# Function to filter ffmpeg output if not showing it
def check_ffmpeg_output():
    if not show_ffmpeg_output and ffmpeg_process and ffmpeg_process.stdout:
        try:
            # Non-blocking read from ffmpeg stdout
            line = ffmpeg_process.stdout.readline().decode('utf-8', errors='ignore').strip()
            pass
        except Exception:
            pass


def put_frame(source_np, out_np):
    global written
    if args.montage:
        write_buffer.put(np.concatenate((source_np, out_np), 1))
    else:
        write_buffer.put(out_np)
    written += 1


def emit_interval(start, end, I0, I1, src_frame, cut):
    """Write every output frame whose source time lies in [start, end)."""
    global out_index
    plan = []
    while True:
        g = out_index * step_ratio
        if timeline_end is not None and g >= timeline_end:
            break
        s = g - timeline_start
        if s >= end:
            break
        t = (s - start) / (end - start)
        plan.append(t)
        out_index += 1
    if not plan:
        return
    if model is not None and not supports_timestep:
        # Legacy models: recursive halving gives multi - 1 evenly spaced frames per pair.
        mids = [] if cut else make_inference(I0, I1, args.multi - 1, current_frame)
        mids = [to_frames(m)[0] for m in mids]
        put_frame(src_frame, src_frame)
        for j in range(1, len(plan)):
            put_frame(src_frame, mids[j - 1] if mids else src_frame)
        return
    need = [float(t) for t in plan if t != 0 and not cut]
    generated = iter(interpolate_frames(I0, I1, need))
    for t in plan:
        if t == 0 or cut:
            put_frame(src_frame, src_frame)
        else:
            put_frame(src_frame, next(generated))


while True:
    if cancel_increase:
        print("\n[INFO] Cancel button clicked. Cancelling processing immediately.")
        break
    if temp is not None:
        frame = temp
        temp = None
    else:
        try:
            frame = next_frame(timeout=0.1)
        except Empty:
            continue
    if frame is None:
        break

    current_frame += 1
    frames_since_update += 1
    src_index += 1

    current_time = time.time()

    if current_frame == 1 or current_frame % 100 == 0 or current_frame == tot_frame:
        elapsed = current_time - start_time
        recent_fps = frames_since_update / (current_time - last_update_time) if current_time > last_update_time else 0
        if recent_fps > 0:
            recent_fps_values.append(recent_fps)
            if len(recent_fps_values) > 5:
                recent_fps_values.pop(0)

        avg_recent_fps = sum(recent_fps_values) / len(recent_fps_values) if recent_fps_values else 0
        remaining_frames = tot_frame - current_frame
        eta_seconds = remaining_frames / avg_recent_fps if avg_recent_fps > 0 else 0

        eta_str = ""
        if eta_seconds > 0:
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds//60:.0f}m {eta_seconds%60:.0f}s"
            else:
                eta_str = f"{eta_seconds//3600:.0f}h {(eta_seconds%3600)//60:.0f}m"

        percent_complete = (current_frame / tot_frame) * 100 if tot_frame > 0 else 0

        bar_length = 30
        filled_length = int(bar_length * current_frame // max(1, tot_frame))
        filled_length = max(0, min(bar_length, filled_length))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)

        print(f"\r[{bar}] {current_frame}/{int(tot_frame)} ({percent_complete:.1f}%) | {avg_recent_fps:.1f} fps | ETA: {eta_str}    ", end="")
        sys.stdout.flush()

        check_ffmpeg_output()

        if current_time > last_update_time:
            last_update_time = current_time
            frames_since_update = 0

    # Duplicate (held) frames: skip them and interpolate across the whole gap instead.
    if args.dedup:
        cur_small = small_luma(frame)
        is_cut_candidate = False
        if args.scene_detect == 'hash':
            is_cut_candidate = float(np.count_nonzero(dct_hash(frame) != prev_hash)) / 256.0 >= scene_threshold
        if not is_cut_candidate:
            diff = np.abs(cur_small - last_unique_small)
            block_max = float(diff.reshape(8, 8, 8, 8).mean(axis=(1, 3)).max())
            if float(diff.mean()) < args.dedup_threshold and block_max < args.dedup_block_threshold:
                dedup_skipped += 1
                continue
        last_unique_small = cur_small

    I0 = I1
    I1 = pad_image(to_tensor(frame))
    pair_end = src_index

    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if args.static_blend and (not args.dedup) and model is not None and ssim > 0.996:
        # Upstream static-frame handling: replace a near-duplicate frame with the midpoint of
        # its neighbours (keeps the frame count).
        frame = next_frame()
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = pad_image(to_tensor(frame))
        I1 = model.inference(I0, I1, scale=args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = to_frames(I1)[0]

    cut = False
    if args.scene_detect == 'ssim':
        cut = bool(ssim < scene_threshold)
    elif args.scene_detect == 'content':
        future = ([temp] if temp is not None else []) + peek_frames(2)
        cut = content_detector.is_cut(frame, future)
    elif args.scene_detect == 'hash':
        cur_hash = dct_hash(frame)
        dist = float(np.count_nonzero(cur_hash != prev_hash)) / float(cur_hash.size)
        prev_hash = cur_hash
        if dist >= scene_threshold and (src_index - last_cut_index) >= min_scene_frames:
            cut = True
            last_cut_index = src_index
    if cut:
        cuts += 1

    if cancel_increase:
        print("\nFPS increase cancelled. Outputting original frames only.")
        break

    emit_interval(pair_start, pair_end, I0, I1, lastframe, cut)

    lastframe = frame
    pair_start = pair_end
    if break_flag:
        break

# Final source frame (and held duplicates at the end): every output frame up to the last source time.
last_source_index = src_index
while out_index * step_ratio - timeline_start <= last_source_index and (
        timeline_end is None or out_index * step_ratio < timeline_end):
    put_frame(lastframe, lastframe)
    out_index += 1
write_buffer.put(None)

# Wait for all frames to be written
print('\nFinalizing video output...', end="")
sys.stdout.flush()
writer_thread.join()

# Calculate total processing time
total_time = time.time() - start_time
final_fps = current_frame / total_time if total_time > 0 else 0

print(f" done")
print(f"[SUCCESS] Processing complete! {current_frame} frames in {total_time:.2f}s ({final_fps:.2f} fps)")
summary = f"[INFO] Output frames: {written} at {rate_text(out_rate)} fps"
if cuts:
    summary += f" | scene cuts: {cuts}"
if args.dedup:
    summary += f" | duplicates removed: {dedup_skipped}"
if feature_cache is not None and feature_cache.enabled:
    summary += f" | encoder cache hits {feature_cache.hits}, misses {feature_cache.misses}"
print(summary)
sys.stdout.flush()

should_transfer_audio = (not args.png) and (args.video is not None) and (not args.no_audio)

if not args.png:
    # <<< Wait for the first ffmpeg process (high-FPS encoding) to finish >>>
    print("\nWaiting for initial high-FPS encoding to complete...")
    sys.stdout.flush()
    ffmpeg_process_1.stdin.close() # Close stdin to signal end of input
    ffmpeg_process_1.wait() # Wait for the process to terminate
    print("Initial encoding finished.")

    # <<< Check if the temporary file was created >>>
    if os.path.exists(vid_out_name_temp_highfps) and os.path.getsize(vid_out_name_temp_highfps) > 0:

        # <<< Calculate target FPS >>>
        target_fps = float(out_rate) if exact_mode else args.fps

        # <<< Conditional Re-encoding >>>
        if target_fps < actual_high_fps - 1e-6:
            # Legacy path only: resample the fps*multi file down to the requested rate.
            print(f"Target FPS ({target_fps:.3f}) is lower than actual high FPS ({actual_high_fps:.2f}). Re-encoding to -> {os.path.basename(vid_out_name)}...")
            sys.stdout.flush()
            ffmpeg_cmd_2 = [
                'ffmpeg', '-y',
                '-i', vid_out_name_temp_highfps, # Input is the high-FPS temp file
                '-r', str(target_fps), # Apply the target frame rate (exact value)
                '-c:v', args.video_codec,
                '-preset', args.video_preset,
                '-crf', str(args.video_crf),
                '-pix_fmt', args.pixel_format,
                '-an', # No audio yet
            ]
            apply_codec_tuning(ffmpeg_cmd_2)
            ffmpeg_cmd_2.append(vid_out_name) # Output to the final video name
            re_encode_successful = False
            try:
                # Run the second ffmpeg process
                process = subprocess.run(ffmpeg_cmd_2, check=True, capture_output=True, text=True, encoding='utf-8',
                                         creationflags=_NO_WINDOW)
                print("Re-encoding finished successfully.")
                re_encode_successful = True
            except subprocess.CalledProcessError as e:
                print(f"\n!!! Error during re-encoding (FFmpeg step 2) !!!")
                print(f"Command: {' '.join(e.cmd)}")
                print(f"Stderr: {e.stderr}")
                should_transfer_audio = False # Don't attempt audio merge if re-encode failed
            except Exception as e:
                print(f"\n!!! Unexpected error during re-encoding: {e}")
                traceback.print_exc()
                should_transfer_audio = False # Don't attempt audio merge if re-encode failed

            # <<< Clean up the temporary high-FPS file ONLY if re-encode was successful >>>
            if re_encode_successful:
                try:
                    print(f"Removing temporary file: {vid_out_name_temp_highfps}")
                    os.remove(vid_out_name_temp_highfps)
                except OSError as e:
                    print(f"Warning: Could not remove temp file {vid_out_name_temp_highfps}: {e}")
            # If re-encode failed, the temp file is left for debugging

        else:
            # Target FPS equals the encoded rate, just rename the temp file
            print(f"Encoded at the output rate ({actual_high_fps:.3f} fps). Renaming temp file to -> {os.path.basename(vid_out_name)}, no re-encode.")
            try:
                os.replace(vid_out_name_temp_highfps, vid_out_name)
                print(f"Renamed '{os.path.basename(vid_out_name_temp_highfps)}' to '{os.path.basename(vid_out_name)}'")
            except OSError as e:
                print(f"!!! Error renaming temporary file {vid_out_name_temp_highfps} to {vid_out_name}: {e}")
                traceback.print_exc()
                should_transfer_audio = False # Don't attempt audio merge if rename failed

    else:
        print(f"Warning: Temporary high-FPS file '{vid_out_name_temp_highfps}' not found or empty. Skipping re-encoding and audio merge.")
        should_transfer_audio = False

if should_transfer_audio:
    print("\nProcessing audio...")
    try:
        transferAudio(args.video, vid_out_name)
    except Exception as audio_err: # Catch specific error
        print(f"Audio transfer failed: {audio_err}. Interpolated video will have no audio")
        # Check if the temp _noaudio file exists from transferAudio attempt
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        if os.path.exists(targetNoAudio):
            try:
                print(f"Renaming {targetNoAudio} back to {vid_out_name}")
                os.rename(targetNoAudio, vid_out_name)
            except OSError as rename_err:
                print(f"Error renaming back after failed audio transfer: {rename_err}")

if vfg is not None:
    try:
        vfg.close()
    except Exception:
        pass

if args.png:
    sequence_label = "JPG" if args.sequence_format == "jpg" else "PNG"
    print(f"\n[DONE] Completed {sequence_label} sequence saved to: {sequence_output_dir}")
else:
    print(f"\n[DONE] Completed video saved to: {vid_out_name}")
