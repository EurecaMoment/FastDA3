"""Check native imports, local assets and CUDA before loading the large model."""

import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile


def check_imports():
    # Include the native libraries used by both image and streaming requests.
    for name in (
        "cv2",
        "torch",
        "torchvision",
        "xformers",
        "triton",
        "fastapi",
        "httpx",
        "faiss",
        "numba",
        "pypose",
        "prettytable",
        "scipy",
        "sklearn",
        "depth_anything_3.api",
        "fastda3.server",
    ):
        importlib.import_module(name)


def validate(args):
    import torch
    from .streaming import StreamingRunner

    if not args.api_key or args.api_key == "replace-with-your-own-secret":
        raise RuntimeError(
            "Set DA3_API_KEY to your own secret in .env or the environment."
        )
    paths = (
        Path(args.model_dir) / "config.json",
        Path(args.model_dir) / "model.safetensors",
        Path(args.streaming_repo) / "configs/base_config.yaml",
        Path(args.streaming_repo) / "loop_utils/salad/models/helper.py",
        Path(args.salad_checkpoint),
        Path(args.dinov2_repo) / "hubconf.py",
    )
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing asset: {path}. Run bash scripts/download.sh.")
    config = json.loads(paths[0].read_text())
    if config.get("model_name") != "da3nested-giant-large":
        raise RuntimeError("FastDA3 requires the DA3NESTED-GIANT-LARGE-1.1 model.")
    streaming = StreamingRunner(None, args)
    if not streaming.availability()["available"]:
        raise RuntimeError(
            "Streaming dependencies are incomplete; run bash scripts/install.sh."
        )
    # Check the actual lazy imports too, before reporting all features available.
    for name in (
        "loop_utils.loop_detector",
        "loop_utils.sim3utils",
        "loop_utils.sim3loop",
    ):
        importlib.import_module(name)
    for name in ("output_dir", "files_dir", "jobs_dir"):
        directory = Path(getattr(args, name))
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError(
            "nvidia-smi is required for the GPU memory guard; check the NVIDIA driver."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; check the NVIDIA driver and CUDA_VISIBLE_DEVICES."
        )
    if torch.cuda.get_device_capability(0)[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "FastDA3 requires native BF16 support (NVIDIA Ampere or newer)."
        )
    properties = torch.cuda.get_device_properties(0)
    total_gb = properties.total_memory / 1e9
    if args.max_gpu_memory_gb > total_gb:
        raise RuntimeError(
            f"GPU has {total_gb:.1f} GB; lower --max-gpu-memory-gb to fit. "
            "The default 32 GB budget requires a GPU with at least 32 decimal GB."
        )
    # Execute a kernel to catch driver/wheel/architecture mismatches, not just CUDA enumeration.
    with torch.inference_mode():
        probe = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        (probe @ probe).sum().item()
    return f"{properties.name}; torch {torch.__version__}; all streaming assets present"


def main():
    try:
        check_imports()
        if sys.argv[1:] == ["--imports-only"]:
            print("FastDA3 serving and native-library imports passed.")
            return
        from .server import arguments

        print("FastDA3 preflight passed: " + validate(arguments()))
    except Exception as exc:
        sys.exit(f"FastDA3 preflight failed: {exc}")


if __name__ == "__main__":
    main()
