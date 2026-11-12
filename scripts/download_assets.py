"""Download the model, checksum-verified SALAD weights, and pinned DINOv2 source."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
import urllib.request
import zipfile

from fastda3.config import repository_root, resolve_path

ROOT = repository_root()
MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
SALAD_URL = "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
SALAD_SHA256 = "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def download(url, destination, checksum=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "FastDA3/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            for block in iter(lambda: response.read(1024**2), b""):
                output.write(block)
        if checksum and digest(temporary) != checksum:
            raise RuntimeError("Checksum mismatch for " + destination.name)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.getenv(
            "DA3_MODEL_DIR", str(ROOT / "weights/DA3NESTED-GIANT-LARGE-1.1")
        ),
    )
    parser.add_argument(
        "--salad-checkpoint",
        type=Path,
        default=os.getenv(
            "DA3_SALAD_CHECKPOINT", str(ROOT / "weights/dino_salad.ckpt")
        ),
    )
    parser.add_argument(
        "--dinov2-repo",
        type=Path,
        default=os.getenv("DA3_DINOV2_REPO", str(ROOT / "vendor/dinov2")),
    )
    parser.add_argument(
        "--streaming-repo",
        type=Path,
        default=os.getenv("DA3_STREAMING_REPO", str(ROOT / "da3_streaming")),
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    for name in ("model_dir", "salad_checkpoint", "dinov2_repo", "streaming_repo"):
        setattr(args, name, resolve_path(getattr(args, name), ROOT))
    salad_source = args.streaming_repo / "loop_utils/salad/models/helper.py"
    if not salad_source.is_file():
        parser.error(
            "SALAD submodule is missing; run git submodule update --init --recursive"
        )
    if not args.check_only:
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL,
            revision=args.model_revision,
            local_dir=args.model_dir,
            allow_patterns=["config.json", "model.safetensors"],
        )
        if (
            not args.salad_checkpoint.is_file()
            or digest(args.salad_checkpoint) != SALAD_SHA256
        ):
            download(SALAD_URL, args.salad_checkpoint, SALAD_SHA256)
        revision_file = args.dinov2_repo / ".fastda3-revision"
        if (
            not revision_file.is_file()
            or revision_file.read_text().strip() != DINOV2_REVISION
        ):
            args.dinov2_repo.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="fastda3-dinov2-") as directory:
                archive = Path(directory) / "source.zip"
                download(
                    f"https://codeload.github.com/facebookresearch/dinov2/zip/{DINOV2_REVISION}",
                    archive,
                )
                with zipfile.ZipFile(archive) as source:
                    prefix = source.namelist()[0].split("/")[0] + "/"
                    for name in source.namelist():
                        relative = PurePosixPath(name.removeprefix(prefix))
                        if not relative.parts or name.endswith("/"):
                            continue
                        if ".." in relative.parts or relative.is_absolute():
                            raise RuntimeError("Unsafe path in source archive")
                        target = args.dinov2_repo.joinpath(*relative.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(source.read(name))
            revision_file.write_text(DINOV2_REVISION + "\n")
    for path in [
        args.model_dir / "config.json",
        args.model_dir / "model.safetensors",
        args.salad_checkpoint,
        args.dinov2_repo / "hubconf.py",
    ]:
        if not path.is_file():
            parser.error("Missing asset: " + str(path))
    if digest(args.salad_checkpoint) != SALAD_SHA256:
        parser.error("SALAD checkpoint checksum mismatch")
    print("DA3 model, SALAD source/weights, and DINOv2 source are ready.")


if __name__ == "__main__":
    main()
