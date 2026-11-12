"""DA3-Streaming adapter: shared model, overlap Sim(3), SALAD loops, disk chunks."""

import asyncio
import copy
import importlib.util
import json
import math
import os
import secrets
import shutil
import sys
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .artifacts import atomic_json
from .protocol import APIError, cameras


class JobCancelled(Exception):
    pass


def chunk_ranges(count, size, overlap):
    if not count:
        return []
    if not 0 < overlap < size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < count:
        end = min(start + size, count)
        chunks.append((start, end))
        if end == count:
            break
        start = end - overlap
    return chunks


def backproject(depth, intrinsics, extrinsics, stride=1):
    depth = np.asarray(depth, dtype=np.float32)
    ex = np.asarray(extrinsics, dtype=np.float32)
    rows = np.arange(0, depth.shape[-2], stride, dtype=np.float32)
    cols = np.arange(0, depth.shape[-1], stride, dtype=np.float32)
    u, v = np.meshgrid(cols, rows)
    pixels = np.stack((u, v, np.ones_like(u)), -1)
    ray = np.einsum("nij,hwj->nhwi", np.linalg.inv(intrinsics), pixels)
    camera = ray * depth[:, ::stride, ::stride, None]
    rotation = ex[:, :3, :3].transpose(0, 2, 1)
    center = -np.einsum("nij,nj->ni", rotation, ex[:, :3, 3])
    return np.einsum("nij,nhwj->nhwi", rotation, camera) + center[:, None, None, :]


def resize_intrinsics(intrinsics, old_sizes, new_sizes):
    result = np.asarray(intrinsics, dtype=np.float32).copy()
    if result.shape != (len(old_sizes), 3, 3) or len(old_sizes) != len(new_sizes):
        raise ValueError("Intrinsics and per-frame sizes must be aligned")
    for index, ((old_w, old_h), (new_w, new_h)) in enumerate(zip(old_sizes, new_sizes)):
        result[index, 0, :] *= new_w / old_w
        result[index, 1, :] *= new_h / old_h
    return result


def center_crop_prediction(result, height, width):
    """Match overlap pixels when adjacent mixed-aspect windows use different crops."""
    old_h, old_w = result["depth"].shape[-2:]
    top, left = (old_h - height) // 2, (old_w - width) // 2
    output = dict(result)
    for key in ("depth", "confidence", "sky"):
        if key in output:
            output[key] = output[key][..., top : top + height, left : left + width]
    output["intrinsics"] = result["intrinsics"].copy()
    output["intrinsics"][..., 0, 2] -= left
    output["intrinsics"][..., 1, 2] -= top
    return output


def compose(first, second):
    s1, r1, t1 = first
    s2, r2, t2 = second
    return float(s1 * s2), r1 @ r2, s1 * (r1 @ t2) + t1


def inverse(transform):
    scale, rotation, translation = transform
    r = rotation.T
    return 1.0 / scale, r, -(r @ translation) / scale


def transformed_cameras(extrinsics, transform):
    scale, rotation, translation = transform
    count = len(extrinsics)
    w2c = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy()
    w2c[:, :3] = extrinsics
    c2w = np.linalg.inv(w2c)
    out = c2w.copy()
    out[:, :3, :3] = rotation @ c2w[:, :3, :3]
    out[:, :3, 3] = scale * (c2w[:, :3, 3] @ rotation.T) + translation
    return np.linalg.inv(out)[:, :3].astype(np.float32), out


class StreamingRunner:
    def __init__(self, scheduler, args):
        self.scheduler = scheduler
        self.args = args
        self.repo = Path(args.streaming_repo)
        self.dinov2_repo = Path(
            getattr(args, "dinov2_repo", None)
            or Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        )
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        self.config = None
        self.loop_model = None

    def availability(self):
        names = ["faiss", "pypose", "numba", "prettytable"]
        missing = [name for name in names if importlib.util.find_spec(name) is None]
        salad = self.repo / "loop_utils/salad/models/helper.py"
        checkpoint = Path(self.args.salad_checkpoint)
        dinov2 = (self.dinov2_repo / "hubconf.py").is_file()
        return {
            "available": bool(
                not missing and salad.is_file() and checkpoint.is_file() and dinov2
            ),
            "missing_dependencies": missing,
            "salad_source": salad.is_file(),
            "salad_checkpoint": checkpoint.is_file(),
            "dinov2_source": dinov2,
            "loop_closure": True,
            "implementation": "upstream alignment and SALAD loop optimizer with shared DA3 model",
        }

    def load_config(self):
        if self.config is None:
            import yaml

            self.config = yaml.safe_load(
                (self.repo / "configs/base_config.yaml").read_text()
            )
            self.config["Model"]["align_lib"] = "numpy"
            self.config["Model"]["align_method"] = "sim3"
            self.config["Loop"]["SIM3_Optimizer"]["lang_version"] = "python"
            self.config["Weights"]["SALAD"] = self.args.salad_checkpoint
        return copy.deepcopy(self.config)

    def align(self, fixed, moving, maximum=100000):
        from loop_utils.sim3utils import weighted_align_point_maps

        count = min(len(fixed["depth"]), len(moving["depth"]))
        if not count:
            raise ValueError("At least one overlapping view is required")
        h = min(fixed["depth"].shape[-2], moving["depth"].shape[-2])
        w = min(fixed["depth"].shape[-1], moving["depth"].shape[-1])
        fixed = center_crop_prediction(fixed, h, w)
        moving = center_crop_prediction(moving, h, w)
        stride = max(1, int(math.ceil(math.sqrt(count * h * w / maximum))))
        points1 = backproject(
            fixed["depth"][:count],
            fixed["intrinsics"][:count],
            fixed["extrinsics"][:count],
            stride,
        )
        points2 = backproject(
            moving["depth"][:count],
            moving["intrinsics"][:count],
            moving["extrinsics"][:count],
            stride,
        )
        conf1 = np.maximum(fixed["confidence"][:count, ::stride, ::stride] - 1, 0)
        conf2 = np.maximum(moving["confidence"][:count, ::stride, ::stride] - 1, 0)
        if "sky" in fixed:
            conf1[fixed["sky"][:count, ::stride, ::stride]] = 0
        if "sky" in moving:
            conf2[moving["sky"][:count, ::stride, ::stride]] = 0
        threshold = float(min(np.median(conf1), np.median(conf2)) * 0.1)
        scale, rotation, translation = weighted_align_point_maps(
            points1,
            conf1,
            points2,
            conf2,
            conf_threshold=threshold,
            config=self.load_config(),
        )
        scale = float(scale)
        rotation = np.asarray(rotation, dtype=np.float32)
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        if not (
            np.isfinite(scale)
            and scale > 0
            and np.isfinite(rotation).all()
            and np.isfinite(translation).all()
        ):
            raise ValueError("Non-finite overlap alignment")
        return scale, rotation, translation

    @staticmethod
    def subset(result, indices):
        return {
            key: value[indices]
            for key, value in result.items()
            if key in ("depth", "confidence", "sky", "intrinsics", "extrinsics")
        }

    async def loop_detector(self, image_dir, options):
        from loop_utils.loop_detector import LoopDetector, VPRModel

        config = self.load_config()
        config["Loop"]["SALAD"].update(
            batch_size=8,
            similarity_threshold=options.streaming["loop_threshold"],
            nms_threshold=options.streaming["loop_min_separation"],
        )
        detector = LoopDetector(str(image_dir), config=config)
        if self.loop_model is None:
            loop = asyncio.get_running_loop()

            def load():
                # Construct on CPU so descriptor initialization does not compete with DA3.
                original = torch.hub.load
                cached_dino = str(self.dinov2_repo)

                def local_hub(repo, model, *args, **kwargs):
                    if repo == "facebookresearch/dinov2":
                        return original(
                            cached_dino, model, source="local", pretrained=False
                        )
                    return original(repo, model, *args, **kwargs)

                torch.hub.load = local_hub
                try:
                    with torch.inference_mode():
                        model = VPRModel(
                            backbone_arch="dinov2_vitb14",
                            backbone_config={
                                "num_trainable_blocks": 4,
                                "return_token": True,
                                "norm_layer": True,
                            },
                            agg_arch="SALAD",
                            agg_config={
                                "num_channels": 768,
                                "num_clusters": 64,
                                "cluster_dim": 128,
                                "token_dim": 256,
                            },
                        )
                        state = torch.load(
                            self.args.salad_checkpoint,
                            map_location="cpu",
                            weights_only=True,
                        )
                        model.load_state_dict(state)
                        self.loop_model = model.eval()
                finally:
                    torch.hub.load = original

            await loop.run_in_executor(self.scheduler.gpu_pool, load)
        detector.model = self.loop_model
        detector.device = torch.device("cuda")
        return detector

    async def descriptors(self, paths, image_dir, options, progress):
        detector = await self.loop_detector(image_dir, options)
        from torchvision.transforms import (
            Compose,
            Resize,
            ToTensor,
            Normalize,
            InterpolationMode,
        )

        transform = Compose(
            [
                Resize([336, 336], interpolation=InterpolationMode.BILINEAR),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        import faiss

        faiss.omp_set_num_threads(4)
        index = faiss.IndexFlatIP(8448)
        pairs = []
        delay = options.streaming["loop_min_separation"]
        threshold = options.streaming["loop_threshold"]
        loop = asyncio.get_running_loop()
        tail = []
        try:
            for begin in range(0, len(paths), 8):
                progress("loop_detection", begin, len(paths))
                selected = paths[begin : begin + 8]

                def prepare():
                    tensors = []
                    for path in selected:
                        with Image.open(path) as image:
                            tensors.append(transform(image.convert("RGB")))
                    return torch.stack(tensors).pin_memory()

                tensors = await loop.run_in_executor(self.scheduler.cpu_pool, prepare)

                def infer():
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        detector.model.to("cuda")
                        value = (
                            detector.model(tensors.to("cuda", non_blocking=True))
                            .float()
                            .cpu()
                            .numpy()
                        )
                    return value

                async with self.scheduler.gpu_lock:
                    batch = await loop.run_in_executor(self.scheduler.gpu_pool, infer)
                faiss.normalize_L2(batch)
                for offset, descriptor in enumerate(batch):
                    current = begin + offset
                    while tail and tail[0][0] <= current - delay:
                        _, old = tail.pop(0)
                        index.add(old[None])
                    if index.ntotal:
                        scores, neighbors = index.search(descriptor[None], 1)
                        if scores[0, 0] >= threshold:
                            pairs.append(
                                (current, int(neighbors[0, 0]), float(scores[0, 0]))
                            )
                    tail.append((current, descriptor.copy()))
        finally:
            async with self.scheduler.gpu_lock:
                await loop.run_in_executor(
                    self.scheduler.gpu_pool, lambda: detector.model.to("cpu")
                )
        pairs.sort(key=lambda row: row[2], reverse=True)
        chosen = []
        for pair in pairs:
            if len(chosen) >= options.streaming["max_loop_pairs"]:
                break
            if not any(
                abs(pair[0] - p[0]) < 5 and abs(pair[1] - p[1]) < 5 for p in chosen
            ):
                chosen.append(pair)
        return chosen

    async def extract(self, job, options, directory):
        directory.mkdir(parents=True, exist_ok=True)
        limit = options.streaming["max_frames"]
        if options.video:
            video = await self.scheduler.materialize_video(options.video, job["id"])
            loop = asyncio.get_running_loop()

            def decode():
                capture = cv2.VideoCapture(str(video))
                if not capture.isOpened():
                    raise APIError("Video could not be decoded")
                native = float(capture.get(cv2.CAP_PROP_FPS))
                if not np.isfinite(native) or native <= 0:
                    native = options.streaming["fps"]
                step = native / options.streaming["fps"]
                frame_idx = 0
                saved = 0
                next_frame = 0.0
                source_sizes = []
                try:
                    while True:
                        self.scheduler.jobs.check_cancel(job["id"])
                        ok = capture.grab()
                        if not ok:
                            break
                        if frame_idx + 1e-6 >= next_frame:
                            ok, frame = capture.retrieve()
                            if not ok:
                                raise APIError("Video frame could not be decoded")
                            if saved >= limit:
                                raise APIError(
                                    "Video exceeds max_frames; raise the limit or lower fps"
                                )
                            h, w = frame.shape[:2]
                            source_sizes.append((w, h))
                            scale = min(1.0, options.process_res / max(h, w))
                            if scale < 1:
                                frame = cv2.resize(
                                    frame,
                                    (round(w * scale), round(h * scale)),
                                    interpolation=cv2.INTER_AREA,
                                )
                            if not cv2.imwrite(
                                str(directory / f"{saved:08d}.png"), frame
                            ):
                                raise OSError("Could not write decoded frame")
                            self.scheduler.jobs.track_file(
                                job["id"], directory / f"{saved:08d}.png"
                            )
                            saved += 1
                            next_frame = saved * step
                            if saved % 50 == 0:
                                self.scheduler.jobs.update(
                                    job["id"], stage="decoding", frames_decoded=saved
                                )
                        frame_idx += 1
                finally:
                    capture.release()
                if saved < 2:
                    raise APIError(
                        "Streaming video requires at least two sampled frames"
                    )
                return saved, native, source_sizes

            count, native, source_sizes = await loop.run_in_executor(
                self.scheduler.cpu_pool, decode
            )
            job["frame_sizes"] = source_sizes
            job["sample_fps"] = min(native, options.streaming["fps"])
        else:
            if len(options.images) > limit:
                raise APIError("Image sequence exceeds max_frames")
            if len(options.images) < 2:
                raise APIError("Streaming requires at least two frames")
            for begin in range(0, len(options.images), 8):
                self.scheduler.jobs.check_cancel(job["id"])
                for index, src in enumerate(options.images[begin : begin + 8], begin):
                    image = await self.scheduler.decode_image(src)
                    path = directory / f"{index:08d}.png"
                    await asyncio.get_running_loop().run_in_executor(
                        self.scheduler.cpu_pool, lambda im=image, p=path: im.save(p)
                    )
                    self.scheduler.jobs.track_file(job["id"], path)
            count = len(options.images)
            source_sizes = []
        if options.video:
            source_sizes = job["frame_sizes"]
        job["frame_sizes"] = source_sizes
        return [directory / f"{i:08d}.png" for i in range(count)]

    async def run(self, job, options):
        availability = self.availability()
        if not availability["available"]:
            raise APIError(
                "DA3-Streaming dependencies or SALAD weights are unavailable",
                503,
                "streaming_unavailable",
            )
        work = self.scheduler.jobs.root / job["id"]
        frames_dir = work / "input"
        chunks_dir = work / "chunks"
        out_dir = work / "output"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        paths = await self.extract(job, options, frames_dir)
        count = len(paths)
        ex, ix = cameras(options.extrinsics, options.intrinsics, count)
        chunk = options.streaming["chunk_size"]
        if chunk is None:
            processor = self.scheduler.engine.model.input_processor
            with Image.open(paths[0]) as image:
                resized = processor._resize_image(
                    image.convert("RGB"), options.process_res, "upper_bound_resize"
                )
            shape = processor._make_divisible_by_resize(resized, 14).size
            pixels = max(1, shape[0] * shape[1])
            target = 4 if options.process_res > 700 else 12
            chunk = min(target, max(2, self.scheduler.args.max_scene_pixels // pixels))
        overlap = options.streaming["overlap"] or max(1, chunk // 3)
        if overlap >= chunk:
            raise APIError("streaming.overlap must be less than chunk_size")
        ranges = chunk_ranges(count, chunk, overlap)
        input_world = ex is not None and options.align_to_input_ext_scale
        relative = []
        previous = None

        def progress(stage, done, total):
            self.scheduler.jobs.check_cancel(job["id"])
            self.scheduler.jobs.update(
                job["id"],
                stage=stage,
                progress={"completed": done, "total": total},
                frame_count=count,
                chunk_size=chunk,
                overlap=overlap,
            )

        for ci, (begin, end) in enumerate(ranges):
            progress("inference", begin, count)
            local_ex = ex[begin:end] if ex is not None else None
            local_ix = ix[begin:end].copy() if ix is not None else None
            if local_ix is not None and options.video:
                resized_sizes = []
                for path in paths[begin:end]:
                    with Image.open(path) as image:
                        resized_sizes.append(image.size)
                local_ix = resize_intrinsics(
                    local_ix, job["frame_sizes"][begin:end], resized_sizes
                )
            result = await self.scheduler.infer_scene_paths(
                paths[begin:end], options, local_ex, local_ix
            )
            if previous is not None and not input_world:
                fixed = self.subset(previous, slice(-overlap, None))
                moving = self.subset(result, slice(0, overlap))
                transform = await asyncio.get_running_loop().run_in_executor(
                    self.scheduler.cpu_pool,
                    self.align,
                    fixed,
                    moving,
                    options.streaming["max_align_points"],
                )
                relative.append(transform)

            def persist():
                np.savez(
                    chunks_dir / f"{ci:08d}.npz",
                    depth=result["depth"],
                    confidence=result["confidence"],
                    sky=result["sky"],
                    intrinsics=result["intrinsics"],
                    extrinsics=result["extrinsics"],
                )

            await asyncio.get_running_loop().run_in_executor(
                self.scheduler.cpu_pool, persist
            )
            self.scheduler.jobs.track_file(job["id"], chunks_dir / f"{ci:08d}.npz")
            previous = self.subset(result, slice(-overlap, None))
        loops = []
        constraints = []
        if options.streaming["loop_closure"] and len(ranges) > 1 and ex is None:
            loops = await self.descriptors(paths, frames_dir, options, progress)

            def owner(frame):
                return next(i for i, (a, b) in enumerate(ranges) if a <= frame < b)

            for li, (current, prior, score) in enumerate(loops):
                progress("loop_alignment", li, len(loops))
                a, b = owner(prior), owner(current)
                if a == b or abs(a - b) <= 1:
                    continue
                joint = await self.scheduler.infer_scene_paths(
                    [paths[prior], paths[current]],
                    options,
                    ex[[prior, current]] if ex is not None else None,
                    ix[[prior, current]] if ix is not None else None,
                )
                with np.load(chunks_dir / f"{a:08d}.npz") as data:
                    fixed_a = {
                        key: data[key][prior - ranges[a][0] : prior - ranges[a][0] + 1]
                        for key in data.files
                    }
                with np.load(chunks_dir / f"{b:08d}.npz") as data:
                    fixed_b = {
                        key: data[key][
                            current - ranges[b][0] : current - ranges[b][0] + 1
                        ]
                        for key in data.files
                    }
                ta = await asyncio.get_running_loop().run_in_executor(
                    self.scheduler.cpu_pool,
                    self.align,
                    fixed_a,
                    self.subset(joint, slice(0, 1)),
                    options.streaming["max_align_points"],
                )
                tb = await asyncio.get_running_loop().run_in_executor(
                    self.scheduler.cpu_pool,
                    self.align,
                    fixed_b,
                    self.subset(joint, slice(1, 2)),
                    options.streaming["max_align_points"],
                )
                constraints.append((a, b, compose(ta, inverse(tb))))
            if constraints:
                from loop_utils.sim3loop import Sim3LoopOptimizer

                optimizer = Sim3LoopOptimizer(self.load_config())
                relative = await asyncio.get_running_loop().run_in_executor(
                    self.scheduler.cpu_pool, optimizer.optimize, relative, constraints
                )
        identity = (1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        if input_world:
            # Each conditioned window already uses the supplied world frame.
            absolute = [identity for _ in ranges]
        else:
            absolute = [identity]
            for transform in relative:
                absolute.append(compose(absolute[-1], transform))
        ply_raw = out_dir / "points.bin"
        point_count = 0
        poses = np.empty((count, 4, 4), np.float32)
        intrinsics = np.empty((count, 3, 3), np.float32)
        written = np.zeros(count, dtype=bool)
        for ci, ((begin, end), transform) in enumerate(zip(ranges, absolute)):
            progress("export", begin, count)
            with np.load(chunks_dir / f"{ci:08d}.npz") as data:
                arrays = {key: data[key] for key in data.files}
            scale, rotation, translation = transform
            aligned_ex, c2w = transformed_cameras(arrays["extrinsics"], transform)
            # Each frame is written once, including short single-chunk sequences.
            keep_end = end if ci == len(ranges) - 1 else ranges[ci + 1][0]
            for frame in range(begin, keep_end):
                local = frame - begin
                depth = arrays["depth"][local] * scale
                poses[frame] = c2w[local]
                intrinsics[frame] = arrays["intrinsics"][local]
                written[frame] = True
                if options.streaming["save_depth"]:
                    target = out_dir / f"frame_{frame:08d}.npz"

                    def save_depth_file(
                        path=target,
                        depth_value=depth,
                        confidence=arrays["confidence"][local],
                        sky=arrays["sky"][local],
                        intrinsic=arrays["intrinsics"][local],
                        extrinsic=aligned_ex[local],
                    ):
                        np.savez(
                            path,
                            depth_m=depth_value,
                            confidence=confidence,
                            sky_mask=sky,
                            intrinsics=intrinsic,
                            extrinsics=extrinsic,
                        )

                    await asyncio.get_running_loop().run_in_executor(
                        self.scheduler.cpu_pool, save_depth_file
                    )
                    self.scheduler.jobs.track_file(job["id"], target)
                if options.streaming["save_pointcloud"]:
                    stride = options.streaming["point_stride"]

                    def append_points(
                        frame_index=frame,
                        local_index=local,
                        depth_value=depth,
                        extrinsic=aligned_ex[local],
                        intrinsic=arrays["intrinsics"][local],
                        confidence=arrays["confidence"][local],
                        sky=arrays["sky"][local],
                    ):
                        xyz = backproject(
                            depth_value[None], intrinsic[None], extrinsic[None], stride
                        )[0]
                        rgb = self.processed_rgb(
                            paths[frame_index], options.process_res, depth_value.shape
                        )[::stride, ::stride]
                        good = np.isfinite(xyz).all(-1) & ~sky[::stride, ::stride]
                        good &= confidence[::stride, ::stride] >= np.quantile(
                            confidence, 0.4
                        )
                        vertices = np.empty(
                            int(good.sum()),
                            dtype=[
                                ("x", "<f4"),
                                ("y", "<f4"),
                                ("z", "<f4"),
                                ("red", "u1"),
                                ("green", "u1"),
                                ("blue", "u1"),
                            ],
                        )
                        for axis, key in enumerate(["x", "y", "z"]):
                            vertices[key] = xyz[..., axis][good]
                        for axis, key in enumerate(["red", "green", "blue"]):
                            vertices[key] = rgb[..., axis][good]
                        with ply_raw.open("ab") as output:
                            vertices.tofile(output)
                        self.scheduler.jobs.track_file(job["id"], ply_raw)
                        return len(vertices)

                    point_count += await asyncio.get_running_loop().run_in_executor(
                        self.scheduler.cpu_pool, append_points
                    )
                await asyncio.sleep(0)
            chunks_dir.joinpath(f"{ci:08d}.npz").unlink(missing_ok=True)
            self.scheduler.jobs.untrack_tree(chunks_dir / f"{ci:08d}.npz")
            await asyncio.sleep(0)
        if not written.all():
            raise RuntimeError("Streaming export did not cover every frame")
        np.savez(out_dir / "cameras.npz", c2w=poses, intrinsics=intrinsics)
        self.scheduler.jobs.track_file(job["id"], out_dir / "cameras.npz")
        np.savetxt(out_dir / "camera_poses.txt", poses.reshape(count, 16))
        self.scheduler.jobs.track_file(job["id"], out_dir / "camera_poses.txt")
        np.savetxt(out_dir / "intrinsics.txt", intrinsics.reshape(count, 9))
        self.scheduler.jobs.track_file(job["id"], out_dir / "intrinsics.txt")
        if options.streaming["save_pointcloud"]:
            header = (
                "ply\nformat binary_little_endian 1.0\nelement vertex "
                + str(point_count)
                + "\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\n"
                "property uchar green\nproperty uchar blue\nend_header\n"
            )
            with (out_dir / "pointcloud.ply").open("wb") as output:
                output.write(header.encode())
                self.scheduler.jobs.track_file(job["id"], out_dir / "pointcloud.ply")
                if ply_raw.exists():
                    with ply_raw.open("rb") as src:
                        shutil.copyfileobj(src, output, 1024 * 1024)
            self.scheduler.jobs.track_file(job["id"], out_dir / "pointcloud.ply")
            ply_raw.unlink(missing_ok=True)
            self.scheduler.jobs.untrack_tree(ply_raw)
        identifier = secrets.token_hex(12)
        manifest = {
            "model": self.args.served_model_name,
            "task": "streaming",
            "implementation": "DA3-Streaming",
            "frame_count": count,
            "process_res": options.process_res,
            "chunk_count": len(ranges),
            "chunk_size": chunk,
            "overlap": overlap,
            "use_ray_pose": options.use_ray_pose,
            "pose_conditioned": ex is not None,
            "loop_closure_requested": options.streaming["loop_closure"],
            "loop_detection_applied": bool(
                options.streaming["loop_closure"] and len(ranges) > 1 and ex is None
            ),
            "loop_closure_applied": bool(constraints),
            "loop_candidates": len(loops),
            "loop_constraints": len(constraints),
            "coordinate_convention": "camera_poses=c2w; per-frame extrinsics=w2c",
            "units": "meters",
            "is_metric": True,
            "point_count": point_count,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "result_url": f"/v1/results/{identifier}.zip",
            "scale_note": (
                "input camera poses define the world coordinate system"
                if input_world
                else "overlap Sim(3) aligns chunk scales to the first metric chunk"
            ),
            "files": sorted(path.name for path in out_dir.iterdir()),
        }
        atomic_json(out_dir / "manifest.json", manifest)
        archive = self.scheduler.store.root / (identifier + ".zip")
        self.scheduler.jobs.check_capacity(
            sum(path.stat().st_size for path in out_dir.iterdir() if path.is_file())
        )

        def bundle():
            with zipfile.ZipFile(
                archive.with_suffix(".zip.tmp"), "w", compression=zipfile.ZIP_STORED
            ) as output:
                for path in sorted(out_dir.iterdir()):
                    output.write(path, path.name)
            os.replace(archive.with_suffix(".zip.tmp"), archive)

        await asyncio.get_running_loop().run_in_executor(
            self.scheduler.cpu_pool, bundle
        )
        try:
            self.scheduler.jobs.track_file(identifier, archive)
        except Exception:
            archive.unlink(missing_ok=True)
            raise
        return manifest

    def processed_rgb(self, path, process_res, target_shape=None):
        processor = self.scheduler.engine.model.input_processor
        with Image.open(path) as image:
            image = image.convert("RGB")
        processed, _, _ = processor(
            [image],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            num_workers=1,
        )
        array = processed[0].permute(1, 2, 0).numpy()
        if target_shape is not None:
            height, width = target_shape
            top = max(0, (array.shape[0] - height) // 2)
            left = max(0, (array.shape[1] - width) // 2)
            array = array[top : top + height, left : left + width]
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        return np.clip((array * std + mean) * 255, 0, 255).astype(np.uint8)


class JobManager:
    def __init__(self, scheduler, root):
        self.scheduler = scheduler
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.tasks = {}
        self.cancelled = set()
        self.cancel_events = {}
        self.slot = asyncio.Semaphore(1)
        self.lock = threading.RLock()
        self.max_disk_bytes = scheduler.args.max_job_disk_gb * 1024**3
        self.disk_files = {}
        for base in (self.root, scheduler.store.root):
            for path in base.rglob("*"):
                if path.is_file():
                    self.disk_files[str(path)] = path.stat().st_size
        self.disk_bytes = sum(self.disk_files.values())
        for path in self.root.glob("job-*/status.json"):
            try:
                value = json.loads(path.read_text())
                if value["status"] in ("queued", "in_progress"):
                    value.update(
                        status="failed",
                        error={
                            "code": "service_restarted",
                            "message": "Service restarted before completion",
                        },
                    )
                    atomic_json(path, value)
                self.records[value["id"]] = value
            except (OSError, ValueError, KeyError):
                pass

    def update(self, identifier, **values):
        with self.lock:
            record = self.records[identifier]
            record.update(values, updated_at=int(time.time()))
            atomic_json(self.root / identifier / "status.json", record)

    def track_file(self, identifier, path, size=None):
        path = Path(path)
        key = str(path)
        if size is None:
            size = path.stat().st_size
        with self.lock:
            old = self.disk_files.get(key, 0)
            if self.disk_bytes + size - old > self.max_disk_bytes:
                raise APIError(
                    "DA3 job and result files exceed the configured disk budget",
                    507,
                    "storage_full",
                )
            self.disk_files[key] = size
            self.disk_bytes += size - old

    def check_capacity(self, additional_bytes):
        with self.lock:
            if self.disk_bytes + additional_bytes > self.max_disk_bytes:
                raise APIError(
                    "DA3 job and result files exceed the configured disk budget",
                    507,
                    "storage_full",
                )

    def untrack_tree(self, path):
        prefix = str(Path(path))
        with self.lock:
            for key, size in list(self.disk_files.items()):
                if key == prefix or key.startswith(prefix + os.sep):
                    self.disk_bytes -= size
                    self.disk_files.pop(key, None)

    def check_cancel(self, identifier):
        if identifier in self.cancelled:
            raise JobCancelled("Job was cancelled")

    def create(self, options):
        with self.lock:
            if self.disk_bytes >= self.max_disk_bytes:
                raise APIError(
                    "DA3 job and result disk budget is full", 507, "storage_full"
                )
            if (
                sum(
                    value["status"] in ("queued", "in_progress")
                    for value in self.records.values()
                )
                >= 4
            ):
                raise APIError(
                    "Background job queue is full", 429, "rate_limit_exceeded"
                )
        identifier = "job-" + secrets.token_hex(12)
        self.root.joinpath(identifier).mkdir()
        record = {
            "id": identifier,
            "object": "depth.job",
            "status": "queued",
            "created_at": int(time.time()),
            "model": self.scheduler.args.served_model_name,
            "task": options.task,
            "stage": "queued",
            "result": None,
            "error": None,
        }
        self.records[identifier] = record
        self.update(identifier)
        self.cancel_events[identifier] = asyncio.Event()
        self.tasks[identifier] = asyncio.create_task(self._run(identifier, options))
        return dict(record)

    async def _run(self, identifier, options):
        record = self.records[identifier]
        acquired = False
        acquire = cancelled = None
        try:
            acquire = asyncio.create_task(self.slot.acquire())
            cancelled = asyncio.create_task(self.cancel_events[identifier].wait())
            done, pending = await asyncio.wait(
                (acquire, cancelled), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if cancelled in done:
                raise JobCancelled("Job was cancelled while queued")
            acquired = acquire.result()
            self.check_cancel(identifier)
            self.update(identifier, status="in_progress", started_at=int(time.time()))
            if options.task == "streaming":
                result = await self.scheduler.streaming.run(record, options)
            else:
                result = await self.scheduler.execute(options)
            self.check_cancel(identifier)
            self.clear_job_data(identifier)
            self.update(
                identifier,
                status="completed",
                stage="completed",
                result=result,
                completed_at=int(time.time()),
            )
        except JobCancelled:
            self.clear_job_data(identifier)
            self.update(
                identifier,
                status="cancelled",
                stage="cancelled",
                completed_at=int(time.time()),
            )
        except asyncio.CancelledError:
            self.clear_job_data(identifier)
            if identifier in self.cancelled:
                self.update(
                    identifier,
                    status="cancelled",
                    stage="cancelled",
                    completed_at=int(time.time()),
                )
            else:
                self.update(
                    identifier,
                    status="failed",
                    stage="failed",
                    error={
                        "code": "service_stopped",
                        "message": "Service stopped before completion",
                    },
                    completed_at=int(time.time()),
                )
        except Exception as exc:
            import logging

            logging.getLogger("da3").exception("Background job failed")
            self.clear_job_data(identifier)
            self.update(
                identifier,
                status="failed",
                stage="failed",
                error={"message": str(exc), "code": getattr(exc, "code", "job_failed")},
                completed_at=int(time.time()),
            )
        finally:
            pending = [
                task
                for task in (acquire, cancelled)
                if task is not None and not task.done()
            ]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if acquired:
                self.slot.release()
            elif (
                acquire is not None
                and acquire.done()
                and not acquire.cancelled()
                and acquire.result()
            ):
                self.slot.release()

    def clear_job_data(self, identifier):
        directory = self.root / identifier
        for path in directory.iterdir():
            if path.name == "status.json":
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        self.untrack_tree(directory)
        status = directory / "status.json"
        if status.is_file():
            try:
                self.track_file(identifier, status)
            except APIError:
                pass

    def get(self, identifier):
        with self.lock:
            if identifier not in self.records:
                raise APIError("Unknown job", 404, "not_found")
            return dict(self.records[identifier])

    def cancel(self, identifier):
        record = self.get(identifier)
        if record["status"] in ("queued", "in_progress"):
            self.cancelled.add(identifier)
            self.cancel_events[identifier].set()
            self.update(identifier, cancel_requested=True)
        return self.get(identifier)

    def cleanup(self):
        cutoff = time.time() - 24 * 3600
        with self.lock:
            records = list(self.records.items())
        for identifier, record in records:
            if record["status"] not in ("completed", "failed", "cancelled"):
                continue
            if record.get("completed_at", record.get("updated_at", 0)) >= cutoff:
                continue
            path = self.root / identifier
            shutil.rmtree(path, ignore_errors=True)
            self.untrack_tree(path)
            with self.lock:
                self.records.pop(identifier, None)
                self.tasks.pop(identifier, None)
                self.cancel_events.pop(identifier, None)
                self.cancelled.discard(identifier)
        # Writers reserve paths before persistence. Rebuilding from a directory
        # scan would drop those reservations while files are still being written.
