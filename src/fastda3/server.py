"""OpenAI-style DA3 service with independent batching, joint scenes and video jobs."""

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import logging
import os
import secrets
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image, ImageOps

from .artifacts import FileStore, ResultStore
from .da3_engine import Engine, calibrate
from .protocol import APIError, parse_request
from .streaming import JobManager, StreamingRunner
from . import __version__

log = logging.getLogger("da3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
VERSION = __version__


@dataclass
class Job:
    tensor: object
    original_size: object
    queued: float
    future: object
    options: object


class Scheduler:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.ready = False
        self.cpu_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.cpu_workers, thread_name_prefix="da3-cpu"
        )
        self.decode_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="da3-decode"
        )
        self.gpu_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="da3-gpu"
        )
        self.gpu_lock = asyncio.Lock()
        self.queues = OrderedDict()
        self.scene_queue = deque()
        self.signal = asyncio.Event()
        self.active = 0
        self.completed = 0
        self.failures = 0
        self.batches = 0
        self.latencies = deque(maxlen=10000)
        self.batch_counts = {}
        self.fallback_batches = 0
        self.process_gpu_gb = 0.0
        self.peak_process_gpu_gb = 0.0
        self.store = ResultStore(args.output_dir, args.result_cache_mb * 1024**2)
        self.files = FileStore(args.files_dir, args.max_file_gb * 1024**3)
        self.jobs = JobManager(self, args.jobs_dir)
        self.store.budget = self.jobs
        self.streaming = StreamingRunner(self, args)
        self.graph_sizes = []
        self.inflight = set()
        self.running = True
        self.http = httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(60, connect=10),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            follow_redirects=True,
        )
        self.preprocess_cache = OrderedDict()
        self.preprocess_cache_bytes = 0
        self._last_guard = 0
        self.late_capture_shapes = set()

    async def initialize(self):
        cv2.setNumThreads(1)
        self.guard_task = asyncio.create_task(self._guard())
        await asyncio.get_running_loop().run_in_executor(self.gpu_pool, self._load)
        self.ready = True
        self.worker_task = asyncio.create_task(self._worker())
        self.cleanup_task = asyncio.create_task(self._cleanup())

    def _load(self):
        self.engine = Engine(
            self.args.model_dir,
            allocator_gb=self.args.max_gpu_memory_gb - 4,
            fused_rope=not self.args.disable_fused_rope,
            cudnn_benchmark=self.args.cudnn_benchmark,
            bf16_convs=self.args.bf16_convs,
        )
        shapes = []
        for h, w in [
            (574, 1022),
            (770, 1022),
            (1022, 574),
            (1022, 770),
            (1022, 1022),
            (686, 1022),
        ]:
            for b in self.args.graph_batches:
                if (
                    b <= self.args.max_batch_size
                    and b * h * w <= self.args.max_batch_pixels
                ):
                    shapes.append((b, 1, 3, h, w))
        for shape in shapes:
            self.engine.warmup(shape)
        for shape in sorted(shapes, key=lambda s: s[0] * s[-1] * s[-2], reverse=True):
            try:
                log.info("Graph ready: %s", self.engine.capture(shape, warmed=True))
                self.graph_sizes.append((shape[0], shape[-2], shape[-1]))
            except torch.cuda.OutOfMemoryError:
                log.exception("Skipping graph that exceeds memory budget: %s", shape)
                self.engine.clear_eager_cache()
        self.graph_sizes.sort()
        self.engine.clear_eager_cache()
        log.info(
            "DA3 %s ready, %s graphs, GPU %s",
            VERSION,
            len(self.graph_sizes),
            torch.cuda.get_device_name(0),
        )

    async def decode_image(self, source):
        if source.startswith("file-"):
            path, _ = self.files.get(source)
            if path.stat().st_size > 24 * 1024**2:
                raise APIError("Image file exceeds 24 MiB", 413, "image_too_large")
            data = await asyncio.to_thread(path.read_bytes)
        elif source.startswith("data:image/"):
            if len(source) > 32 * 1024**2:
                raise APIError("Image data URL exceeds 32 MiB", 413, "image_too_large")

            def decode():
                try:
                    header, encoded = source.split(",", 1)
                    if ";base64" not in header:
                        raise ValueError("Use base64 data URLs")
                    return base64.b64decode(encoded, validate=True)
                except (ValueError, IndexError) as exc:
                    raise APIError("Invalid image data URL") from exc

            data = await asyncio.get_running_loop().run_in_executor(
                self.decode_pool, decode
            )
        elif source.startswith(("https://", "http://")):
            try:
                chunks = []
                size = 0
                async with self.http.stream("GET", source) as response:
                    response.raise_for_status()
                    async for block in response.aiter_bytes():
                        size += len(block)
                        if size > 24 * 1024**2:
                            raise APIError(
                                "Remote image exceeds 24 MiB", 413, "image_too_large"
                            )
                        chunks.append(block)
                data = b"".join(chunks)
            except httpx.HTTPError as exc:
                raise APIError("Image download failed") from exc
        else:
            raise APIError("Invalid image source")

        def open_image():
            try:
                with Image.open(io.BytesIO(data)) as image:
                    if image.width * image.height > 40_000_000:
                        raise APIError(
                            "Image exceeds 40 million pixels", 413, "image_too_large"
                        )
                    return ImageOps.exif_transpose(image).convert("RGB")
            except APIError:
                raise
            except Exception as exc:
                raise APIError("Invalid image") from exc

        return await asyncio.get_running_loop().run_in_executor(
            self.decode_pool, open_image
        )

    def preprocess(self, image):
        width, height = image.size
        processor = self.engine.model.input_processor
        resized = processor._resize_image(
            image, self.args.process_res, "upper_bound_resize"
        )
        resized = processor._make_divisible_by_resize(resized, 14)
        new_w, new_h = resized.size
        if min(new_w, new_h) < 14:
            raise APIError("Image aspect ratio is too extreme")
        tensor = torch.from_numpy(np.asarray(resized).transpose(2, 0, 1).copy())
        return tensor, (width, height, new_w, new_h)

    async def materialize_video(self, source, identifier):
        if source.startswith("file-"):
            return self.files.get(source)[0]
        if not source.startswith(("http://", "https://")):
            raise APIError("Videos require an uploaded file_id or HTTP(S) URL")
        target = self.jobs.root / identifier / "video.data"
        size = 0
        try:
            async with self.http.stream("GET", source) as response:
                response.raise_for_status()
                with target.open("wb") as output:
                    async for block in response.aiter_bytes(1024 * 1024):
                        self.jobs.check_cancel(identifier)
                        size += len(block)
                        if size > self.files.max_bytes:
                            raise APIError(
                                "Video exceeds configured file limit",
                                413,
                                "file_too_large",
                            )
                        await asyncio.to_thread(output.write, block)
                        self.jobs.track_file(identifier, target)
            return target
        except BaseException:
            target.unlink(missing_ok=True)
            self.jobs.untrack_tree(target)
            raise

    async def _enqueue(self, tensor, size, options, scene=False):
        future = asyncio.get_running_loop().create_future()
        job = Job(tensor, size, time.perf_counter(), future, options)
        if scene:
            self.scene_queue.append(job)
        else:
            key = tuple(tensor.shape)
            self.queues.setdefault(key, deque()).append(job)
        self.signal.set()
        try:
            return await asyncio.wait_for(
                future, self.args.scene_timeout if scene else self.args.request_timeout
            )
        except asyncio.TimeoutError as exc:
            raise APIError(
                "Depth inference timed out", 504, "inference_timeout"
            ) from exc

    async def execute(self, options):
        if self.active >= self.args.max_num_seqs:
            raise APIError(
                "Maximum in-flight requests reached", 429, "rate_limit_exceeded"
            )
        self.active += 1
        started = time.perf_counter()
        try:
            if options.independent:
                image = await self.decode_image(options.images[0])
                tensor, size = await asyncio.get_running_loop().run_in_executor(
                    self.decode_pool, self.preprocess, image
                )
                result = await self._enqueue(tensor, size, options)
            else:
                images = []
                for source in options.images:
                    images.append(await self.decode_image(source))
                prepared = await asyncio.get_running_loop().run_in_executor(
                    self.decode_pool,
                    self.engine.prepare_scene,
                    images,
                    options.extrinsics,
                    options.intrinsics,
                    options.process_res,
                )
                self.check_scene_budget(prepared)
                sizes = [(im.width, im.height) for im in images]
                result = await self._enqueue(prepared, sizes, options, scene=True)
            result["timing"]["server_total_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return result
        finally:
            self.active -= 1

    def check_scene_budget(self, prepared):
        images = prepared[0]
        views, _, height, width = images.shape
        if views * height * width > self.args.max_scene_pixels:
            raise APIError(
                f"Joint scene has {views * height * width} pixels; budget is {self.args.max_scene_pixels}. "
                "Use a lower process_res or smaller streaming.chunk_size.",
                422,
                "scene_memory_budget",
            )

    async def infer_scene_paths(self, paths, options, extrinsics=None, intrinsics=None):
        def prepare():
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            return self.engine.prepare_scene(
                images, extrinsics, intrinsics, options.process_res
            )

        prepared = await asyncio.get_running_loop().run_in_executor(
            self.decode_pool, prepare
        )
        self.check_scene_budget(prepared)
        # Same ordered GPU queue as HTTP scenes; one video window never monopolizes the GPU.
        settings = copy_options(options)
        settings._raw_result = True
        return await self._enqueue(prepared, None, settings, scene=True)

    async def _worker(self):
        loop = asyncio.get_running_loop()
        while self.running:
            await self.signal.wait()
            # Limit pending output buffers, so slow storage cannot grow pinned memory without bound.
            if len(self.inflight) >= 3:
                await asyncio.wait(self.inflight, return_when=asyncio.FIRST_COMPLETED)
            for key in list(self.queues):
                self.queues[key] = deque(
                    job for job in self.queues[key] if not job.future.cancelled()
                )
                if not self.queues[key]:
                    del self.queues[key]
            while self.scene_queue and self.scene_queue[0].future.cancelled():
                self.scene_queue.popleft()
            oldest_single = min(
                (queue[0].queued for queue in self.queues.values()),
                default=float("inf"),
            )
            scene = bool(
                self.scene_queue and self.scene_queue[0].queued <= oldest_single
            )
            if not scene and not self.queues:
                if not self.scene_queue:
                    self.signal.clear()
                    continue
                scene = True
            if scene:
                jobs = [self.scene_queue.popleft()]
            else:
                key = min(self.queues, key=lambda k: self.queues[k][0].queued)
                queue = self.queues[key]
                _, h, w = key
                limit = min(
                    self.args.max_batch_size,
                    max(1, self.args.max_batch_pixels // (h * w)),
                )
                if len(queue) < limit and not self.scene_queue:
                    await asyncio.sleep(self.args.batch_wait_ms / 1000)
                jobs = []
                while queue and len(jobs) < limit:
                    job = queue.popleft()
                    if not job.future.cancelled():
                        jobs.append(job)
                if not queue:
                    del self.queues[key]
            if not self.queues and not self.scene_queue:
                self.signal.clear()
            if not jobs:
                continue
            try:
                async with self.gpu_lock:
                    if scene:
                        options = jobs[0].options

                        def infer_scene():
                            return self.engine.run_scene(
                                jobs[0].tensor,
                                use_ray_pose=options.use_ray_pose,
                                ref_view_strategy=options.ref_view_strategy,
                            )

                        pending = await loop.run_in_executor(self.gpu_pool, infer_scene)
                    else:

                        def infer_single():
                            tensors = (
                                torch.stack([job.tensor for job in jobs])
                                .unsqueeze(1)
                                .pin_memory()
                            )
                            return self.engine.run(
                                tensors, graph=True, asynchronous=True
                            )

                        key = (len(jobs), 1, 3, *jobs[0].tensor.shape[-2:])
                        if key not in self.engine.graphs:
                            self.fallback_batches += 1
                        pending = await loop.run_in_executor(
                            self.gpu_pool, infer_single
                        )
                self.batches += 1
                self.batch_counts[str(len(jobs))] = (
                    self.batch_counts.get(str(len(jobs)), 0) + 1
                )
                finish = asyncio.create_task(self._finish_batch(jobs, pending, scene))
                self.inflight.add(finish)
                finish.add_done_callback(self.inflight.discard)
            except Exception as exc:
                log.exception("GPU batch failed")
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    async with self.gpu_lock:
                        await loop.run_in_executor(
                            self.gpu_pool, self.engine.clear_eager_cache
                        )
                    exc = APIError(
                        "Scene exceeded available GPU memory; lower process_res or chunk_size",
                        422,
                        "scene_memory_budget",
                    )
                for job in jobs:
                    self.failures += 1
                    if not job.future.done():
                        job.future.set_exception(exc)

    async def _finish_batch(self, jobs, pending, scene):
        loop = asyncio.get_running_loop()
        try:
            raw, gpu_ms = await loop.run_in_executor(self.cpu_pool, pending.resolve)

            async def finish(index, job):
                try:
                    item = raw if scene else [array[index] for array in raw]
                    result = await loop.run_in_executor(
                        self.cpu_pool, self._encode, item, job, scene
                    )
                    if getattr(job.options, "_raw_result", False):
                        output = result
                    else:
                        result["timing"] = {
                            "queue_to_result_ms": round(
                                (time.perf_counter() - job.queued) * 1000, 3
                            ),
                            "batch_gpu_ms": round(gpu_ms, 3),
                            "batch_size": len(jobs),
                            "joint_views": len(job.tensor[0]) if scene else 1,
                        }
                        output = result
                        self.latencies.append(result["timing"]["queue_to_result_ms"])
                    self.completed += 1
                    if not job.future.done():
                        job.future.set_result(output)
                except Exception as exc:
                    self.failures += 1
                    log.exception("Output processing failed")
                    if not job.future.done():
                        job.future.set_exception(exc)

            await asyncio.gather(
                *(finish(index, job) for index, job in enumerate(jobs))
            )
        except Exception as exc:
            self.failures += len(jobs)
            for job in jobs:
                if not job.future.done():
                    job.future.set_exception(exc)

    def _encode(self, raw, job, scene):
        result = calibrate(raw)
        if scene:
            result = self.engine.align_scene(
                result, job.tensor, job.options.align_to_input_ext_scale
            )
        if getattr(job.options, "_raw_result", False):
            return result
        depth = result["depth"]
        conf = result["confidence"]
        sky = result["sky"]
        n = depth.shape[0] if scene else 1
        if job.options.include_arrays and depth.size > 1_100_000:
            raise APIError(
                "include_arrays exceeds the JSON array budget; use the NPZ result",
                413,
                "result_too_large",
            )
        estimated = depth.nbytes + conf.nbytes + sky.nbytes
        if job.options.return_inline and estimated > 16 * 1024**2:
            raise APIError(
                "Inline NPZ exceeds 16 MiB; use the download URL",
                413,
                "result_too_large",
            )
        identifier, inline = self.store.put_npz(
            {
                "depth_m": depth,
                "confidence": conf,
                "sky_mask": sky,
                "intrinsics": result["intrinsics"],
                "extrinsics": result["extrinsics"],
            },
            job.options.return_inline,
        )
        h, w = depth.shape[-2:]
        output = {
            "id": identifier,
            "model": self.args.served_model_name,
            "task": job.options.task,
            "units": "meters",
            "is_metric": True,
            "depth_shape": list(depth.shape),
            "inference_shape": [h, w],
            "requested_long_edge": job.options.process_res,
            "num_views": n,
            "depth_range_m": [float(depth.min()), float(depth.max())],
            "depth_npz_url": "/v1/results/" + identifier + ".npz",
            "intrinsics": result["intrinsics"].tolist(),
            "extrinsics": result["extrinsics"].tolist(),
            "extrinsics_convention": "world_to_camera",
            "intrinsics_coordinates": "processed_image",
            "scale_factor": result["scale_factor"],
            "use_ray_pose": job.options.use_ray_pose,
            "pose_conditioned": job.options.extrinsics is not None,
        }
        if scene:
            output["original_sizes"] = [
                [height, width] for width, height in job.original_size
            ]
        else:
            width, height, _, _ = job.original_size
            output["original_size"] = [height, width]
        if inline is not None:
            output["depth_npz_b64"] = base64.b64encode(inline).decode()
        if job.options.include_arrays:
            output["depth_m"] = depth.tolist()
        return output

    async def _guard(self):
        import subprocess

        while self.running:
            try:

                def query():
                    return subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=3,
                    )

                output = await asyncio.to_thread(query)
                for line in output.splitlines():
                    pid, memory = [piece.strip() for piece in line.split(",")]
                    if pid == str(os.getpid()):
                        self.process_gpu_gb = float(memory) * 1024**2 / 1e9
                        self.peak_process_gpu_gb = max(
                            self.peak_process_gpu_gb, self.process_gpu_gb
                        )
                        if self.process_gpu_gb > self.args.max_gpu_memory_gb:
                            log.critical(
                                "GPU process exceeds %.1f GB guard",
                                self.args.max_gpu_memory_gb,
                            )
                            os._exit(75)
            except Exception:
                log.exception("Memory monitor failed")
            await asyncio.sleep(0.5)

    async def _cleanup(self):
        while self.running:
            try:
                await asyncio.to_thread(self._cleanup_storage)
            except OSError:
                log.exception("Artifact cleanup failed")
            await asyncio.sleep(60)

    def _cleanup_storage(self):
        self.store.cleanup()
        self.files.cleanup()
        self.jobs.cleanup()

    def stats(self):
        data = {
            "ready": self.ready,
            "version": VERSION,
            "model": self.args.served_model_name,
            "gpu_index": 0,
            "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "all"),
            "process_res": self.args.process_res,
            "max_num_seqs": self.args.max_num_seqs,
            "max_batch_size": self.args.max_batch_size,
            "max_batch_pixels": self.args.max_batch_pixels,
            "max_scene_pixels": self.args.max_scene_pixels,
            "max_views": self.args.max_views,
            "graph_batches": self.graph_sizes,
            "queued": sum(len(q) for q in self.queues.values()) + len(self.scene_queue),
            "active_requests": self.active,
            "completed": self.completed,
            "failures": self.failures,
            "batches": self.batches,
            "batch_counts": self.batch_counts,
            "eager_batches": self.fallback_batches,
            "gpu_memory_limit_gb": self.args.max_gpu_memory_gb,
            "gpu_memory_process_gb": round(self.process_gpu_gb, 3),
            "gpu_memory_peak_gb": round(self.peak_process_gpu_gb, 3),
            "features": {
                "single_image": True,
                "multiview": True,
                "pose_conditioning": True,
                "ray_pose": True,
                "streaming": self.streaming.availability(),
            },
            "optimizations": {
                "fused_rope": getattr(self.engine, "fused_rope", False),
                "uint8_gpu_normalization": getattr(
                    getattr(self.engine, "raw", None), "gpu_normalization", False
                ),
                "async_result_copy": True,
                "exact_batch_execution": True,
                "result_cache_bytes": self.store.size,
                "persistence_failures": self.store.write_failures,
            },
        }
        if self.latencies:
            data["latency_p95_ms"] = round(float(np.percentile(self.latencies, 95)), 3)
        return data

    async def close(self):
        self.running = False
        tasks = [
            self.worker_task,
            self.guard_task,
            self.cleanup_task,
            *self.jobs.tasks.values(),
        ]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*self.inflight, return_exceptions=True)
        await self.http.aclose()
        self.store.pool.shutdown(wait=True)
        await asyncio.to_thread(self.gpu_pool.shutdown, wait=True)
        self.cpu_pool.shutdown(wait=True)
        self.decode_pool.shutdown(wait=True)


def copy_options(options):
    import copy

    return copy.copy(options)


def chat_response(output, model):
    return {
        "id": "chatcmpl-" + output["id"],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(output, separators=(",", ":")),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def make_app(args):
    scheduler = Scheduler(args)

    @asynccontextmanager
    async def lifespan(app):
        await scheduler.initialize()
        yield
        await scheduler.close()

    app = FastAPI(
        title="DA3 Depth, Geometry and Streaming API",
        version=VERSION,
        lifespan=lifespan,
    )
    app.state.scheduler = scheduler

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error"
                    if exc.status < 500
                    else "server_error",
                    "param": None,
                    "code": exc.code,
                }
            },
            status_code=exc.status,
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "message": str(exc.detail),
                    "type": "invalid_request_error",
                    "code": exc.status_code,
                }
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid request fields",
                    "type": "invalid_request_error",
                    "code": "validation_error",
                }
            },
            status_code=422,
        )

    @app.middleware("http")
    async def authorize(request, call_next):
        if request.url.path.startswith("/v1/") and args.api_key:
            if not secrets.compare_digest(
                request.headers.get("authorization", ""), "Bearer " + args.api_key
            ):
                return JSONResponse(
                    {
                        "error": {
                            "message": "Invalid API key",
                            "type": "authentication_error",
                            "code": "invalid_api_key",
                        }
                    },
                    401,
                )
        raw_length = request.headers.get("content-length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            return JSONResponse({"error": {"message": "Invalid Content-Length"}}, 400)
        if (
            request.url.path == "/v1/files"
            and not raw_length
            and request.headers.get("transfer-encoding")
        ):
            return JSONResponse(
                {"error": {"message": "Content-Length is required for file uploads"}},
                411,
            )
        cap = (
            scheduler.files.max_bytes + 1024**2
            if request.url.path == "/v1/files"
            else 36 * 1024**2
        )
        if length > cap:
            return JSONResponse(
                {
                    "error": {
                        "message": "Request too large",
                        "type": "invalid_request_error",
                        "code": "payload_too_large",
                    }
                },
                413,
            )
        return await call_next(request)

    @app.get("/health")
    async def health():
        return scheduler.stats()

    @app.get("/v1/metrics")
    async def metrics():
        return scheduler.stats()

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": args.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "depth-anything",
                    "capabilities": scheduler.stats()["features"],
                }
            ],
        }

    @app.get("/v1/results/{filename}")
    async def result(filename):
        data, path = scheduler.store.get(filename)
        if data is not None:
            return Response(
                data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        return FileResponse(
            path, media_type="application/octet-stream", filename=filename
        )

    @app.post("/v1/files")
    async def upload(request: Request):
        form = await request.form(max_files=1, max_fields=3)
        file = form.get("file")
        if file is None or not hasattr(file, "read"):
            raise APIError("file is required")
        try:
            return await scheduler.files.upload(file, form.get("purpose", "user_data"))
        finally:
            await file.close()

    @app.get("/v1/files/{identifier}")
    async def file_metadata(identifier):
        return scheduler.files.get(identifier)[1]

    @app.get("/v1/files/{identifier}/content")
    async def file_content(identifier):
        path, info = scheduler.files.get(identifier)
        return FileResponse(path, filename=info["filename"])

    @app.get("/v1/jobs/{identifier}")
    async def job(identifier):
        return scheduler.jobs.get(identifier)

    @app.post("/v1/jobs/{identifier}/cancel")
    async def cancel(identifier):
        return scheduler.jobs.cancel(identifier)

    @app.get("/v1/jobs")
    async def jobs():
        return {"object": "list", "data": list(scheduler.jobs.records.values())[-100:]}

    async def handle(request, chat, force_task=None):
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise APIError("Invalid JSON body") from exc
        if force_task:
            if not isinstance(body, dict):
                raise APIError("Request body must be an object")
            body["task"] = force_task
            for key in ("extra_body", "da3"):
                if isinstance(body.get(key), dict):
                    body[key]["task"] = force_task
        options = parse_request(
            body, chat=chat, model=args.served_model_name, max_views=args.max_views
        )
        stream_options = body.get("stream_options", {})
        if not isinstance(stream_options, dict):
            raise APIError("stream_options must be an object")
        if options.background:
            if (
                options.task == "streaming"
                and not scheduler.streaming.availability()["available"]
            ):
                raise APIError(
                    "DA3-Streaming dependencies are unavailable",
                    503,
                    "streaming_unavailable",
                )
            output = scheduler.jobs.create(options)
            output["status_url"] = "/v1/jobs/" + output["id"]
            output["cancel_url"] = "/v1/jobs/" + output["id"] + "/cancel"
        elif options.task == "streaming":
            output = scheduler.jobs.create(options)
            await scheduler.jobs.tasks[output["id"]]
            output = scheduler.jobs.get(output["id"])
            if output["status"] != "completed":
                raise APIError(
                    str(output.get("error") or output["status"]), 422, "job_failed"
                )
            output = output["result"]
            output["id"] = secrets.token_hex(12)
        else:
            try:
                output = await scheduler.execute(options)
            except (APIError, asyncio.CancelledError):
                raise
            except Exception as exc:
                log.exception("Inference request failed")
                raise APIError(
                    "Inference failed: " + str(exc), 422, "inference_failed"
                ) from exc
        if not chat:
            return output
        response = chat_response(output, args.served_model_name)
        if options.stream:

            async def events():
                common = {key: response[key] for key in ("id", "created", "model")}
                for delta, reason in [
                    ({"role": "assistant"}, None),
                    ({"content": response["choices"][0]["message"]["content"]}, None),
                    ({}, "stop"),
                ]:
                    chunk = {
                        **common,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": reason}
                        ],
                    }
                    yield "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
                if stream_options.get("include_usage"):
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                **common,
                                "object": "chat.completion.chunk",
                                "choices": [],
                                "usage": response["usage"],
                            }
                        )
                        + "\n\n"
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return response

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle(request, True)

    @app.post("/v1/depth/estimations")
    async def depth(request: Request):
        return await handle(request, False)

    @app.post("/v1/video/estimations")
    async def video(request: Request):
        return await handle(request, False, "streaming")

    return app


def arguments(argv=None):
    root = Path(__file__).resolve().parents[2]
    if not (root / "da3_streaming/configs/base_config.yaml").is_file():
        root = Path.cwd()
    root = Path(os.getenv("FASTDA3_ROOT", str(root)))
    data = Path(os.getenv("DA3_DATA_DIR", str(root / "runtime")))
    parser = argparse.ArgumentParser(
        description="FastDA3 OpenAI-style depth, pose and streaming service"
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv(
            "DA3_MODEL_DIR", str(root / "weights/DA3NESTED-GIANT-LARGE-1.1")
        ),
    )
    parser.add_argument("--served-model-name", default="da3")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8035)
    parser.add_argument("--process-res", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument(
        "--graph-batches",
        type=lambda value: [int(v) for v in value.split(",")],
        default=[1, 2, 3, 4],
    )
    parser.add_argument("--batch-wait-ms", type=float, default=3)
    parser.add_argument("--max-batch-pixels", type=int, default=2_500_000)
    parser.add_argument("--max-scene-pixels", type=int, default=2_500_000)
    parser.add_argument("--max-views", type=int, default=64)
    parser.add_argument("--max-gpu-memory-gb", type=float, default=32)
    parser.add_argument("--cpu-workers", type=int, default=12)
    parser.add_argument("--request-timeout", type=float, default=30)
    parser.add_argument("--scene-timeout", type=float, default=180)
    parser.add_argument("--result-cache-mb", type=int, default=256)
    parser.add_argument("--output-dir", default=str(data / "results"))
    parser.add_argument("--jobs-dir", default=str(data / "jobs"))
    parser.add_argument("--files-dir", default=str(data / "files"))
    parser.add_argument("--max-file-gb", type=int, default=8)
    parser.add_argument("--max-job-disk-gb", type=int, default=64)
    parser.add_argument(
        "--streaming-repo",
        default=os.getenv("DA3_STREAMING_REPO", str(root / "da3_streaming")),
    )
    parser.add_argument(
        "--salad-checkpoint",
        default=os.getenv(
            "DA3_SALAD_CHECKPOINT", str(root / "weights/dino_salad.ckpt")
        ),
    )
    parser.add_argument(
        "--dinov2-repo",
        default=os.getenv("DA3_DINOV2_REPO", str(root / "vendor/dinov2")),
    )
    parser.add_argument("--api-key", default=os.getenv("DA3_API_KEY", ""))
    parser.add_argument("--disable-fused-rope", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--bf16-convs", action="store_true")
    args = parser.parse_args(argv)
    if not 4 < args.max_gpu_memory_gb <= 32:
        parser.error("GPU budget must be >4 and <=32 decimal GB")
    if args.process_res != 1024:
        parser.error("Independent-image default must remain 1024")
    for name in (
        "max_num_seqs",
        "max_batch_size",
        "max_batch_pixels",
        "max_scene_pixels",
        "max_views",
        "cpu_workers",
        "request_timeout",
        "scene_timeout",
        "result_cache_mb",
        "max_file_gb",
        "max_job_disk_gb",
    ):
        if getattr(args, name) <= 0:
            parser.error(name.replace("_", "-") + " must be positive")
    if not np.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        parser.error("batch-wait-ms must be finite and nonnegative")
    if not args.graph_batches or any(
        b < 1 or b > args.max_batch_size for b in args.graph_batches
    ):
        parser.error("graph-batches must contain batch sizes from 1 to max-batch-size")
    return args


def main():
    args = arguments()
    uvicorn.run(
        make_app(args),
        host=args.host,
        port=args.port,
        workers=1,
        backlog=1024,
        limit_concurrency=1024,
        access_log=False,
    )


if __name__ == "__main__":
    main()
