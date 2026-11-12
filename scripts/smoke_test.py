"""Exercise a running FastDA3 server; includes GPU inference and video export checks."""

import argparse
import asyncio
import base64
import collections
import io
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import cv2
import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def image_url(payload):
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def completion(response):
    response.raise_for_status()
    return json.loads(response.json()["choices"][0]["message"]["content"])


async def wait_job(client, record, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get("/v1/jobs/" + record["id"])
        response.raise_for_status()
        result = response.json()
        if result["status"] in ("completed", "failed", "cancelled"):
            return result
        await asyncio.sleep(0.25)
    raise TimeoutError("Video job did not finish before the test deadline")


async def check_archive(
    client, record, expected_frames, expected_poses=None, pointcloud=True
):
    assert record["status"] == "completed", record.get("error")
    response = await client.get(record["result"]["result_url"])
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["frame_count"] == expected_frames
        frames = sorted(name for name in bundle.namelist() if name.startswith("frame_"))
        assert len(frames) == expected_frames
        for name in frames:
            with np.load(io.BytesIO(bundle.read(name)), allow_pickle=False) as arrays:
                assert arrays["depth_m"].ndim == 2
                assert np.isfinite(arrays["depth_m"]).all()
                assert np.isfinite(arrays["intrinsics"]).all()
                assert arrays["extrinsics"].shape == (3, 4)
        with np.load(
            io.BytesIO(bundle.read("cameras.npz")), allow_pickle=False
        ) as arrays:
            cameras = arrays["c2w"]
            assert cameras.shape == (expected_frames, 4, 4)
            assert np.isfinite(cameras).all()
            if expected_poses is not None:
                np.testing.assert_allclose(cameras, expected_poses, atol=1e-5)
        if pointcloud:
            raw = bundle.read("pointcloud.ply")
            header, vertices = raw.split(b"end_header\n", 1)
            assert f"element vertex {manifest['point_count']}\n".encode() in header
            assert manifest["point_count"] > 0
            assert len(vertices) == manifest["point_count"] * 15
            xyzrgb = np.frombuffer(
                vertices, dtype=[("xyz", "<f4", (3,)), ("rgb", "u1", (3,))]
            )
            assert np.isfinite(xyzrgb["xyz"]).all()
            assert xyzrgb["rgb"].max() > 0
    return {**manifest, "archive_bytes": len(response.content)}


async def run(args, report):
    secret = os.environ.get("DA3_API_KEY", "")
    headers = {"Authorization": "Bearer " + secret} if secret else {}
    payload = args.image.read_bytes()
    source = image_url(payload)
    with Image.open(io.BytesIO(payload)) as image:
        width, height = image.size
    focal = 0.9 * width
    intrinsic = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    views = [source]
    for path in sorted(args.image.parent.glob("*.png")):
        if path != args.image:
            views.append(image_url(path.read_bytes()))
        if len(views) == 3:
            break
    while len(views) < 3:
        views.append(source)
    async with httpx.AsyncClient(
        base_url=args.base_url,
        headers=headers,
        timeout=httpx.Timeout(180, connect=10),
        trust_env=False,
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=32),
    ) as client:
        response = await client.get("/health")
        response.raise_for_status()
        report["health_before"] = response.json()
        assert report["health_before"]["ready"]
        if secret:
            unauthorized = await client.get("/v1/models", headers={"Authorization": ""})
            assert unauthorized.status_code == 401
        response = await client.get("/v1/models")
        response.raise_for_status()
        assert response.json()["data"][0]["id"] == args.model
        malformed = await client.post(
            "/v1/depth/estimations", json={"image_url": source, "task": []}
        )
        assert malformed.status_code == 400
        report["tests"]["authentication_and_validation"] = "passed"

        async def depth_request(url=source):
            start = time.perf_counter()
            response = await client.post(
                "/v1/depth/estimations", json={"model": args.model, "image_url": url}
            )
            response.raise_for_status()
            return time.perf_counter() - start, response.json()

        elapsed, single = await depth_request()
        response = await client.get(single["depth_npz_url"])
        response.raise_for_status()
        with np.load(io.BytesIO(response.content), allow_pickle=False) as result:
            depth = result["depth_m"]
            assert np.isfinite(depth).all()
            assert list(depth.shape) == single["depth_shape"]
            if args.baseline:
                with np.load(args.baseline, allow_pickle=False) as before:
                    np.testing.assert_array_equal(depth, before["depth_m"])
        report["tests"]["single_image"] = {
            "latency_seconds": elapsed,
            "shape": single["depth_shape"],
            "timing": single["timing"],
            "baseline_pixel_identical": bool(args.baseline),
        }

        content = [{"type": "image_url", "image_url": {"url": url}} for url in views]
        multi = completion(
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "task": "multiview",
                    "process_res": 504,
                },
            )
        )
        assert multi["num_views"] == 3 and multi["depth_shape"][0] == 3
        assert np.isfinite(multi["extrinsics"]).all()
        report["tests"]["multiview"] = {"num_views": 3, "shape": multi["depth_shape"]}

        poses = np.broadcast_to(np.eye(4, dtype=np.float32), (3, 4, 4)).copy()
        poses[1, 0, 3] = -1
        poses[2, 1, 3] = -1
        ray = completion(
            await client.post(
                "/v1/chat/completions",
                json={
                    "model": args.model,
                    "messages": [{"role": "user", "content": content}],
                    "process_res": 504,
                    "use_ray_pose": True,
                    "extrinsics": poses.tolist(),
                    "intrinsics": [intrinsic] * 3,
                },
            )
        )
        assert ray["use_ray_pose"] and ray["pose_conditioned"]
        np.testing.assert_allclose(ray["extrinsics"], poses[:, :3], atol=1e-5)
        report["tests"]["pose_conditioned_ray_pose"] = {
            "shape": ray["depth_shape"],
            "passed": True,
        }

        small = await client.post(
            "/v1/depth/estimations",
            json={
                "model": args.model,
                "image_url": source,
                "process_res": 140,
                "use_ray_pose": True,
            },
        )
        small.raise_for_status()
        assert small.json()["task"] == "depth" and small.json()["use_ray_pose"]
        report["tests"]["single_view_ray_pose"] = "passed"

        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": args.model,
                "messages": [{"role": "user", "content": content[:1]}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        response.raise_for_status()
        events = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(value) for value in events[:-1]]
        assert chunks[-1]["usage"]["total_tokens"] == 0
        assert json.loads(chunks[1]["choices"][0]["delta"]["content"])["num_views"] == 1
        report["tests"]["openai_sse"] = "passed"

        with Image.open(io.BytesIO(payload)) as image:
            temporary = io.BytesIO()
            image.convert("RGB").resize((997, 611)).save(temporary, format="PNG")
        eager = await asyncio.gather(
            *(depth_request(image_url(temporary.getvalue())) for _ in range(3))
        )
        assert all(row[1]["inference_shape"] == [630, 1022] for row in eager)
        report["tests"]["uncaptured_aspect_ratio"] = {
            "shape": [630, 1022],
            "batch_sizes": [row[1]["timing"]["batch_size"] for row in eager],
        }

        for _ in range(2):
            await depth_request()
        serial = [await depth_request() for _ in range(6)]
        report["tests"]["throughput_serial"] = {
            "requests": len(serial),
            "images_per_second": len(serial) / sum(row[0] for row in serial),
        }
        semaphore = asyncio.Semaphore(args.concurrency)

        async def limited():
            async with semaphore:
                return await depth_request()

        start = time.perf_counter()
        results = await asyncio.gather(*(limited() for _ in range(args.requests)))
        elapsed = time.perf_counter() - start
        report["tests"]["throughput_concurrent"] = {
            "requests": len(results),
            "concurrency": args.concurrency,
            "elapsed_seconds": elapsed,
            "images_per_second": len(results) / elapsed,
            "latency_p50_seconds": float(
                np.percentile([row[0] for row in results], 50)
            ),
            "latency_p95_seconds": float(
                np.percentile([row[0] for row in results], 95)
            ),
            "batch_sizes": dict(
                collections.Counter(row[1]["timing"]["batch_size"] for row in results)
            ),
        }

        with tempfile.TemporaryDirectory(prefix="fastda3-smoke-") as directory:
            video = Path(directory) / "video.mp4"
            frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (w, h)
            )
            assert writer.isOpened()
            for index in range(52):
                writer.write(np.roll(frame, index % 5, axis=1))
            writer.release()
            response = await client.post(
                "/v1/files",
                data={"purpose": "user_data"},
                files={"file": ("video.mp4", video.read_bytes(), "video/mp4")},
            )
        response.raise_for_status()
        file_id = response.json()["id"]
        response = await client.post(
            "/v1/video/estimations",
            json={
                "model": args.model,
                "video_file_id": file_id,
                "process_res": 280,
                "streaming": {
                    "fps": 2,
                    "chunk_size": 8,
                    "overlap": 2,
                    "loop_closure": True,
                    "max_loop_pairs": 4,
                    "point_stride": 16,
                },
            },
        )
        response.raise_for_status()
        # A video job and an ordinary image request share the GPU queue.
        record, during_video = await asyncio.gather(
            wait_job(client, response.json()), depth_request()
        )
        video_manifest = await check_archive(client, record, 26)
        assert video_manifest["chunk_count"] == 4
        assert (
            video_manifest["loop_constraints"] > 0
            and video_manifest["loop_closure_applied"]
        )
        report["tests"]["streaming_video"] = video_manifest
        report["tests"]["single_during_video"] = {"latency_seconds": during_video[0]}

        camera_poses = np.broadcast_to(np.eye(4, dtype=np.float32), (26, 4, 4)).copy()
        camera_poses[:, 0, 3] = np.arange(26) * 0.05
        response = await client.post(
            "/v1/video/estimations",
            json={
                "model": args.model,
                "video_file_id": file_id,
                "process_res": 140,
                "extrinsics": np.linalg.inv(camera_poses).tolist(),
                "intrinsics": [intrinsic] * 26,
                "streaming": {
                    "fps": 2,
                    "chunk_size": 4,
                    "overlap": 1,
                    "loop_closure": True,
                    "point_stride": 16,
                },
            },
        )
        response.raise_for_status()
        record = await wait_job(client, response.json())
        conditioned_manifest = await check_archive(client, record, 26, camera_poses)
        assert conditioned_manifest["chunk_count"] == 9
        assert not conditioned_manifest["loop_closure_applied"]
        report["tests"]["conditioned_streaming_video"] = conditioned_manifest

        ex = np.broadcast_to(np.eye(4, dtype=np.float32), (5, 4, 4)).copy()
        ex[:, :2, 3] = -np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]])
        response = await client.post(
            "/v1/depth/estimations",
            json={
                "model": args.model,
                "task": "streaming",
                "images": [source] * 5,
                "process_res": 140,
                "extrinsics": ex.tolist(),
                "intrinsics": [intrinsic] * 5,
                "align_to_input_ext_scale": False,
                "streaming": {
                    "chunk_size": 3,
                    "overlap": 1,
                    "loop_closure": False,
                    "save_pointcloud": False,
                },
            },
        )
        response.raise_for_status()
        record = await wait_job(client, response.json())
        report["tests"]["streaming_without_input_scale"] = await check_archive(
            client, record, 5, pointcloud=False
        )

        response = await client.post(
            "/v1/video/estimations",
            json={
                "model": args.model,
                "video_file_id": file_id,
                "process_res": 140,
                "streaming": {"loop_closure": False},
            },
        )
        response.raise_for_status()
        pending = response.json()
        response = await client.post("/v1/jobs/" + pending["id"] + "/cancel")
        response.raise_for_status()
        cancelled = await wait_job(client, pending)
        assert cancelled["status"] == "cancelled", cancelled
        report["tests"]["job_cancellation"] = "passed"

        response = await client.get("/health")
        response.raise_for_status()
        report["health_after"] = response.json()
        assert report["health_after"]["ready"]
        assert report["health_after"]["failures"] == 0
        assert report["health_after"]["optimizations"]["persistence_failures"] == 0
        assert (
            report["health_after"]["gpu_memory_peak_gb"]
            <= report["health_after"]["gpu_memory_limit_gb"]
        )
        report["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8035")
    parser.add_argument("--model", default="da3")
    parser.add_argument(
        "--image", type=Path, default=ROOT / "assets/examples/SOH/000.png"
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "runtime/smoke-report.json"
    )
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    report = {
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tests": {},
        "passed": False,
    }
    try:
        asyncio.run(run(args, report))
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
