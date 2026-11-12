import concurrent.futures
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fastda3.protocol import APIError, parse_request


class ProtocolTests(unittest.TestCase):
    def test_openai_multiview_and_pose_options(self):
        identity = np.eye(4, dtype=np.float32)
        intrinsic = np.array([[500, 0, 32], [0, 500, 24], [0, 0, 1]], dtype=np.float32)
        image = "data:image/png;base64,AA=="
        request = {
            "model": "da3",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "image_url", "image_url": {"url": image}},
                    ],
                }
            ],
            "use_ray_pose": True,
            "extrinsics": [identity.tolist(), identity.tolist()],
            "intrinsics": [intrinsic.tolist(), intrinsic.tolist()],
        }
        options = parse_request(request, chat=True)
        self.assertEqual(options.task, "multiview")
        self.assertEqual(len(options.images), 2)
        self.assertTrue(options.use_ray_pose)
        self.assertEqual(options.extrinsics.shape, (2, 4, 4))

    def test_streaming_defaults_and_size_options(self):
        options = parse_request(
            {
                "model": "da3",
                "image_url": "https://example.test/video.mp4",
                "task": "streaming",
            },
            chat=False,
        )
        self.assertEqual(options.process_res, 504)
        self.assertTrue(options.background)
        self.assertIsNone(options.streaming["chunk_size"])

    def test_openai_extra_body_options(self):
        options = parse_request(
            {
                "model": "da3",
                "image": "https://example.test/frame.png",
                "extra_body": {"task": "depth", "process_res": 280},
            },
            chat=False,
        )
        self.assertEqual(options.process_res, 280)

    def test_bad_pose_metadata_is_rejected(self):
        with self.assertRaises(APIError):
            parse_request(
                {
                    "image": "data:image/png;base64,AA==",
                    "extrinsics": [np.eye(4).tolist()],
                },
                chat=False,
            )

    def test_task_must_be_a_string(self):
        with self.assertRaises(APIError):
            parse_request(
                {"task": [], "image": "https://example.test/a.png"}, chat=False
            )


class StreamingMathTests(unittest.TestCase):
    def test_overlap_crop_preserves_camera_pixel_coordinates(self):
        from fastda3.streaming import center_crop_prediction

        result = {
            "depth": np.arange(8 * 10, dtype=np.float32).reshape(1, 8, 10),
            "confidence": np.ones((1, 8, 10)),
            "sky": np.zeros((1, 8, 10), bool),
            "intrinsics": np.array([[[10.0, 0, 5], [0, 10, 4], [0, 0, 1]]]),
        }
        cropped = center_crop_prediction(result, 4, 6)
        np.testing.assert_array_equal(cropped["depth"], result["depth"][:, 2:6, 2:8])
        np.testing.assert_allclose(cropped["intrinsics"][0, :2, 2], [3, 2])
        np.testing.assert_allclose(result["intrinsics"][0, :2, 2], [5, 4])

    @classmethod
    def setUpClass(cls):
        from fastda3.streaming import (
            backproject,
            chunk_ranges,
            compose,
            inverse,
            resize_intrinsics,
            transformed_cameras,
        )

        cls.backproject = staticmethod(backproject)
        cls.chunk_ranges = staticmethod(chunk_ranges)
        cls.compose = staticmethod(compose)
        cls.inverse = staticmethod(inverse)
        cls.transformed_cameras = staticmethod(transformed_cameras)
        cls.resize_intrinsics = staticmethod(resize_intrinsics)

    def test_chunk_ranges_cover_sequence_and_overlap(self):
        ranges = self.chunk_ranges(20, 8, 2)
        self.assertEqual(ranges, [(0, 8), (6, 14), (12, 20)])
        covered = set(index for start, stop in ranges for index in range(start, stop))
        self.assertEqual(covered, set(range(20)))

    def test_similarity_transform_inverse_and_camera(self):
        angle = 0.4
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        transform = (1.7, rotation, np.array([2.0, -1.0, 0.5], dtype=np.float32))
        identity = self.compose(transform, self.inverse(transform))
        self.assertAlmostEqual(identity[0], 1.0, places=5)
        np.testing.assert_allclose(identity[1], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(identity[2], 0, atol=1e-5)
        ext = np.eye(4, dtype=np.float32)[None, :3]
        moved, c2w = self.transformed_cameras(ext, transform)
        np.testing.assert_allclose(
            moved[0, :3, 3], -rotation.T @ transform[2], atol=1e-5
        )
        moved_full = np.eye(4, dtype=np.float32)[None]
        moved_full[:, :3] = moved
        np.testing.assert_allclose(np.linalg.inv(moved_full)[0], c2w[0], atol=1e-5)

    def test_backprojection_and_world_to_camera(self):
        depth = np.ones((1, 2, 2), dtype=np.float32)
        intrinsic = np.broadcast_to(np.eye(3, dtype=np.float32), (1, 3, 3))
        extrinsic = np.eye(4, dtype=np.float32)[None, :3]
        points = self.backproject(depth, intrinsic, extrinsic)
        np.testing.assert_allclose(points[0, 1, 1], [1, 1, 1])

    def test_intrinsics_follow_video_frame_resize(self):
        intrinsic = np.array(
            [[[100, 0, 320], [0, 120, 240], [0, 0, 1]]], dtype=np.float32
        )
        resized = self.resize_intrinsics(intrinsic, [(640, 480)], [(320, 240)])
        np.testing.assert_allclose(resized[0], [[50, 0, 160], [0, 60, 120], [0, 0, 1]])


class JobTests(unittest.IsolatedAsyncioTestCase):
    async def test_queued_job_cancels_without_waiting_for_gpu(self):
        from fastda3.streaming import JobManager

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SimpleNamespace(root=root / "results")
            args = SimpleNamespace(max_job_disk_gb=1, served_model_name="da3")
            scheduler = SimpleNamespace(args=args, store=store)
            manager = JobManager(scheduler, root / "jobs")
            options = SimpleNamespace(task="depth")
            record = manager.create(options)
            manager.cancel(record["id"])
            await manager.tasks[record["id"]]
            self.assertEqual(manager.get(record["id"])["status"], "cancelled")
            self.assertEqual(manager.slot._value, 1)

    async def test_conditioned_streaming_exports_all_windows(self):
        from PIL import Image
        from fastda3.streaming import JobManager, StreamingRunner

        for align_scale in (True, False):
            with (
                self.subTest(align_to_input_ext_scale=align_scale),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                store = SimpleNamespace(root=root / "results")
                store.root.mkdir()
                args = SimpleNamespace(
                    max_job_disk_gb=1,
                    served_model_name="da3",
                    streaming_repo=str(root),
                    salad_checkpoint=str(root / "salad.ckpt"),
                )
                scheduler = SimpleNamespace(
                    args=args,
                    store=store,
                    cpu_pool=concurrent.futures.ThreadPoolExecutor(max_workers=2),
                )
                scheduler.jobs = JobManager(scheduler, root / "jobs")
                runner = scheduler.streaming = StreamingRunner(scheduler, args)
                runner.availability = lambda: {"available": True}

                async def decode(source):
                    return Image.new("RGB", (2, 2), (30, 80, 120))

                async def infer(paths, options, extrinsics=None, intrinsics=None):
                    n = len(paths)
                    return {
                        "depth": np.ones((n, 2, 2), dtype=np.float32),
                        "confidence": np.ones((n, 2, 2), dtype=np.float32) * 2,
                        "sky": np.zeros((n, 2, 2), dtype=bool),
                        "intrinsics": intrinsics.copy(),
                        "extrinsics": extrinsics[:, :3].copy(),
                    }

                scheduler.decode_image = decode
                scheduler.infer_scene_paths = infer
                runner.align = lambda *args: (
                    1.0,
                    np.eye(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                )
                ex = np.broadcast_to(np.eye(4, dtype=np.float32), (5, 4, 4)).copy()
                ex[:, 0, 3] = -np.arange(5)
                options = parse_request(
                    {
                        "task": "streaming",
                        "images": ["https://example.test/a.png"] * 5,
                        "process_res": 140,
                        "extrinsics": ex.tolist(),
                        "intrinsics": np.broadcast_to(np.eye(3), (5, 3, 3)).tolist(),
                        "align_to_input_ext_scale": align_scale,
                        "streaming": {
                            "chunk_size": 3,
                            "overlap": 1,
                            "loop_closure": False,
                            "save_pointcloud": False,
                        },
                    },
                    chat=False,
                )
                try:
                    record = scheduler.jobs.create(options)
                    await scheduler.jobs.tasks[record["id"]]
                    completed = scheduler.jobs.get(record["id"])
                    self.assertEqual(
                        completed["status"], "completed", completed.get("error")
                    )
                    manifest = completed["result"]
                    self.assertEqual(manifest["chunk_count"], 2)
                    self.assertFalse(manifest["loop_closure_applied"])
                    with zipfile.ZipFile(
                        store.root / Path(manifest["result_url"]).name
                    ) as bundle:
                        self.assertEqual(
                            len(
                                [n for n in bundle.namelist() if n.startswith("frame_")]
                            ),
                            5,
                        )
                        with np.load(io.BytesIO(bundle.read("cameras.npz"))) as cameras:
                            self.assertEqual(cameras["c2w"].shape, (5, 4, 4))
                            np.testing.assert_allclose(
                                cameras["c2w"][:, 0, 3], np.arange(5)
                            )
                finally:
                    scheduler.cpu_pool.shutdown()


class ArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_image_results_obey_the_shared_job_disk_budget(self):
        from fastda3.artifacts import ResultStore
        from fastda3.streaming import JobManager

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResultStore(root / "results")
            manager = JobManager(
                SimpleNamespace(store=store, args=SimpleNamespace(max_job_disk_gb=1)),
                root / "jobs",
            )
            manager.max_disk_bytes = 700
            store.budget = manager
            writes = []
            try:
                with patch.object(
                    store.pool,
                    "submit",
                    side_effect=lambda fn, *a: writes.append((fn, a)),
                ):
                    store.put_npz({"depth_m": np.ones((2, 2))})
                    store.put_npz({"depth_m": np.ones((2, 2))})
                    with self.assertRaises(APIError) as caught:
                        store.put_npz({"depth_m": np.ones((2, 2))})
                    self.assertEqual(caught.exception.status, 507)
                    reserved = manager.disk_bytes
                    manager.cleanup()
                    self.assertEqual(manager.disk_bytes, reserved)
            finally:
                for fn, args in writes:
                    fn(*args)
                store.pool.shutdown()

    async def test_rejected_upload_preserves_existing_disk_accounting(self):
        from fastda3.artifacts import FileStore

        class Upload:
            filename = "video.mp4"

            def __init__(self, data):
                self.data = data

            async def read(self, count):
                result, self.data = self.data, b""
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "existing.data").write_bytes(b"1234")
            store = FileStore(root, max_bytes=5)
            with self.assertRaises(APIError):
                await store.upload(Upload(b"12"))
            self.assertEqual(store.total_bytes, 4)
            await store.upload(Upload(b"3"))
            self.assertEqual(store.total_bytes, 5)

    async def test_evicted_result_is_available_while_persistence_is_pending(self):
        from fastda3.artifacts import ResultStore

        with tempfile.TemporaryDirectory() as directory:
            store = ResultStore(directory, cache_bytes=1)
            writes = []
            try:
                with patch.object(
                    store.pool,
                    "submit",
                    side_effect=lambda fn, *a: writes.append((fn, a)),
                ):
                    first, _ = store.put_npz({"depth_m": np.ones((2, 2))})
                    store.put_npz({"depth_m": np.zeros((2, 2))})
                    data, path = store.get(first + ".npz")
                    self.assertIsNone(path)
                    with np.load(io.BytesIO(data)) as arrays:
                        np.testing.assert_array_equal(arrays["depth_m"], 1)
            finally:
                for fn, args in writes:
                    fn(*args)
                store.pool.shutdown()


class PoseFallbackTests(unittest.TestCase):
    def test_conditioned_depth_follows_input_baseline_scale(self):
        import torch
        from fastda3.da3_engine import Engine

        predicted = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()[
            :, :3
        ]
        predicted[1, 0, 3] = -2
        target = predicted.copy()
        target[1, 0, 3] = -1
        intrinsic = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 3, 3)).copy()
        result = {
            "depth": np.full((2, 2, 2), 4.0, np.float32),
            "intrinsics": intrinsic,
            "extrinsics": predicted,
            "scale_factor": 1.0,
        }
        aligned = Engine.align_scene(
            SimpleNamespace(),
            result,
            (None, torch.from_numpy(target), torch.from_numpy(intrinsic)),
            True,
        )
        np.testing.assert_allclose(aligned["depth"], 2.0)
        np.testing.assert_allclose(aligned["extrinsics"], target)
        self.assertAlmostEqual(aligned["scale_factor"], 0.5)

    def test_two_view_baseline_scale_and_static_camera_rotation(self):
        from fastda3.da3_engine import align_degenerate_camera_poses

        predicted = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()[
            :, :3
        ]
        predicted[1, 0, 3] = -2
        target = predicted.copy()
        target[1, 0, 3] = -1
        scale, aligned = align_degenerate_camera_poses(predicted, target)
        self.assertAlmostEqual(scale, 0.5, places=5)
        full = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
        full[:, :3] = aligned
        centers = np.linalg.inv(full)[:, :3, 3]
        np.testing.assert_allclose(centers, [[0, 0, 0], [1, 0, 0]], atol=1e-5)

        angle = 0.3
        target = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()[:, :3]
        target[:, :3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        scale, aligned = align_degenerate_camera_poses(predicted[:1], target[:1])
        np.testing.assert_allclose(aligned, target[:1], atol=1e-5)
        self.assertAlmostEqual(scale, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
