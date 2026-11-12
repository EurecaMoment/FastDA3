"""Validated DA3 extensions to OpenAI chat-completion requests."""

import dataclasses
import re

import numpy as np


class APIError(Exception):
    def __init__(self, message, status=400, code="invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def boolean(value, name, default=False):
    if value is None:
        return default
    if not isinstance(value, bool):
        raise APIError(f"{name} must be a boolean")
    return value


def integer(value, name, minimum, maximum, default):
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise APIError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def source(value):
    if isinstance(value, dict):
        if "file_id" in value:
            value = value["file_id"]
        else:
            value = value.get("url")
    if not isinstance(value, str) or not value:
        raise APIError("Image/video sources require a URL, data URL, or file_id")
    if not value.startswith(("http://", "https://", "data:", "file-")):
        raise APIError("Use HTTP(S), a data URL, or an uploaded file_id")
    if value.startswith("file-") and not re.fullmatch(r"file-[0-9a-f]{24}", value):
        raise APIError("Invalid file_id")
    return value


def cameras(extrinsics, intrinsics, count=None):
    if extrinsics is None and intrinsics is None:
        return None, None
    if extrinsics is None or intrinsics is None:
        raise APIError("Pose conditioning requires both extrinsics and intrinsics")
    try:
        ex = np.asarray(extrinsics, dtype=np.float32)
        ix = np.asarray(intrinsics, dtype=np.float32)
    except (ValueError, TypeError, OverflowError) as exc:
        raise APIError("Camera matrices must be finite numeric arrays") from exc
    if ex.ndim != 3 or ex.shape[1:] not in [(3, 4), (4, 4)]:
        raise APIError("extrinsics must have shape [N,3,4] or [N,4,4], world-to-camera")
    if ix.shape != (len(ex), 3, 3) or (count is not None and len(ex) != count):
        raise APIError("Each image needs one extrinsic and one 3x3 intrinsic matrix")
    if not np.isfinite(ex).all() or not np.isfinite(ix).all():
        raise APIError("Camera matrices must be finite")
    if ex.shape[1:] == (3, 4):
        full = np.broadcast_to(np.eye(4, dtype=np.float32), (len(ex), 4, 4)).copy()
        full[:, :3] = ex
        ex = full
    if not np.allclose(ex[:, 3], [0, 0, 0, 1], atol=1e-4):
        raise APIError("Extrinsic homogeneous rows must be [0,0,0,1]")
    rotation = ex[:, :3, :3]
    if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=0.02):
        raise APIError("Extrinsic rotations must be orthonormal")
    if np.any(np.linalg.det(rotation) < 0.98):
        raise APIError("Extrinsic rotations must be right-handed")
    if np.any(ix[:, 0, 0] <= 0) or np.any(ix[:, 1, 1] <= 0):
        raise APIError("Camera focal lengths must be positive")
    if not np.allclose(ix[:, 2], [0, 0, 1], atol=1e-4):
        raise APIError("Intrinsic last rows must be [0,0,1]")
    return ex, ix


@dataclasses.dataclass
class Options:
    task: str
    images: list
    video: str | None
    process_res: int
    use_ray_pose: bool
    ref_view_strategy: str
    extrinsics: object = None
    intrinsics: object = None
    align_to_input_ext_scale: bool = True
    return_inline: bool = False
    include_arrays: bool = False
    background: bool = False
    stream: bool = False
    streaming: dict = dataclasses.field(default_factory=dict)

    @property
    def independent(self):
        return (
            self.task == "depth"
            and len(self.images) == 1
            and not self.use_ray_pose
            and self.extrinsics is None
            and self.process_res == 1024
        )


def parse_request(body, *, chat, model="da3", max_views=64):
    if not isinstance(body, dict):
        raise APIError("Request body must be a JSON object")
    if body.get("model", model) != model:
        raise APIError("Unknown model", 404, "model_not_found")
    extra_body = body.get("extra_body", {})
    settings = body.get("da3", {})
    if not isinstance(extra_body, dict) or not isinstance(settings, dict):
        raise APIError("extra_body and da3 must be objects")
    settings = {**body, **extra_body, **settings}
    images = []
    videos = []
    if chat:
        messages = body.get("messages", [])
        if not isinstance(messages, list):
            raise APIError("messages must be an array")
        for message in messages:
            if not isinstance(message, dict):
                raise APIError("Each message must be an object")
            content = message.get("content", [])
            if isinstance(content, str):
                continue
            if not isinstance(content, list):
                raise APIError("Message content must be text or a content-part array")
            for part in content:
                if not isinstance(part, dict):
                    raise APIError("Content parts must be objects")
                if part.get("type") in ("image_url", "input_image"):
                    images.append(source(part.get("image_url", part.get("file_id"))))
                elif part.get("type") == "image_file":
                    images.append(source(part.get("image_file")))
                elif part.get("type") in ("video_url", "input_video"):
                    videos.append(source(part.get("video_url", part.get("file_id"))))
    extra_images = settings.get("images")
    if extra_images is not None:
        if images or not isinstance(extra_images, list):
            raise APIError("Specify images once, as an array")
        images = [source(value) for value in extra_images]
    if not chat and not images:
        single = settings.get("image_url", settings.get("image"))
        if single is not None:
            images = [source(single)]
    video = settings.get("video_file_id", settings.get("video_url"))
    if video is not None:
        videos.append(source(video))
    if len(videos) > 1 or (videos and images):
        raise APIError("Provide one video or an image sequence")
    task = settings.get(
        "task", "streaming" if videos else "multiview" if len(images) > 1 else "depth"
    )
    if not isinstance(task, str):
        raise APIError("task must be a string")
    task = {
        "video": "streaming",
        "da3-streaming": "streaming",
        "joint": "multiview",
    }.get(task, task)
    if task not in ("depth", "multiview", "streaming"):
        raise APIError("task must be depth, multiview, or streaming")
    if videos and task != "streaming":
        raise APIError("Video inputs require task=streaming")
    if videos and not videos[0].startswith(("http://", "https://", "file-")):
        raise APIError("Videos require an uploaded file_id or HTTP(S) URL")
    if not videos and not images:
        raise APIError("An image, image sequence, or video is required")
    if task != "streaming" and len(images) > max_views:
        raise APIError(
            f"At most {max_views} joint views; use task=streaming for longer sequences"
        )
    if task == "depth" and len(images) != 1:
        raise APIError("Multiple images use task=multiview or streaming")
    if boolean(settings.get("infer_gs"), "infer_gs"):
        raise APIError("Gaussian export is not exposed by this depth/pose service")
    process_res = integer(
        settings.get("process_res"),
        "process_res",
        140,
        1024,
        504 if task == "streaming" else 1024,
    )
    strategy = settings.get("ref_view_strategy", "saddle_balanced")
    if strategy not in ("first", "middle", "saddle_balanced", "saddle_sim_range"):
        raise APIError("Unsupported ref_view_strategy")
    ex, ix = cameras(
        settings.get("extrinsics"),
        settings.get("intrinsics"),
        None if task == "streaming" else len(images),
    )
    streaming = settings.get("streaming", {})
    if not isinstance(streaming, dict):
        raise APIError("streaming must be an object")
    if task == "streaming":
        allowed = {
            "chunk_size",
            "overlap",
            "fps",
            "loop_closure",
            "loop_threshold",
            "loop_min_separation",
            "max_loop_pairs",
            "max_frames",
            "save_depth",
            "save_pointcloud",
            "point_stride",
            "max_align_points",
        }
        unknown = set(streaming) - allowed
        if unknown:
            raise APIError("Unknown streaming options: " + ", ".join(sorted(unknown)))
        for name, lo, hi, default in [
            ("chunk_size", 2, 64, None),
            ("overlap", 1, 32, None),
            ("max_frames", 2, 100000, 100000),
            ("max_loop_pairs", 0, 1000, 100),
            ("loop_min_separation", 1, 100000, 25),
            ("point_stride", 1, 100, 8),
            ("max_align_points", 1000, 1000000, 100000),
        ]:
            streaming[name] = integer(streaming.get(name), name, lo, hi, default)
        for name, default in [
            ("loop_closure", True),
            ("save_depth", True),
            ("save_pointcloud", True),
        ]:
            streaming[name] = boolean(streaming.get(name), name, default)
        fps = streaming.get("fps", 2.0)
        threshold = streaming.get("loop_threshold", 0.85)
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not 0.1 <= fps <= 60
        ):
            raise APIError("streaming.fps must be in [0.1,60]")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 < threshold <= 1
        ):
            raise APIError("loop_threshold must be in (0,1]")
        streaming.update(fps=float(fps), loop_threshold=float(threshold))
    return Options(
        task=task,
        images=images,
        video=videos[0] if videos else None,
        process_res=process_res,
        use_ray_pose=boolean(settings.get("use_ray_pose"), "use_ray_pose"),
        ref_view_strategy=strategy,
        extrinsics=ex,
        intrinsics=ix,
        align_to_input_ext_scale=boolean(
            settings.get("align_to_input_ext_scale"), "align_to_input_ext_scale", True
        ),
        return_inline=boolean(settings.get("return_inline"), "return_inline"),
        include_arrays=boolean(settings.get("include_arrays"), "include_arrays"),
        background=boolean(
            settings.get("background"), "background", task == "streaming"
        ),
        stream=boolean(body.get("stream"), "stream"),
        streaming=streaming,
    )
