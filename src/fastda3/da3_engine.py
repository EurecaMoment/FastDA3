"""Independent-image DA3 inference. Keep metric calibration per image."""

import gc
import logging
import types
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from addict import Dict
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.head_utils import custom_interpolate

log = logging.getLogger("da3.engine")


def depth_only_impl(head, feats, H, W, patch_start_idx):
    """Original main depth head, omitting the unused ray auxiliary branch."""
    batch, _, channels = feats[0].shape
    ph, pw = H // head.patch_size, W // head.patch_size
    resized = []
    for stage, take in enumerate(head.intermediate_layer_idx):
        x = head.norm(feats[take][:, patch_start_idx:])
        x = x.permute(0, 2, 1).reshape(batch, channels, ph, pw)
        x = head.projects[stage](x)
        if head.pos_embed:
            x = head._add_pos_embed(x, W, H)
        resized.append(head.resize_layers[stage](x))
    l1, l2, l3, l4 = [
        getattr(head.scratch, f"layer{i + 1}_rn")(x) for i, x in enumerate(resized)
    ]
    x = head.scratch.refinenet4(l4, size=l3.shape[2:])
    x = head.scratch.refinenet3(x, l3, size=l2.shape[2:])
    x = head.scratch.refinenet2(x, l2, size=l1.shape[2:])
    x = head.scratch.refinenet1(x, l1)
    x = head.scratch.output_conv1(x)
    x = custom_interpolate(
        x,
        (
            int(ph * head.patch_size / head.down_ratio),
            int(pw * head.patch_size / head.down_ratio),
        ),
        mode="bilinear",
        align_corners=True,
    )
    if head.pos_embed:
        x = head._add_pos_embed(x, W, H)
    logits = head.scratch.output_conv2(x).permute(0, 2, 3, 1)
    return {
        "depth": head._apply_activation_single(
            logits[..., :-1], head.activation
        ).squeeze(-1),
        "depth_conf": head._apply_activation_single(
            logits[..., -1], head.conf_activation
        ),
    }


class RawNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.anyview = model.model.da3
        self.metric = model.model.da3_metric
        self.fast_heads = True
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1),
        )
        self.gpu_normalization = True

    def forward(
        self,
        image,
        extrinsics=None,
        intrinsics=None,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ):
        if image.dtype == torch.uint8:
            if self.gpu_normalization:
                from .kernels import normalize_u8

                image = normalize_u8(image)
            else:
                image = (image.float() / 255.0 - self.mean) / self.std
        height, width = image.shape[-2:]
        camera_token = None
        if extrinsics is not None:
            camera_token = self.anyview.cam_enc(extrinsics, intrinsics, (height, width))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats, _ = self.anyview.backbone(
                image,
                cam_token=camera_token,
                export_feat_layers=[],
                ref_view_strategy=ref_view_strategy,
            )
        feats = tuple((a.float(), b.float()) for a, b in feats)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.fast_heads):
            if use_ray_pose:
                depth = self.anyview.head(feats, height, width, patch_start_idx=0)
            else:
                batch, views, tokens, channels = feats[0][0].shape
                flat = [
                    feat[0].reshape(batch * views, tokens, channels) for feat in feats
                ]
                out = depth_only_impl(self.anyview.head, flat, height, width, 0)
                depth = Dict(
                    {
                        key: value.reshape(batch, views, *value.shape[1:])
                        for key, value in out.items()
                    }
                )
        if use_ray_pose:
            depth.ray = depth.ray.float()
            if "ray_conf" in depth:
                depth.ray_conf = depth.ray_conf.float()
            depth = self.anyview._process_ray_pose_estimation(depth, height, width)
        else:
            depth = self.anyview._process_camera_estimation(feats, height, width, depth)
        del feats
        with torch.autocast("cuda", dtype=torch.bfloat16):
            metric_feats, _ = self.metric.backbone(image, export_feat_layers=[])
        metric_feats = tuple((a.float(), b.float()) for a, b in metric_feats)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.fast_heads):
            metric = self.metric.head(metric_feats, height, width, patch_start_idx=0)
        return (
            depth.depth,
            depth.depth_conf,
            metric.depth,
            metric.sky,
            depth.intrinsics,
            depth.extrinsics,
        )


def calibrate(raw):
    """One scale per scene; N views remain in one common metric coordinate system."""
    depth, conf, metric, sky, intrinsics, extrinsics = [x.copy() for x in raw]
    non_sky = sky < 0.3
    if non_sky.sum() <= 10:
        raise ValueError("Insufficient non-sky pixels for metric depth alignment")
    if (~non_sky).sum() > 10:
        metric[~non_sky] = np.quantile(metric[non_sky], 0.99)
    focal = (intrinsics[..., 0, 0] + intrinsics[..., 1, 1]) / 2
    if np.ndim(focal):
        focal = focal[..., None, None]
    metric *= focal / 300
    median_conf = np.quantile(conf[non_sky], 0.5)
    valid = non_sky & (conf >= median_conf) & (metric > 0.01) & (depth > 0.001)
    if not np.any(valid):
        raise ValueError("No confident non-sky pixels for metric depth alignment")
    scale = np.sum(metric[valid] * depth[valid]) / max(
        float(np.sum(depth[valid] ** 2)), 1e-12
    )
    depth *= scale
    extrinsics[..., :3, 3] *= scale
    depth[~non_sky] = min(float(np.quantile(depth[non_sky], 0.99)), 200.0)
    conf[~non_sky] = 1.0
    if not np.isfinite(depth).all():
        raise ValueError("Non-finite model depth")
    return {
        "depth": depth,
        "confidence": conf,
        "sky": ~non_sky,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "scale_factor": float(scale),
    }


def align_degenerate_camera_poses(predicted, target, align_scale=True):
    """Align orientation and whatever camera-center scale the trajectory constrains."""

    def homogeneous(extrinsics):
        full = np.broadcast_to(
            np.eye(4, dtype=np.float64), (len(extrinsics), 4, 4)
        ).copy()
        full[:, :3] = np.asarray(extrinsics, dtype=np.float64)
        return full

    predicted_c2w = np.linalg.inv(homogeneous(predicted))
    target_c2w = np.linalg.inv(homogeneous(target))
    covariance = np.zeros((3, 3), dtype=np.float64)
    for source, destination in zip(predicted_c2w[:, :3, :3], target_c2w[:, :3, :3]):
        covariance += destination @ source.T
    u, _, vt = np.linalg.svd(covariance)
    handed = np.eye(3)
    handed[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ handed @ vt
    source_centers = predicted_c2w[:, :3, 3]
    target_centers = target_c2w[:, :3, 3]
    source_mean = source_centers.mean(axis=0)
    target_mean = target_centers.mean(axis=0)
    source_centered = source_centers - source_mean
    target_centered = target_centers - target_mean
    rotated = source_centered @ rotation.T
    denominator = float(np.sum(source_centered**2))
    scale = (
        (float(np.sum(rotated * target_centered)) / denominator)
        if align_scale and denominator > 1e-10
        else 1.0
    )
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = predicted_c2w.copy()
    aligned[:, :3, :3] = rotation @ predicted_c2w[:, :3, :3]
    aligned[:, :3, 3] = (scale * (rotation @ source_centers.T)).T + translation
    return scale, np.linalg.inv(aligned)[:, :3].astype(np.float32)


@dataclass
class PendingBatch:
    host: list
    device: list
    copied: object
    start: object
    end: object
    is_scene: bool

    def resolve(self):
        self.copied.synchronize()
        arrays = [value.numpy() for value in self.host]
        if self.is_scene:
            arrays = [value[0] for value in arrays]
        else:
            arrays = [value[:, 0] for value in arrays]
        return arrays, self.start.elapsed_time(self.end)


class Engine:
    def __init__(
        self,
        model_dir,
        allocator_gb=28.0,
        *,
        fused_rope=True,
        cudnn_benchmark=False,
        bf16_convs=False,
        input_uint8=True,
    ):
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(
            allocator_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory
        )
        self.model = (
            DepthAnything3.from_pretrained(model_dir, local_files_only=True)
            .eval()
            .to("cuda")
        )
        self.model.model.da3.backbone.to(torch.bfloat16)
        self.model.model.da3_metric.backbone.to(torch.bfloat16)
        self.raw = RawNetwork(self.model).eval()
        self.input_dtype = torch.uint8 if input_uint8 else torch.float32
        self.fused_rope = False
        if fused_rope:
            try:
                from .kernels import fused_rope as rope_kernel

                self.rope_kernel = rope_kernel
                self.fused_rope = True
            except ImportError:
                log.warning("Fused RoPE unavailable; using reference operations")
        if bf16_convs:
            for head in [self.raw.anyview.head, self.raw.metric.head]:
                for module in head.modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                        module.to(dtype=torch.bfloat16)
        rope_cache = {}
        self.rope_cache = rope_cache
        for module in self.raw.modules():
            if module.__class__.__name__ == "RotaryPositionEmbedding2D" or (
                hasattr(module, "_compute_frequency_components")
                and hasattr(module, "_apply_1d_rope")
            ):

                def cached_rope(module, tokens, positions, _cache=rope_cache):
                    key = (positions.data_ptr(), tokens.shape[-1], tokens.dtype)
                    if key not in _cache:
                        cos, sin = module._compute_frequency_components(
                            tokens.shape[-1] // 2,
                            int(positions.max()) + 1,
                            tokens.device,
                            tokens.dtype,
                        )
                        _cache[key] = tuple(
                            F.embedding(positions[..., axis], freq)[:, None, :, :]
                            for axis in [0, 1]
                            for freq in [cos, sin]
                        )
                    cy, sy, cx, sx = _cache[key]
                    if self.fused_rope:
                        return self.rope_kernel(tokens, cy, sy, cx, sx)
                    vertical, horizontal = tokens.chunk(2, dim=-1)
                    return torch.cat(
                        (
                            vertical * cy + module._rotate_features(vertical) * sy,
                            horizontal * cx + module._rotate_features(horizontal) * sx,
                        ),
                        dim=-1,
                    )

                module.forward = types.MethodType(cached_rope, module)
        # RoPE and positional interpolation are input-independent for a fixed shape.
        # Cache them before graph capture; upstream allocates a CPU tensor in RoPE.
        self.constant_caches = []
        for backbone in [
            self.raw.anyview.backbone.pretrained,
            self.raw.metric.backbone.pretrained,
        ]:
            for name in ["_prepare_rope", "interpolate_pos_encoding"]:
                original = getattr(backbone, name)
                cache = {}
                self.constant_caches.append(cache)

                def memoized(module, *args, _original=original, _cache=cache, **kwargs):
                    key = tuple(
                        (tuple(v.shape), v.dtype, str(v.device))
                        if isinstance(v, torch.Tensor)
                        else str(v)
                        for v in args
                    )
                    if key not in _cache:
                        _cache[key] = _original(*args, **kwargs)
                    return _cache[key]

                setattr(backbone, name, types.MethodType(memoized, backbone))
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.copy_stream = torch.cuda.Stream()
        self.capture_misses = 0
        self.capture_failures = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.inference_mode()
    def run(self, images, graph=False, asynchronous=False):
        key = tuple(images.shape)
        if graph and key in self.graphs:
            captured, inp, out = self.graphs[key]
            if images.dtype != inp.dtype:
                raise ValueError("Captured graph input dtype does not match request")
            inp.copy_(images, non_blocking=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            captured.replay()
            end.record()
        else:
            caches = self.constant_caches + [self.rope_cache]
            previous = [set(cache) for cache in caches]
            try:
                image = images.to("cuda", non_blocking=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = self.raw(image)
                end.record()
                return self._copy_out(out, start, end, False, asynchronous)
            finally:
                # Keep graph constants, but do not accumulate GPU constants for
                # every uncaptured aspect ratio seen over a long-lived service.
                for cache, keep in zip(caches, previous):
                    for added in set(cache) - keep:
                        del cache[added]
        # Single sync/copy per tensor and batch; result buffers are independent of the graph pool.
        return self._copy_out(out, start, end, False, asynchronous)

    def _copy_out(self, out, start, end, is_scene, asynchronous):
        # Separate storage lets the next graph reuse its shared pool while DMA runs.
        device = [value.to(dtype=torch.float32, copy=True) for value in out]
        ready = torch.cuda.Event()
        ready.record()
        host = [
            torch.empty(value.shape, dtype=torch.float32, pin_memory=True)
            for value in device
        ]
        self.copy_stream.wait_event(ready)
        with torch.cuda.stream(self.copy_stream):
            for cpu, gpu in zip(host, device):
                cpu.copy_(gpu, non_blocking=True)
                gpu.record_stream(self.copy_stream)
            copied = torch.cuda.Event()
            copied.record()
        pending = PendingBatch(host, device, copied, start, end, is_scene)
        if asynchronous:
            return pending
        return pending.resolve()[0]

    @torch.inference_mode()
    def run_scene(
        self, prepared, *, use_ray_pose=False, ref_view_strategy="saddle_balanced"
    ):
        images, extrinsics, intrinsics = prepared
        image = images[None].to("cuda", non_blocking=True)
        ex = extrinsics.to("cuda")[None] if extrinsics is not None else None
        ix = intrinsics.to("cuda")[None] if intrinsics is not None else None
        if ex is not None:
            ex = self.model._normalize_extrinsics(ex.clone())
        caches = self.constant_caches + [self.rope_cache]
        previous = [set(cache) for cache in caches]
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self.raw(
                image,
                ex,
                ix,
                use_ray_pose=use_ray_pose,
                ref_view_strategy=ref_view_strategy,
            )
            end.record()
            return self._copy_out(out, start, end, True, True)
        finally:
            for cache, keep in zip(caches, previous):
                for added in set(cache) - keep:
                    del cache[added]

    @torch.inference_mode()
    def prepare_scene(self, images, extrinsics, intrinsics, process_res):
        return self.model.input_processor(
            images,
            extrinsics,
            intrinsics,
            process_res,
            "upper_bound_resize",
            num_workers=1,
        )

    def align_scene(self, result, prepared, align_to_input_ext_scale=True):
        _, ex, ix = prepared
        if ex is None:
            return result
        if len(ex) == 1:
            result["extrinsics"] = ex.numpy()[..., :3, :]
            result["intrinsics"] = ix.numpy()
            return result
        prediction = types.SimpleNamespace(
            depth=result["depth"],
            intrinsics=result["intrinsics"],
            extrinsics=result["extrinsics"],
        )
        target = ex.numpy()[..., :3, :]
        predicted = prediction.extrinsics
        predicted_full = np.broadcast_to(
            np.eye(4, dtype=np.float32), (len(predicted), 4, 4)
        ).copy()
        predicted_full[:, :3] = predicted
        target_full = np.broadcast_to(
            np.eye(4, dtype=np.float32), (len(target), 4, 4)
        ).copy()
        target_full[:, :3] = target
        predicted_centers = np.linalg.inv(predicted_full)[:, :3, 3]
        target_centers = np.linalg.inv(target_full)[:, :3, 3]
        source_rank = np.linalg.matrix_rank(
            predicted_centers - predicted_centers.mean(axis=0)
        )
        target_rank = np.linalg.matrix_rank(
            target_centers - target_centers.mean(axis=0)
        )
        if min(source_rank, target_rank) < 2:
            scale, aligned = align_degenerate_camera_poses(
                predicted, target, align_to_input_ext_scale
            )
            result["intrinsics"] = ix.numpy()
            if align_to_input_ext_scale:
                result["extrinsics"] = target
                # The fallback maps predicted centers to target centers, unlike
                # upstream's inverse alignment. Depth must follow that scale.
                result["depth"] *= scale
                result["scale_factor"] *= scale
            else:
                result["extrinsics"] = aligned
            log.warning(
                "Pose alignment used the rotation/baseline fallback for a degenerate camera trajectory"
            )
            return result
        prediction = self.model._align_to_input_extrinsics_intrinsics(
            ex, ix, prediction, align_to_input_ext_scale
        )
        result.update(
            depth=prediction.depth,
            intrinsics=prediction.intrinsics,
            extrinsics=prediction.extrinsics[..., :3, :],
        )
        return result

    def clear_eager_cache(self):
        # Graph private pools survive; unused eager workspaces are returned to CUDA.
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup(self, shape):
        inp = torch.zeros(shape, dtype=self.input_dtype, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.raw(inp)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        del inp
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def capture(self, shape, warmed=False):
        if not warmed:
            self.warmup(shape)
        inp = torch.zeros(shape, dtype=self.input_dtype, device="cuda")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.raw(inp)
        self.graphs[tuple(shape)] = (graph, inp, out)
        torch.cuda.synchronize()
        return {
            "shape": shape,
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }
