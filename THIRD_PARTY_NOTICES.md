# Third-party notices

FastDA3 extends [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3),
starting at commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. Its original source,
Git history, Apache-2.0 license, attribution, and research documentation are retained.
The original project README is available as [README.upstream.md](README.upstream.md).
The optimized depth head follows the upstream DualDPT implementation.

DA3-Streaming includes code adapted by its authors from
[VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long). Preserve the notices in
`da3_streaming/` when redistributing those files.

[SALAD](https://github.com/serizba/salad) is an upstream Git submodule at
`6aede13a3f6c25750bf7fde10209c06cb73060bb`. Its own license and notices apply;
the submodule is licensed under GPL-3.0. Preserve those terms when distributing
or combining its code. The top-level Apache license does not relicense SALAD.
The pretrained SALAD checkpoint is downloaded separately from its official release.

[DINOv2](https://github.com/facebookresearch/dinov2) source is downloaded separately
at `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, preserving its Apache-2.0 license.
SALAD supplies its backbone weights, so a separate DINOv2 checkpoint is unnecessary.

DA3 model checkpoints retain the licenses on their respective Hugging Face model
cards. In particular, DA3NESTED-GIANT-LARGE-1.1 is CC BY-NC 4.0.
Repository licensing does not replace checkpoint or dependency licensing.
No model weights, service credentials, uploaded videos, or inference outputs are
included in the FastDA3 Git repository.
