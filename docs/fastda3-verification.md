# FastDA3 0.1.0 verification — 2026-11-12

Base: `ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
Hardware: NVIDIA A100-SXM4-80GB. Runtime: Python 3.10, torch 2.5.1+cu121,
torchvision 0.20.1+cu121, Triton 3.1.0. GPU process guard: 32 decimal GB.

The final serving package was tested on a staging port, deployed to the existing
production port, and tested again before repository publication. No API keys,
request data, private deployment paths, or downloaded weights are included here.

## Functional checks

| Check | Result |
|---|---|
| Logic/edge-case regressions | 17 passed |
| Python compilation and Ruff undefined/unused-name checks | Passed |
| Package build and installed CLI | Wheel built and CLI executed |
| Documented OpenAI SDK client | Single image, two-view ray pose, and video download passed with SDK 1.109.1 |
| Existing single-image compatibility | Float32 depth arrays matched the prior service pixel for pixel on the same test image |
| Three-view joint inference | Passed, shape `[3,280,504]` |
| Three-view camera conditions plus ray pose | Passed; output camera conditions checked |
| Single-view ray pose | Passed |
| OpenAI SSE plus usage event | Passed |
| Uncaptured aspect ratio | Real batch of 3 at `[630,1022]` passed |
| Video without camera conditions | 26 sampled frames, 4 windows, 1 loop constraint and successful graph optimization |
| Video with camera conditions | 26 sampled frames, 9 windows; all exported camera poses matched input world poses |
| Streaming without input scale alignment | 5 frames, 2 windows; passed |
| ZIP and numeric artifacts | CRC, every frame depth, camera dimensions, finite values, and extrinsic shapes checked |
| Colored point cloud | Binary PLY vertex count, payload length, finite XYZ and RGB values checked |
| Ordinary image request during video task | Passed, about 0.247 seconds in the production run |
| Job cancellation | Passed |
| Bearer authentication and invalid request errors | Passed |

Regression fixes cover camera-baseline scale direction, conditioned multi-window
export, mixed-aspect overlap crops, image-color preprocessing, upload quota
accounting, download availability during result persistence, queued cancellation,
and a shared storage budget for image and video outputs. Completed video jobs
remove temporary frames and window files.

## Short performance runs

The input is the upstream SOH sample, resized to a 1024-pixel target long edge
(actual shape `[574,1022]`). Each concurrent run contains 48 requests, concurrency
16. Timing covers request submission through the result metadata response and
does not include downloading each NPZ. These are short, warm service runs.

| Run | Images/s | P50 seconds | P95 seconds | Process GPU peak GB |
|---|---:|---:|---:|---:|
| Staging, final package | 8.418 | 1.703 | 2.140 | 23.769 across full regression |
| Production, same package | 8.455 | 1.693 | 2.034 | 23.769 across full regression |

The production regression completed 81 ordinary or scene inference operations
with zero inference failures and zero persistence failures. At completion the
request queue and active foreground request count were both zero. Both streaming
ZIPs included all 26 frames; the unconditioned PLY contained 2,863 points and the
conditioned PLY contained 896 points at the selected point strides.

The previous service's earlier ~7.29 images/s observation used a different request
location/schedule, so it is contextual evidence rather than a controlled A/B
experiment. Output equivalence against the old service does not establish
equivalence to full-precision upstream inference or dataset-wide accuracy.

## Reproduction and limits

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py --base-url http://127.0.0.1:8035 \
  --report runtime/smoke-report.json
```

Set `DA3_API_KEY` before the live checks. Add `--baseline /path/to/before-output.npz`
only when that baseline was generated from the same image and old deployment.

Wheel/SDK validation used an isolated virtual environment that inherited the
existing CUDA environment. A fresh operating-system installation was not tested.
The inherited environment also contains optional upstream evaluation packages
with missing dependencies; a whole-environment `pip check` therefore does not
pass. Those evaluation packages are not called by this service. The documented
installation resolves the declared serving dependencies through normal pip.

The video probe is synthetic (52 source frames sampled to 26), intended to verify
multiple-window control flow and exported artifacts. No 100,000-frame video,
dataset accuracy evaluation, or multi-day stability run was performed. Large
jobs remain bounded by window pixel, GPU, frame-count, and disk budgets.
