# FastDA3 0.1.0 verification — 2026-11-12

Base: `ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
Hardware: NVIDIA A100-SXM4-80GB. Runtime: Python 3.10, torch 2.5.1+cu121,
torchvision 0.20.1+cu121, Triton 3.1.0. GPU process guard: 32 decimal GB.

The final serving package was tested on a staging port, deployed to the existing
production port, and tested again before repository publication. No API keys,
request data, private deployment paths, or downloaded weights are included here.

## Bash and Conda deployment update

The portable deployment was verified separately using a newly created Python
3.10.22 Conda environment with no inherited site packages. The complete serving
dependencies were installed from downloaded Linux wheels, followed by the
checkout's editable installation. `pip check` and native-library imports passed.
The same dependency resolution is recorded in
`requirements/fastda3.constraints.txt` and was reapplied through the installer.
The clean install exposed a missing `editables` build dependency, which is now
included explicitly. OpenCV uses the headless distribution.

This verification used direct SSH and the repository's Bash scripts. It used no
Slurm commands, job scheduler, system service, or root installation. The GPU test
service was started with `nohup bash scripts/serve.sh` in an activated Conda
environment; later SSH sessions accessed the service successfully. SIGTERM
completed application shutdown and released the listening port and GPU process.

| Deployment check | Result |
|---|---|
| `environment.yml` | Conda parsed and solved successfully using cached packages |
| Independent Conda environment | Python 3.10.22; no old environment or user site packages |
| Full offline dependency installation | Passed with a complete Linux CPython 3.10 wheelhouse |
| Dependency health and native imports | `pip check` and import checks passed |
| Bash syntax and logical regressions | 21 passed, including 4 deployment regressions |
| Other working directory and checkout path containing spaces | Passed |
| Caller environment overriding `.env`, alternate configuration file | Passed |
| CLI paths overriding environment, relative paths resolved from checkout | Passed |
| Assets and GPU preflight | Model/SALAD/DINOv2, writable output directories, BF16 CUDA operation passed |
| Background service and SIGTERM | Cross-session access, preserved PID, graceful exit, port/GPU released |
| Complete HTTP/GPU regression | Passed; single-image arrays remain pixel-identical to the old baseline |
| Multi-view, pose conditions, ray pose, SSE | Passed |
| Unconditioned video and pose-conditioned video | 26 frames each; 4 and 9 windows respectively; loop optimization and exports passed |

The clean-environment short run used 48 image requests at concurrency 16: **8.464
images/s**, P50 **1.695 s**, P95 **2.029 s**. The GPU process peak sampled over the
complete regression was **22.159 decimal GB**. All 81 inference operations and
artifact persistence completed with zero failures. This is a short run on an
A100, not a hardware maximum or a measurement across GPU models.

Downloaded packages and the pre-existing model assets were staged on the test
host because that host could not directly resolve external domains. These
network accommodations are not part of the release or runtime. A fresh operating
system, another GPU model, and the full online weight download were not tested
in this update. The supported baseline is Linux x86_64 with glibc 2.31+, Python
3.10 and a CUDA-12.1-compatible NVIDIA GPU with native BF16; the default memory
budget requires at least 32 decimal GB on one GPU.

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

The earlier wheel/SDK validation used an isolated virtual environment that
inherited the existing CUDA environment, which also contained unrelated
evaluation packages with missing dependencies. The independent Conda validation
above replaces that limitation for the documented serving installation:
whole-environment `pip check` now passes. A fresh operating system was not tested.

The video probe is synthetic (52 source frames sampled to 26), intended to verify
multiple-window control flow and exported artifacts. No 100,000-frame video,
dataset accuracy evaluation, or multi-day stability run was performed. Large
jobs remain bounded by window pixel, GPU, frame-count, and disk budgets.
