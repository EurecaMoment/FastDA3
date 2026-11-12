# FastDA3

基于 [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) 的高吞吐推理服务，保留官方源码、提交历史与 CLI，新增 OpenAI 风格 HTTP API。

使用 **DA3NESTED-GIANT-LARGE-1.1**，提供单图米制深度、多视图联合推理、相机位姿/内参条件、ray pose，以及使用 DA3-Streaming 对齐和回环组件的长视频分块推理。

| 能力 | 行为与输出 |
|---|---|
| 单图深度 | 默认长边 1024；NPZ 包含深度、置信度、天空掩码、内外参 |
| 多视图联合推理 | 同一请求内多张图片构成一个场景，共享几何坐标系 |
| 位姿条件 | 输入 `extrinsics` 和 `intrinsics`，支持静止、双视图和直线轨迹 |
| Ray pose | `use_ray_pose=true` 启用射线位姿分支 |
| 长视频/图片序列 | 有界窗口推理、重叠 Sim(3) 对齐、SALAD 回环检测、图优化 |
| 视频导出 | 每帧深度、完整相机轨迹、内参、带颜色 PLY 点云、ZIP |
| 服务接口 | Bearer 认证、文件上传、后台任务、进度查询、取消、SSE |

不同单图请求按独立场景批处理；同一个多视图请求内的图片才联合推理。HTTP 服务聚焦深度、位姿和视频几何；Gaussian Splatting、新视角渲染和 Gradio 使用[官方入口](README.upstream.md)。

## 环境与安装

推荐 **Linux、Python 3.10、NVIDIA GPU、支持 CUDA 12.1 的驱动**。使用 PyTorch wheel 不需要单独安装 CUDA Toolkit。验收硬件为 A100-SXM4-80GB，服务显存保护上限 **32 十进制 GB**；Windows 和 CPU 推理不在本服务验收范围内。

核心版本：PyTorch `2.5.1+cu121`、torchvision `0.20.1+cu121`、Triton `3.1.0`、xformers `0.0.28.post3`、NumPy `1.26.4`。完整依赖见 [requirements/fastda3.txt](requirements/fastda3.txt)。

```bash
git clone --recurse-submodules https://github.com/EurecaMoment/FastDA3.git
cd FastDA3
git submodule update --init --recursive

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/fastda3.txt
python -m pip install --no-deps -e .
```

Ubuntu 上 OpenCV 导入提示 `libGL.so.1` 或 `libgthread` 缺失时：

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

依赖包含官方 API 导入所需的导出库；`gsplat` 仅在使用官方 Gaussian 渲染功能时另行安装。

## 下载权重与长视频组件

```bash
# 下载阶段需要联网，请在设置 HF_HUB_OFFLINE 之前执行。
python scripts/download_assets.py
python scripts/download_assets.py --check-only
```

脚本准备 DA3 模型（约 6.8 GiB）、`dino_salad.ckpt`（约 336 MiB，验证 SHA256）、固定提交的 DINOv2 源码。SALAD 源码由 Git 子模块提供，其 checkpoint 已包含所需的 backbone 权重。

默认目录为 `weights/DA3NESTED-GIANT-LARGE-1.1`、`weights/dino_salad.ckpt`、`vendor/dinov2`。已有资源可指定 `--model-dir`、`--salad-checkpoint` 和 `--dinov2-repo`，并把相同路径写入 `.env`。离线部署时复制这些资源和完整 SALAD 子模块即可。

## 配置与启动

```bash
cp .env.example .env
# 编辑 .env：填写 DA3_API_KEY，选择 GPU，必要时修改权重和运行数据路径。
source .venv/bin/activate
bash scripts/serve.sh --host 0.0.0.0 --port 8035
```

默认模型名 **`da3`**，客户端 base URL 为 `http://SERVER:8035/v1`。首次启动预热模型并捕获常用尺寸的 CUDA Graph，准备完成后 `/health` 才可用。使用 **1 个 worker**，避免重复加载模型。

`scripts/serve.sh` 自动读取 `.env` 并要求设置 `DA3_API_KEY`。也可用 `fastda3 --model-dir /path/to/model --port 8035` 或 `python -m fastda3.server`，此时环境变量由调用者提供。

```bash
curl http://127.0.0.1:8035/health
curl -H "Authorization: Bearer $DA3_API_KEY" http://127.0.0.1:8035/v1/models
```

在另一终端调用时需要在该终端设置 `DA3_API_KEY`；启动脚本不会修改父终端环境。健康检查显示功能、缺失依赖、队列及显存统计。持久服务可参考 [deploy/fastda3.service.example](deploy/fastda3.service.example)，修改 checkout/Python 路径后保存为 `~/.config/systemd/user/fastda3.service`：

```bash
systemctl --user daemon-reload
systemctl --user enable --now fastda3
journalctl --user -u fastda3 -f
```

| 配置 | 默认值 | 用途 |
|---|---|---|
| `DA3_MODEL_DIR` | `weights/DA3NESTED-GIANT-LARGE-1.1` | 模型目录 |
| `DA3_SALAD_CHECKPOINT` | `weights/dino_salad.ckpt` | 回环权重 |
| `DA3_DINOV2_REPO` | `vendor/dinov2` | 离线 DINOv2 源码 |
| `DA3_STREAMING_REPO` | `da3_streaming` | 官方流式组件 |
| `DA3_DATA_DIR` | `runtime` | 文件、任务和结果目录 |
| `CUDA_VISIBLE_DEVICES` | `0` | 一个物理 GPU 编号或 UUID |
| `--max-gpu-memory-gb` | `32` | 进程显存采样保护，Torch 预算预留 4 GB |
| `--max-batch-size` | `4` | 独立图片真实批次上限 |
| `--graph-batches` | `1,2,3,4` | 捕获的 CUDA Graph 批次 |
| `--batch-wait-ms` | `3` | 聚合等待 |
| `--max-num-seqs` | `256` | 前台处理中及排队请求总数 |
| `--max-scene-pixels` | `2500000` | 单窗口联合推理总像素预算 |
| `--max-file-gb` | `8` | 全部上传文件总量，单位 GiB |
| `--max-job-disk-gb` | `64` | 任务和结果共享预算，单位 GiB |

单图默认长边 1024，实际尺寸对齐 14；请求级 `process_res` 可取 140–1024。多视图默认 1024，视频默认 504；超出场景预算时降低分辨率或窗口大小。显存采样保护不是硬件隔离，也不保证任意数量的高分辨率视图可同时推理。

## OpenAI 风格调用

请求/响应外形兼容 Chat Completions；这是几何模型，`message.content` 是 **JSON 字符串**，token 用量为 0。示例客户端依赖安装命令：`python -m pip install -r requirements/client.txt`（已验证 OpenAI SDK 1.109.1）。

可运行示例：[examples/fastda3_client.py](examples/fastda3_client.py)。

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8035/v1
export DA3_API_KEY=your-secret
python examples/fastda3_client.py image assets/examples/SOH/000.png
python examples/fastda3_client.py image assets/examples/SOH/000.png assets/examples/SOH/010.png \
  --process-res 504 --ray-pose
```

Python SDK 多视图示例：

```python
import base64, json, os
from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8035/v1", api_key=os.environ["DA3_API_KEY"])

def image_part(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type":"image_url", "image_url":{"url":"data:image/png;base64,"+encoded}}

response = client.chat.completions.create(
    model="da3",
    messages=[{"role":"user", "content":[
        image_part("assets/examples/SOH/000.png"),
        image_part("assets/examples/SOH/010.png"),
    ]}],
    extra_body={"task":"multiview", "process_res":504, "use_ray_pose":True},
)
geometry = json.loads(response.choices[0].message.content)
print(geometry["depth_npz_url"], geometry["extrinsics"])
```

位姿条件在 `extra_body` 中同时加入：

```python
extra_body = {
    "task":"multiview", "process_res":504, "use_ray_pose":True,
    "extrinsics": world_to_camera_matrices,  # [N,3,4] 或 [N,4,4]
    "intrinsics": camera_intrinsic_matrices, # [N,3,3]，原始图像像素坐标
    "align_to_input_ext_scale":True,
}
```

矩阵顺序与图片一致。`align_to_input_ext_scale=true` 用输入位姿定义坐标与尺度，输入平移应以米为单位；`false` 保留模型米制深度尺度并返回对齐后的相机位姿。输出内参属于处理后的图像尺寸，外参为 world-to-camera。单个静止视角不能通过相机运动约束额外尺度。

直接深度接口：

```bash
curl -H "Authorization: Bearer $DA3_API_KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8035/v1/depth/estimations \
  -d '{"model":"da3","image_url":"https://example.com/image.png"}'
```

输入支持 HTTP(S)、base64 图片 data URL、上传后的 `file_id`。下载 `depth_npz_url` 也需要 Bearer 认证。NPZ 字段为 `depth_m`、`confidence`、`sky_mask`、`intrinsics`、`extrinsics`；深度形状为 `[H,W]` 或 `[N,H,W]`。`return_inline=true` 返回 NPZ base64，`include_arrays=true` 返回 JSON 深度数组，均有大小限制。

## 长视频与图片序列

```bash
python examples/fastda3_client.py video /path/to/video.mp4 \
  --fps 2 --process-res 504 --chunk-size 12 --overlap 4 --output scene.zip
```

手工调用：

```bash
curl -H "Authorization: Bearer $DA3_API_KEY" \
  -F purpose=user_data -F file=@/path/to/video.mp4 http://127.0.0.1:8035/v1/files

# 使用上一步返回的真实 file_id。
curl -H "Authorization: Bearer $DA3_API_KEY" -H 'Content-Type: application/json' \
  http://127.0.0.1:8035/v1/video/estimations \
  -d '{"model":"da3","video_file_id":"file-REPLACE_WITH_UPLOAD_ID","process_res":504,
       "streaming":{"fps":2,"chunk_size":12,"overlap":4,"loop_closure":true}}'

curl -H "Authorization: Bearer $DA3_API_KEY" http://127.0.0.1:8035/v1/jobs/job-REPLACE_WITH_JOB_ID
```

成功后 `result.result_url` 是 ZIP 下载地址，包含 manifest、逐帧深度、`cameras.npz`、轨迹文本和点云。取消用 `POST /v1/jobs/{id}/cancel`；已执行的单次 GPU 运算结束后，在后续阶段停止任务。

图片序列使用 `/v1/depth/estimations`，传 `task="streaming"` 和顺序排列的 `images`；Chat Completions 也支持图片序列加相同 task。视频位姿条件的矩阵数需等于**抽帧后**帧数，内参属于视频原始分辨率。

`streaming` 选项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `fps` | `2` | 抽帧频率 0.1–60，不重复原始帧 |
| `chunk_size` | 自动 | 联合推理窗口大小，受像素预算约束 |
| `overlap` | 窗口约 1/3 | 重叠帧数，必须小于窗口 |
| `loop_closure` | `true` | SALAD 检测与图优化；有输入位姿时跳过 |
| `loop_threshold` | `0.85` | 描述子相似度阈值 |
| `loop_min_separation` | `25` | 回环候选最小帧间隔 |
| `max_loop_pairs` | `100` | 最多验证的候选 |
| `max_frames` | `100000` | 最大抽样帧数，仍受存储限额约束 |
| `save_depth` / `save_pointcloud` | `true` | 导出开关 |
| `point_stride` | `8` | 点云采样步长 |

视频 streaming 是有界显存的分块处理。OpenAI 的 `stream=true` 是 SSE 格式，封装完成后的 JSON，表示不同的功能；任务真实进度通过 `/v1/jobs/{id}` 查询。

结果约保留 1 小时、任务记录约 24 小时、上传文件约 30 天，定时清理；完成任务的中间帧/窗口立即清理。请及时下载所需结果。

## 优化与验收

- 精确批次 1/2/3/4 的动态批处理和 CUDA Graph；未捕获宽高比支持真正的 eager 批处理。
- BF16 backbone/head 运算、TF32、SDPA、输入无关位置编码缓存。
- Triton 融合 2D RoPE，保留参考 BF16 分步舍入；uint8 输入在 GPU 归一化。
- 独立结果缓冲、异步 GPU 到 CPU 传输、CPU 标定和编码流水线。
- 内存缓存、原子后台落盘、共享存储预算；长视频共用单份 DA3 模型。

2026-11-12，A100、1024 长边、48 请求/16 并发，生产端口本机短测约 **8.46 图/秒**，P95 **2.03 秒**，完整回归进程显存峰值 **23.77 GB**。旧部署此前约 7.29 图/秒，但测量位置与安排不同，不是严格 A/B。性能取决于图片尺寸、混合负载、网络、CPU/磁盘和 GPU 共享情况，不代表硬件极限。详见[验收记录](docs/fastda3-verification.md)。

验收覆盖 17 项逻辑回归；单图与旧版同图逐像素一致；多视图、位姿条件、ray pose、SSE、特殊宽高比；26 帧/4 窗口回环视频；26 帧/9 窗口位姿视频；逐帧深度、相机轨迹、ZIP、彩色二进制 PLY；取消及视频运行期间的单图请求。旧版一致性不等于官方全精度或数据集精度评估，尚未进行 10 万帧或多日稳定性验收。

```bash
# 无需加载 DA3 权重，仍需要安装依赖。
python -m unittest discover -s tests -v

# 服务启动后，实际执行 GPU 推理、视频和导出校验。
export DA3_API_KEY=your-secret
python scripts/smoke_test.py --base-url http://127.0.0.1:8035
# 可附加 --baseline /path/to/before-output.npz 检查旧版同图一致性。
```

## 代码与来源

`src/fastda3/` 包含引擎、调度、协议、存储和视频任务；`src/depth_anything_3/` 与 `da3_streaming/` 保留官方实现；`scripts/` 提供下载/启动/验收，`examples/` 提供客户端，`deploy/` 提供服务配置。

官方文档见 [README.upstream.md](README.upstream.md)，来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。官方代码保留 Apache-2.0；SALAD 子模块采用其 GPL-3.0 许可证；默认 DA3NESTED-GIANT-LARGE-1.1 权重为 CC BY-NC 4.0，具体以模型卡为准。
