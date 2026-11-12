"""OpenAI-style image and video client for FastDA3."""

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import time

import httpx
from openai import OpenAI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8035/v1"),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    image = sub.add_parser("image")
    image.add_argument("images", type=Path, nargs="+")
    image.add_argument("--process-res", type=int, default=1024)
    image.add_argument("--ray-pose", action="store_true")
    image.add_argument(
        "--cameras", type=Path, help="JSON object with extrinsics and intrinsics"
    )
    image.add_argument("--output", type=Path, default=Path("depth.npz"))
    video = sub.add_parser("video")
    video.add_argument("video", type=Path)
    video.add_argument("--process-res", type=int, default=504)
    video.add_argument("--fps", type=float, default=2.0)
    video.add_argument("--chunk-size", type=int, default=12)
    video.add_argument("--overlap", type=int, default=4)
    video.add_argument("--no-loop-closure", action="store_true")
    video.add_argument("--output", type=Path, default=Path("scene.zip"))
    args = parser.parse_args()
    secret = os.environ["DA3_API_KEY"]
    base = args.base_url.rstrip("/")
    origin = base.removesuffix("/v1")
    headers = {"Authorization": "Bearer " + secret}
    with httpx.Client(headers=headers, timeout=180, trust_env=False) as http:
        if args.command == "image":
            client = OpenAI(base_url=base, api_key=secret)
            content = []
            for path in args.images:
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                encoded = base64.b64encode(path.read_bytes()).decode()
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{encoded}"},
                    }
                )
            options = {
                "task": "multiview" if len(content) > 1 else "depth",
                "process_res": args.process_res,
                "use_ray_pose": args.ray_pose,
            }
            if args.cameras:
                options.update(json.loads(args.cameras.read_text()))
            response = client.chat.completions.create(
                model="da3",
                messages=[{"role": "user", "content": content}],
                extra_body=options,
            )
            result = json.loads(response.choices[0].message.content)
            client.close()
            result_url = result["depth_npz_url"]
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            with args.video.open("rb") as stream:
                response = http.post(
                    base + "/files",
                    data={"purpose": "user_data"},
                    files={"file": (args.video.name, stream, "video/mp4")},
                )
            response.raise_for_status()
            file_id = response.json()["id"]
            response = http.post(
                base + "/video/estimations",
                json={
                    "model": "da3",
                    "video_file_id": file_id,
                    "process_res": args.process_res,
                    "streaming": {
                        "fps": args.fps,
                        "chunk_size": args.chunk_size,
                        "overlap": args.overlap,
                        "loop_closure": not args.no_loop_closure,
                    },
                },
            )
            response.raise_for_status()
            job = response.json()
            try:
                while job["status"] in ("queued", "in_progress"):
                    time.sleep(1)
                    response = http.get(base + "/jobs/" + job["id"])
                    response.raise_for_status()
                    job = response.json()
                    print(
                        job["status"],
                        job.get("stage"),
                        job.get("progress", ""),
                        flush=True,
                    )
            except KeyboardInterrupt:
                http.post(base + "/jobs/" + job["id"] + "/cancel").raise_for_status()
                raise
            if job["status"] != "completed":
                raise RuntimeError(job.get("error") or job["status"])
            result_url = job["result"]["result_url"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with http.stream("GET", origin + result_url) as response:
            response.raise_for_status()
            with args.output.open("wb") as output:
                for block in response.iter_bytes(1024**2):
                    output.write(block)
    print("Saved", args.output)


if __name__ == "__main__":
    main()
