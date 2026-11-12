"""Bounded memory cache with asynchronous, atomic durable result writes."""

import asyncio
import concurrent.futures
import io
import json
import os
import re
import secrets
import threading
import time
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .protocol import APIError


RESULT_NAME = re.compile(r"^[a-f0-9]{24}\.(npz|zip|json|ply)$")
FILE_ID = re.compile(r"^file-[a-f0-9]{24}$")


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, allow_nan=False)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class ResultStore:
    def __init__(self, root, cache_bytes=256 * 1024**2, ttl=3600):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_bytes = cache_bytes
        self.ttl = ttl
        self.cache = OrderedDict()
        self.writes = {}
        self.size = 0
        self.lock = threading.Lock()
        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="da3-persist"
        )
        self.pending = threading.BoundedSemaphore(16)
        self.write_failures = 0
        self.budget = None

    def put_npz(self, arrays, return_inline=False):
        identifier = secrets.token_hex(12)
        filename = identifier + ".npz"
        output = io.BytesIO()
        np.savez(output, **arrays)
        payload = output.getvalue()
        self.pending.acquire()
        if self.budget is not None:
            try:
                self.budget.track_file(
                    identifier, self.root / filename, size=len(payload)
                )
            except BaseException:
                self.pending.release()
                raise
        with self.lock:
            self.writes[filename] = payload
            self.cache[filename] = (payload, time.time())
            self.size += len(payload)
            while self.size > self.cache_bytes and len(self.cache) > 1:
                _, (old, _) = self.cache.popitem(last=False)
                self.size -= len(old)
        self.pool.submit(self._persist, filename, payload)
        return identifier, payload if return_inline else None

    def _persist(self, filename, payload):
        path = self.root / filename
        try:
            temp = path.with_suffix(".npz.tmp")
            temp.write_bytes(payload)
            os.replace(temp, path)
        except Exception:
            self.write_failures += 1
            if self.budget is not None:
                self.budget.untrack_tree(path)
            import logging

            logging.getLogger("da3").exception("Result persistence failed")
        finally:
            with self.lock:
                self.writes.pop(filename, None)
            self.pending.release()

    def get(self, filename):
        if not RESULT_NAME.fullmatch(filename):
            raise APIError("Unknown result", 404, "not_found")
        with self.lock:
            cached = self.cache.get(filename)
            if cached and time.time() - cached[1] < self.ttl:
                self.cache.move_to_end(filename)
                return cached[0], None
            if filename in self.writes:
                return self.writes[filename], None
        path = self.root / filename
        if not path.is_file():
            raise APIError("Unknown or expired result", 404, "not_found")
        return None, path

    def cleanup(self):
        cutoff = time.time() - self.ttl
        with self.lock:
            for filename, (payload, created) in list(self.cache.items()):
                if created < cutoff:
                    self.cache.pop(filename)
                    self.size -= len(payload)
        for path in self.root.iterdir():
            if (
                path.is_file()
                and RESULT_NAME.fullmatch(path.name)
                and path.stat().st_mtime < cutoff
            ):
                path.unlink(missing_ok=True)
                if self.budget is not None:
                    self.budget.untrack_tree(path)


class FileStore:
    def __init__(self, root, max_bytes=8 * 1024**3):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.lock = threading.Lock()
        self.total_bytes = sum(path.stat().st_size for path in self.root.glob("*.data"))

    async def upload(self, upload, purpose="user_data"):
        if purpose not in ("user_data", "vision"):
            raise APIError("purpose must be user_data or vision")
        identifier = "file-" + secrets.token_hex(12)
        path = self.root / (identifier + ".data")
        count = 0
        reserved = 0
        try:
            with path.open("wb") as output:
                while block := await upload.read(1024 * 1024):
                    count += len(block)
                    with self.lock:
                        if self.total_bytes + len(block) > self.max_bytes:
                            raise APIError(
                                "Uploaded files exceed configured disk limit",
                                413,
                                "file_too_large",
                            )
                        self.total_bytes += len(block)
                        reserved += len(block)
                    await asyncio.to_thread(output.write, block)
            if count == 0:
                raise APIError("Uploaded file is empty")
            data = {
                "id": identifier,
                "object": "file",
                "bytes": count,
                "created_at": int(time.time()),
                "filename": Path((upload.filename or "upload").replace("\\", "/")).name[
                    :200
                ],
                "purpose": purpose,
                "status": "processed",
            }
            atomic_json(self.root / (identifier + ".json"), data)
            return data
        except BaseException:
            with self.lock:
                self.total_bytes = max(0, self.total_bytes - reserved)
            path.unlink(missing_ok=True)
            raise

    def get(self, identifier):
        if not FILE_ID.fullmatch(identifier):
            raise APIError("Invalid file_id")
        path = self.root / (identifier + ".data")
        metadata = self.root / (identifier + ".json")
        if not path.is_file() or not metadata.is_file():
            raise APIError("Unknown file_id", 404, "not_found")
        return path, json.loads(metadata.read_text())

    def cleanup(self, ttl=30 * 24 * 3600):
        cutoff = time.time() - ttl
        removed = 0
        for metadata in self.root.glob("file-*.json"):
            if metadata.stat().st_mtime >= cutoff:
                continue
            data = metadata.with_suffix(".data")
            try:
                size = data.stat().st_size
            except FileNotFoundError:
                size = 0
            metadata.unlink(missing_ok=True)
            data.unlink(missing_ok=True)
            removed += size
        for data in self.root.glob("file-*.data"):
            if not data.with_suffix(".json").exists() and data.stat().st_mtime < cutoff:
                size = data.stat().st_size
                data.unlink(missing_ok=True)
                removed += size
        if removed:
            with self.lock:
                self.total_bytes = max(0, self.total_bytes - removed)
