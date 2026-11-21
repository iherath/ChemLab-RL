"""Upload one file to the expected workspace with bounded IPv4 transfers and visible progress.

Set CHEMLAB_MODAL_WORKSPACE to use your own workspace (default: ish70).

No Modal App or GPU is created. This is pinned to Modal 1.6.4 because its public
batch_upload API does not expose connection limits, request timeouts, or progress.
The private _VolumeUploadContextManager2 and HTTP session factory below are the
same paths used by that SDK's `modal volume put`; re-audit before upgrading.
The factory replacement is local to this standalone process, never an SDK edit.

Block progress means bytes handed to HTTP, not a durable file. Only the
upload_registered event confirms the server's final VolumePutFiles2 response.
Use --verify to additionally download and check the complete file SHA-256.
SDK output is suppressed because warnings/errors may contain token identifiers
or signed upload URLs. This program emits only controlled JSON records.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path, PurePosixPath
import socket
import ssl
import subprocess
import sys
import time


SDK_VERSION = "1.6.4"


def verify_workspace() -> None:
    if os.environ.get("MODAL_TOKEN_ID") or os.environ.get("MODAL_TOKEN_SECRET"):
        raise ValueError("Use the verified config profile without token environment overrides")
    result = subprocess.run(
        [sys.executable, "-m", "modal", "profile", "list", "--json"],
        capture_output=True, text=True, timeout=30, check=True,
    )
    profiles = json.loads(result.stdout)
    selected = os.environ.get("MODAL_PROFILE")
    active = [p for p in profiles if p["name"] == selected] if selected else [p for p in profiles if p["active"]]
    expected = os.environ.get("CHEMLAB_MODAL_WORKSPACE", "ish70")
    if not expected or len(active) != 1 or active[0]["workspace"] != expected:
        raise ValueError(f"Refusing upload outside the authenticated {expected!r} workspace")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


async def upload(args, emit, started: float) -> None:
    import aiohttp
    import certifi
    from modal._utils import http_utils
    from modal._utils.proxy_support import get_proxy_url
    from modal.volume import _Volume, _VolumeUploadContextManager2
    from modal_proto import api_pb2

    if get_proxy_url(None, use_ssl=True):
        raise ValueError("An SDK proxy is configured; this IPv4-specific uploader requires a direct connection")
    original_factory = http_utils._http_client_with_tls
    progress = {"phase": "resolving_volume", "bytes_sent": 0, "bytes_verified": 0, "retry_events": 0}

    def bounded_http_session(timeout=None):
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(cafile=certifi.where()),
            family=socket.AF_INET, limit=args.concurrency,
            limit_per_host=args.concurrency,
        )
        return aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(
                total=args.request_seconds, connect=min(20, args.request_seconds),
                sock_connect=min(20, args.request_seconds),
                sock_read=min(45, args.request_seconds),
            ),
        )

    def report(**event):
        if "name" in event:
            progress["phase"] = "sending_blocks"
            return "file"
        advance = int(event.get("advance", 0))
        if advance < 0:
            progress["retry_events"] += 1
        progress["bytes_sent"] = max(0, progress["bytes_sent"] + advance)
        if event.get("complete"):
            progress["phase"] = "registering_file" if event.get("task_id") else "registered"
        return "file"

    async def heartbeat():
        while True:
            emit("progress", **progress, elapsed_seconds=round(time.monotonic() - started, 1))
            await asyncio.sleep(5)

    async def work():
        volume = await _Volume.from_name(args.volume, environment_name="main").hydrate()
        if volume._metadata.version != api_pb2.VolumeFsVersion.VOLUME_FS_VERSION_V2:
            raise ValueError("This uploader supports an existing Volume V2 only")
        progress["phase"] = "hashing_blocks"
        async with _VolumeUploadContextManager2(
            volume.object_id, volume._client, progress_cb=report, force=args.force,
            hash_concurrency=args.concurrency, put_concurrency=args.concurrency,
        ) as batch:
            batch.put_file(args.file, args.remote_path)
        emit("upload_registered", volume=args.volume, remote_path=args.remote_path,
             size_bytes=args.file.stat().st_size,
             elapsed_seconds=round(time.monotonic() - started, 1))
        if args.verify:
            progress["phase"] = "verifying_download"
            expected = await asyncio.to_thread(file_digest, args.file)
            digest = hashlib.sha256()
            received = 0
            async for block in volume.read_file(args.remote_path):
                digest.update(block)
                received += len(block)
                progress["bytes_verified"] = received
            if received != args.file.stat().st_size or digest.hexdigest() != expected:
                raise ValueError("Remote file size or SHA-256 does not match the local source")
            emit("verified", size_bytes=received, sha256=expected,
                 elapsed_seconds=round(time.monotonic() - started, 1))

    http_utils._http_client_with_tls = bounded_http_session
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Upload deadline expired before transfer")
        await asyncio.wait_for(work(), timeout=remaining)
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task
        await http_utils.ClientSessionRegistry.close_session()
        http_utils._http_client_with_tls = original_factory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--volume", default="chemlab-rl-data")
    parser.add_argument("--remote-path")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--request-seconds", type=float, default=120)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--force", action="store_true", help="Explicitly permit replacing the named remote file")
    parser.add_argument("--verify", action="store_true", help="Download the uploaded file and verify SHA-256")
    args = parser.parse_args(argv)
    args.file = args.file.resolve()
    if not args.file.is_file():
        parser.error("Source must be an existing regular file")
    if not 1 <= args.concurrency <= 16 or args.request_seconds <= 0 or args.max_seconds <= 30:
        parser.error("Use concurrency1–16, positive request timeout, and total timeout over30seconds")
    args.remote_path = args.remote_path or "/" + args.file.name
    remote = PurePosixPath(args.remote_path)
    if not remote.is_absolute() or ".." in remote.parts or args.remote_path.endswith("/"):
        parser.error("Remote path must be an absolute file path without '..'")
    output = sys.stdout

    def emit(event, **fields):
        print(json.dumps({"event": event, **fields}), file=output, flush=True)

    started = time.monotonic()
    try:
        if importlib.metadata.version("modal") != SDK_VERSION:
            raise ValueError("This private API adapter requires modal==1.6.4")
        verify_workspace()
        emit("starting", workspace=os.environ.get("CHEMLAB_MODAL_WORKSPACE", "ish70"), volume=args.volume, remote_path=args.remote_path,
             size_bytes=args.file.stat().st_size, ipv4=True, concurrency=args.concurrency,
             request_seconds=args.request_seconds, max_seconds=args.max_seconds)
        # Do not surface SDK server warnings, signed URLs, or credential identifiers.
        with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            logging.disable(logging.CRITICAL)
            asyncio.run(upload(args, emit, started))
    except (Exception, KeyboardInterrupt) as exc:
        emit("failed", error_type=type(exc).__name__, elapsed_seconds=round(time.monotonic() - started, 1),
             detail="Upload or verification did not complete; SDK exception text omitted to protect signed URLs and credentials")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
