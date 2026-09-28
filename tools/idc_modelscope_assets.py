"""Inventory and checksum-verified sequential downloads on IDC only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time
import urllib.parse
import urllib.request


def request(url, extra_headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "lingbot-vla2-idc-preflight", **(extra_headers or {})}), timeout=120)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch_file(root, repo, item, resume=False):
    target = root / item["Path"]
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Unsafe artifact path")
    expected_size, expected_sha = item["Size"], item["Sha256"]
    if not expected_sha:
        raise ValueError(f"Missing checksum: {item['Path']}")
    if target.exists():
        if target.stat().st_size == expected_size and digest(target) == expected_sha:
            print("VERIFIED_EXISTING", item["Path"], flush=True)
            return
        raise RuntimeError(f"Existing artifact differs; preserve and investigate: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    if partial.exists() and not resume:
        raise RuntimeError(f"Partial artifact already exists: {partial}")
    query = urllib.parse.urlencode({"Revision": item["Revision"], "FilePath": item["Path"]})
    url = f"https://modelscope.cn/api/v1/models/{repo}/repo?{query}"
    start = last = time.monotonic()
    downloaded = partial.stat().st_size if partial.exists() else 0
    initial_size = downloaded
    if downloaded >= expected_size:
        raise RuntimeError(f"Partial size is not resumable: {downloaded}/{expected_size}")
    h = hashlib.sha256()
    if downloaded:
        with partial.open('rb') as previous:
            for block in iter(lambda: previous.read(8 * 1024 * 1024), b''):
                h.update(block)
    print("START", item["Path"], expected_size, flush=True)
    headers = {'Range': f'bytes={downloaded}-'} if downloaded else {}
    with request(url, headers) as src:
        if downloaded and (src.status != 206 or not src.headers.get('Content-Range', '').startswith(f'bytes {downloaded}-')):
            raise RuntimeError('Server did not honor resume range; existing partial preserved')
        with partial.open("ab" if downloaded else "xb") as dst:
            while block := src.read(4 * 1024 * 1024):
                dst.write(block)
                h.update(block)
                downloaded += len(block)
                if downloaded > expected_size:
                    raise RuntimeError("Server exceeded manifest size")
                now = time.monotonic()
                if now - last >= 15:
                    print("PROGRESS", item["Path"], downloaded, f"{(downloaded-initial_size) / (now-start) / 2**20:.2f} MiB/s", flush=True)
                    last = now
    if downloaded != expected_size or h.hexdigest() != expected_sha:
        raise RuntimeError(f"Size/hash mismatch: {partial}; retained for inspection")
    partial.rename(target)
    print("VERIFIED", item["Path"], downloaded, f"{downloaded / max(time.monotonic()-start, .001) / 2**20:.2f} MiB/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["inventory", "download"])
    p.add_argument("--repo", required=True)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--resume", action='store_true', help='Explicitly resume a partial using verified HTTP Content-Range')
    a = p.parse_args()
    root = a.root.resolve()
    if socket.gethostname() != "dev-instance-shenrongtian" or not root.is_relative_to(Path("/pfs/user/data")):
        raise RuntimeError("Only the designated IDC host and /pfs/user/data are allowed")
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "download_manifest.json"
    if a.phase == "inventory":
        if manifest.exists():
            raise RuntimeError("Manifest exists; use download or inspect existing inventory")
        url = f"https://modelscope.cn/api/v1/models/{a.repo}/repo/files?Revision=master&Recursive=true"
        with request(url) as stream:
            response = json.load(stream)
        files = [f for f in response["Data"]["Files"] if f["Type"] == "blob" and not f["Path"].startswith("assets/")]
        total = sum(f["Size"] for f in files)
        free = shutil.disk_usage(root).free
        print(json.dumps({"host": socket.gethostname(), "root": str(root), "repo": a.repo, "total_bytes": total, "free_bytes": free, "proxy_variable_names": [k for k in os.environ if k.lower().endswith("_proxy")], "files": [{"path": f["Path"], "bytes": f["Size"]} for f in files]}, indent=2), flush=True)
        if free < total + 20 * 2**30:
            raise RuntimeError("Insufficient free space with 20 GiB reserve")
        with manifest.open("x") as stream:
            json.dump({"repo": a.repo, "files": files}, stream, indent=2)
        pilot = next(f for f in files if f["Path"] == "config.json")
        if pilot["Size"] > 1024 * 1024:
            raise RuntimeError("Pilot unexpectedly large")
        fetch_file(root, a.repo, pilot)
        print("PILOT_PASS; bulk download requires a separate download invocation", flush=True)
    else:
        record = json.loads(manifest.read_text())
        if record["repo"] != a.repo:
            raise RuntimeError("Manifest repository mismatch")
        for item in record["files"]:
            fetch_file(root, a.repo, item, resume=a.resume)
        print("ALL_ASSETS_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
