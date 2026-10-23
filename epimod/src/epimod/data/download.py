"""Resumable HTTP download with retries (stdlib only)."""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def download(url: str, dest: str | Path, retries: int = 6, chunk: int = 1 << 20, quiet: bool = True) -> Path:
    """Download ``url`` to ``dest`` atomically, resuming a partial ``.part`` file."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            have = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "epimod/0.1"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=120) as resp:
                status = getattr(resp, "status", 200)
                mode = "ab" if (have and status == 206) else "wb"
                with open(part, mode) as fh:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
            os.replace(part, dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            last_err = err
            if isinstance(err, urllib.error.HTTPError) and err.code == 416:
                # range not satisfiable: the partial file is already complete or corrupt
                part.unlink(missing_ok=True)
            if not quiet:
                print(f"[download] retry {attempt + 1}/{retries} for {url}: {err}")
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise RuntimeError(f"failed to download {url}: {last_err}")
