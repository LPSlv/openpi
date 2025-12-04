#!/usr/bin/env python3
"""Download checkpoints at build time to avoid runtime downloads."""
import pathlib
import sys
import urllib.parse

import fsspec


def download_checkpoint(url: str, cache_dir: pathlib.Path) -> None:
    """Download a checkpoint from gs:// to the cache directory."""
    parsed = urllib.parse.urlparse(url)
    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    
    if local_path.exists():
        print(f"Already cached: {local_path}")
        return
    
    print(f"Downloading {url} to {local_path}")
    fs, _ = fsspec.core.url_to_fs(url)
    fs.get(url, str(local_path), recursive=True)
    print(f"Downloaded {url}")


def main():
    cache_dir = pathlib.Path("/root/.cache/openpi")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs passed as command-line arguments
    for url in sys.argv[1:]:
        if url:
            download_checkpoint(url, cache_dir)


if __name__ == "__main__":
    main()

