"""Download original PyTorch weights listed in config.yaml into weights/.

    python scripts/download_models.py ddcolor-tiny ddcolor-large

The downloaded .pth files must then be converted with scripts/export_onnx.py.
The download URLs come from the model registry (colorstream/config.py) and can be
overridden in config.yaml under 'models:'. If a URL no longer works, download the file
by hand from the model's project page and put it in the weights folder.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colorstream.config import load_config  # noqa: E402


def download(url: str, dest: Path, sha256: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        h = hashlib.sha256()
        while chunk := r.read(1 << 20):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    if sha256 and h.hexdigest() != sha256:
        tmp.unlink()
        raise SystemExit(f"Checksum mismatch for {dest.name}, file deleted.")
    tmp.replace(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("-c", "--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for name in a.models:
        spec = cfg.model_spec(name)
        if not spec.url:
            print(f"{name}: no download URL configured (it is exported from a repository, see README).")
            continue
        dest = Path(cfg.paths.models_dir) / Path(spec.url).name
        if dest.exists():
            print(f"{name}: {dest} already exists, skipping.")
            continue
        download(spec.url, dest, spec.sha256)
    print("Next step: python scripts/export_onnx.py <model>")


if __name__ == "__main__":
    main()
