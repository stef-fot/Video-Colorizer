"""Download the translation model used for subtitles in 21 languages (once, about 620 MB).

    python scripts/download_translation.py

This is a CTranslate2 int8 build of Meta's NLLB-200 distilled 600M (a community conversion, repository
JustFrederik/nllb-200-distilled-600M-ct2-int8; it contains model.bin and sentencepiece.bpe.model).
LICENSE: NLLB-200 is CC-BY-NC-4.0, which means NON-COMMERCIAL use only. If that does not fit you, set
subs.translator: argos in config.yaml (pip install argostranslate) or use another model.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from colorstream.config import load_config  # noqa: E402

REPO = "JustFrederik/nllb-200-distilled-600M-ct2-int8"
FILES = ["config.json", "model.bin", "sentencepiece.bpe.model", "shared_vocabulary.txt"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub   (it is also installed by requirements-subtitles.txt)")
    dest = Path(cfg.subs.nllb_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {a.repo} to {dest} ...")
    snapshot_download(repo_id=a.repo, local_dir=str(dest), allow_patterns=FILES)
    missing = [f for f in ("model.bin", "sentencepiece.bpe.model") if not (dest / f).exists()]
    if missing:
        raise SystemExit(f"Download incomplete, missing {missing}")
    print("Done. Subtitle translation is ready.")


if __name__ == "__main__":
    main()
