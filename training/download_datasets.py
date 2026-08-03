"""Dataset download / preparation helpers.

The platform can be trained on free public datasets:
- Fashion-MNIST (product image classification, mapped to retail labels)
- Women's E-Commerce Clothing Reviews (sentiment analysis)
- LFW (face recognition reference dataset)

Run:  python -m training.download_datasets  [--fashion] [--sentiment] [--lfw]
All downloads are optional; the API runs with fallbacks without them.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DATA_DIR = settings.DATA_DIR
FASHION_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
LFW_URL = "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
SENTIMENT_URL = (
    "https://raw.githubusercontent.com/hanzhang0420/Women-Clothing-E-commerce/"
    "master/Womens%20Clothing%20E-Commerce%20Reviews.csv"
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)
    return dest


def download_fashion_mnist() -> None:
    """Download Fashion-MNIST binary files for product-model training."""
    dest_dir = DATA_DIR / "fashion-mnist"
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for name in files:
        _download(FASHION_URL + name, dest_dir / name)
    logger.info("Fashion-MNIST ready in %s", dest_dir)


def download_sentiment() -> None:
    """Download the e-commerce clothing reviews CSV for sentiment training."""
    dest = DATA_DIR / "ecommerce_reviews.csv"
    _download(SENTIMENT_URL, dest)
    logger.info("Sentiment dataset ready in %s", dest)


def download_lfw() -> None:
    """Download LFW deep-funneled faces (for face recognition references)."""
    dest = DATA_DIR / "lfw-deepfunneled.tgz"
    _download(LFW_URL, dest)
    extract_to = DATA_DIR / "lfw"
    if not extract_to.exists():
        import tarfile

        with tarfile.open(dest, "r:gz") as tar:
            tar.extractall(path=extract_to, filter="data")
        logger.info("LFW extracted to %s", extract_to)
    else:
        logger.info("LFW already extracted at %s", extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download free public datasets.")
    parser.add_argument("--fashion", action="store_true", help="Fashion-MNIST")
    parser.add_argument("--sentiment", action="store_true", help="Reviews CSV")
    parser.add_argument("--lfw", action="store_true", help="LFW face dataset")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if args.all or args.fashion:
        download_fashion_mnist()
    if args.all or args.sentiment:
        download_sentiment()
    if args.all or args.lfw:
        download_lfw()
    if not (args.all or args.fashion or args.sentiment or args.lfw):
        parser.print_help()


if __name__ == "__main__":
    main()
