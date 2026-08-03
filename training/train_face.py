"""Enrol face encodings into the database from a labelled image folder.

Face models are not "trained" like CNN classifiers — recognition works by
comparing 128-d encodings. This script computes encodings for every image in
``data/faces/<person_name>/*.{jpg,png}`` and upserts a Customer per person,
making them instantly recognisable via POST /recognize-face.

Usage:
    python -m training.train_face data/faces --name-column folder --seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger  # noqa: E402
from app.database.models import Customer  # noqa: E402
from app.database.session import SessionLocal, init_db  # noqa: E402
from app.models.face_model import FACE_RECOGNITION_AVAILABLE, FaceDetector  # noqa: E402
from app.utils.image_utils import load_image_from_bytes  # noqa: E402

logger = get_logger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_images(root: Path) -> dict[str, list[Path]]:
    """Return {person_name: [image_paths]} from subfolder layout."""
    if not root.exists():
        logger.error("Faces root does not exist: %s", root)
        return {}
    people: dict[str, list[Path]] = {}
    for person_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = [
            p for p in person_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXT and p.is_file()
        ]
        if images:
            people[person_dir.name] = images
    return people


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrol face encodings.")
    parser.add_argument("root", type=Path, help="Folder of per-person subfolders")
    parser.add_argument("--seed", action="store_true", help="Reset existing customers first")
    args = parser.parse_args()

    if not FACE_RECOGNITION_AVAILABLE:
        logger.error("face_recognition (dlib) is required to enrol quality encodings.")
        sys.exit(1)

    people = collect_images(args.root)
    if not people:
        logger.error("No per-person subfolders with images found under %s", args.root)
        sys.exit(1)

    init_db()
    db = SessionLocal()
    detector = FaceDetector()

    if args.seed:
        deleted = db.query(Customer).delete()
        db.commit()
        logger.info("Seeded DB: removed %d existing customers.", deleted)

    enrolled = 0
    for name, images in people.items():
        encodings: list[list[float]] = []
        for path in images:
            image = load_image_from_bytes(path.read_bytes())
            encs = detector.face_encodings(image)
            encodings.extend(e.tolist() for e in encs)
        if not encodings:
            logger.warning("No faces found in '%s', skipping.", name)
            continue

        customer = db.query(Customer).filter(Customer.name == name).first()
        if customer:
            customer.face_encodings = encodings
            logger.info("Updated '%s' with %d encodings.", name, len(encodings))
        else:
            db.add(Customer(name=name, face_encodings=encodings))
            logger.info("Enrolled '%s' with %d encodings.", name, len(encodings))
        enrolled += 1
        db.commit()

    db.close()
    logger.info("Enrolment complete: %d people, %d total images processed.", enrolled, sum(len(v) for v in people.values()))


if __name__ == "__main__":
    main()
