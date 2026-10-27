"""Normalize FineFake or MDSM annotations into view-specific JSONL manifests."""

import argparse
import json
from pathlib import Path


def main(args):
    from PIL import Image
    from hmdr.data import FINEFAKE, MDSM, VIEWS

    source, target = Path(args.input_dir), Path(args.output_dir)
    if source.resolve() == target.resolve():
        raise ValueError("Input and output directories must differ.")
    target.mkdir(parents=True, exist_ok=False)
    split_path = Path(args.split_ids or ("configs/splits/" + args.dataset + ".json"))
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    seen = set()
    for split in ["train", "val", "test"]:
        rows = []
        for line in (source / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row["id"])
            if identity in seen:
                raise ValueError("Duplicate ID across splits: " + identity)
            seen.add(identity)
            labels = row.get("labels")
            if labels is None:
                category = row["category"]
                values = FINEFAKE[int(category)] if args.dataset == "finefake" else MDSM[category]
                labels = dict(zip(VIEWS, values))
            if any(v not in labels or labels[v] not in (0, 1, None) for v in VIEWS):
                raise ValueError("Invalid view labels: " + identity)
            image = row.get("image")
            valid = image is not None
            if args.verify_images and valid:
                path = Path(image)
                path = path if path.is_absolute() else Path(args.image_root) / path
                try:
                    with Image.open(path) as handle:
                        handle.verify()
                except (OSError, ValueError):
                    valid = False
            rows.append({"id": identity, "text": row.get("text"), "image": image,
                         "labels": labels, "valid_text": bool(str(row.get("text") or "").strip()),
                         "media_valid": valid})
        by_id = {row["id"]: row for row in rows}
        missing = set(split_ids[split]) - set(by_id)
        if missing:
            raise ValueError("Annotations are missing IDs required by the fixed " + split + " split.")
        rows = [by_id[identity] for identity in split_ids[split]]
        with (target / (split + ".jsonl")).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({"split": split, "rows": len(rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["finefake", "mdsm_nyt"], required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--split-ids", help="Optional replacement split-ID JSON file.")
    parser.add_argument("--verify-images", action="store_true")
    main(parser.parse_args())
