"""Calibrate fixed head preferences and training reference partners."""

import argparse
import json
from pathlib import Path
import random


def main(args):
    import torch
    from hmdr.attention import AttentionManager
    from hmdr.backbone import load_backbone, load_processor
    from hmdr.data import encode, load_rows
    from hmdr.runtime import configure, read_json, sha256, write_json

    config = read_json(args.config)
    configure(config, args.device)
    rows = load_rows(config, "train")
    if len(rows) < 2:
        raise ValueError("At least two complete training examples are required.")
    out = Path(config["calibration_dir"])
    out.mkdir(parents=True, exist_ok=True)
    processor = load_processor(config)
    partners = read_json(args.partners or config["partners_path"])
    bank = {row["id"] for row in rows}
    if not partners or any(a == b or a not in bank or b not in bank for a, b in partners.items()):
        raise ValueError("No valid distinct training reference partners.")
    selected = list(rows)
    random.Random(config["calibration_seed"]).shuffle(selected)
    selected = selected[:config["hmdr"]["calibration_samples"]]
    model = load_backbone(config, args.device).to(args.device).eval()
    manager = AttentionManager(model, probe=True, layers=config["hmdr"]["layers"])
    manager.training = False
    try:
        with torch.inference_mode():
            for start in range(0, len(selected), config["eval_batch"]):
                batch = encode(selected[start:start + config["eval_batch"]], processor)
                batch = {k: v.to(args.device) for k, v in batch.items()}
                model(**manager.prepare(batch), use_cache=False, logits_to_keep=1)
                print(json.dumps({"stage": "calibration", "processed":
                                  min(start + config["eval_batch"], len(selected))}), flush=True)
        roles = manager.roles()
    finally:
        manager.detach()
    roles.update({
        "ids": [row["id"] for row in selected],
        "train_sha256": sha256(Path(config["data_dir"]) / "train.jsonl"),
    })
    write_json(out / "roles.json", roles)
    write_json(out / "partners.json", partners)
    print(json.dumps({"calibration_samples": len(selected), "reference_pairs": len(partners)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--partners", help="Optional existing training partner JSON map.")
    main(parser.parse_args())
