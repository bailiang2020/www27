"""Train HMDR on complete image-text news and select by validation Macro-F1."""

import argparse
import json
import math
from pathlib import Path
import random
import time


def main(args):
    import torch
    from hmdr import HMDRModel
    from hmdr.backbone import load_backbone, load_processor
    from hmdr.data import Collator, PairedCollator, load_rows, to_device
    from hmdr.evaluation import predict, summarize
    from hmdr.runtime import (configure, optimizer_and_scheduler, prepared_batches,
                              read_json, save_checkpoint, sha256, write_json)

    config = read_json(args.config)
    configure(config, args.device)
    rows, validation = load_rows(config, "train"), load_rows(config, "val")
    if not rows or {r["label"] for r in validation} != {0, 1}:
        raise ValueError("Training must be nonempty and validation must contain both classes.")
    if {r["id"] for r in rows} & {r["id"] for r in validation}:
        raise ValueError("Training and validation IDs overlap.")
    calibration = Path(config["calibration_dir"])
    roles, partners = read_json(calibration / "roles.json"), read_json(calibration / "partners.json")
    if roles["train_sha256"] != sha256(Path(config["data_dir"]) / "train.jsonl"):
        raise ValueError("Calibration was generated from a different training manifest.")
    if not partners:
        raise ValueError("Training reference partners are empty.")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run_config.json", config)
    write_json(out / "roles.json", roles)
    processor = load_processor(config)
    model = load_backbone(config, args.device, trainable=True).to(args.device)
    if any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
        raise ValueError("Expected FP32 trainable LoRA parameters.")
    engine = HMDRModel(model, roles, config["hmdr"]["lambda"], config["hmdr"]["layers"])
    collator = PairedCollator(processor, {r["id"]: r for r in rows}, partners)
    eval_collator = Collator(processor)
    total_steps = config["epochs"] * math.ceil(len(rows) / config["micro_batch"])
    optimizer, scheduler, params = optimizer_and_scheduler(model, config, total_steps)
    rng, best, updates = random.Random(config["seed"]), -float("inf"), 0
    try:
        for epoch in range(config["epochs"]):
            rng.shuffle(rows)
            engine.train()
            start_time, seen, matched, ce_sum, reg_sum = time.monotonic(), 0, 0, 0., 0.
            for _, item in prepared_batches(rows, collator, config["micro_batch"]):
                item = to_device(item, args.device)
                optimizer.zero_grad(set_to_none=True)
                loss, ce, consistency, _ = engine(item)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, config["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                updates += 1
                count = len(item["rows"])
                seen += count
                matched += engine.last_pair_count
                ce_sum += float(ce.detach()) * count
                reg_sum += float(consistency.detach()) * count
                if updates == 1 or updates % 25 == 0:
                    print(json.dumps({"epoch": epoch + 1, "step": updates, "total_steps": total_steps,
                                      "ce": float(ce.detach()), "mil": float(consistency.detach()),
                                      "gradient_norm": float(norm)}), flush=True)
            predictions = predict(engine, validation, eval_collator, processor,
                                  config["eval_batch"], args.device)
            metrics = summarize(predictions)
            record = {"epoch": epoch + 1, "updates": updates, "ce": ce_sum / seen,
                      "mil": reg_sum / seen, "matched_pairs": matched,
                      "validation_TI": metrics, "seconds": time.monotonic() - start_time}
            with (out / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                save_checkpoint(model, processor, roles, config, out, epoch + 1, best)
            print(json.dumps(record), flush=True)
        if updates != total_steps:
            raise RuntimeError("Optimizer update count mismatch.")
        write_json(out / "training_metrics.json", {"updates": updates, "best_validation_macro_f1": best})
    finally:
        engine.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
