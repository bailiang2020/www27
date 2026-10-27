"""Evaluate a saved HMDR checkpoint on TI, text-only and image-only views."""

import argparse
import json
from pathlib import Path


def main(args):
    from hmdr import HMDRModel
    from hmdr.backbone import load_backbone, load_processor
    from hmdr.data import Collator, load_rows
    from hmdr.evaluation import predict, summarize
    from hmdr.runtime import configure, read_json, write_json

    checkpoint = Path(args.checkpoint)
    config = read_json(args.config or checkpoint / "run_config.json")
    configure(config, args.device)
    roles = read_json(checkpoint / "roles.json")
    processor = load_processor(config, checkpoint / "processor")
    model = load_backbone(config, args.device, adapter=checkpoint).to(args.device)
    engine = HMDRModel(model, roles, config["hmdr"]["lambda"], config["hmdr"]["layers"]).eval()
    collator = Collator(processor)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    results = {}
    try:
        for view in args.views:
            rows = load_rows(config, args.split, view)
            predictions = predict(engine, rows, collator, processor, config["eval_batch"], args.device)
            with (out / (view + "-predictions.jsonl")).open("w", encoding="utf-8") as handle:
                for row in predictions:
                    handle.write(json.dumps(row) + "\n")
            results[view] = summarize(predictions)
        if all(view in results for view in ["T", "I"]):
            results["Avg_uni"] = {}
            for key in ["accuracy", "macro_f1"]:
                values = [results[v][key] for v in ["T", "I"]]
                results["Avg_uni"][key] = (sum(values) / 2 if all(x is not None for x in values) else None)
        write_json(out / "metrics.json", results)
        print(json.dumps(results, indent=2))
    finally:
        engine.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="Optional configuration with updated data/backbone paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--views", nargs="+", choices=["TI", "T", "I"], default=["TI", "T", "I"])
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
