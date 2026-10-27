"""Training utilities and adapter checkpoints."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(config, device):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 2))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def prepared_batches(rows, collator, size):
    if not rows:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(collator, rows[:size])
        for start in range(0, len(rows), size):
            item = future.result()
            if start + size < len(rows):
                future = pool.submit(collator, rows[start + size:start + 2 * size])
            yield start, item


def optimizer_and_scheduler(model, config, total_steps):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["lr"], weight_decay=config["weight_decay"],
        betas=tuple(config["optimizer_betas"]),
    )
    warmup = int(total_steps * config["warmup_ratio"])

    def scale(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1., max(0., (step - warmup) / max(1, total_steps - warmup)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, scale), params


def save_checkpoint(model, processor, roles, config, out, epoch, best):
    out = Path(out).resolve()
    target, previous = out / "best_adapter", None
    if target.is_symlink():
        name = os.readlink(target)
        previous = out / name
        if Path(name).name != name or not name.startswith(".best-adapter-") \
                or previous.is_symlink() or previous.resolve().parent != out:
            raise ValueError("Unexpected adapter pointer.")
    elif target.exists():
        raise FileExistsError(target)
    token = uuid.uuid4().hex
    staging, link = out / (".best-adapter-" + token), out / (".adapter-link-" + token)
    staging.mkdir(exist_ok=False)
    published = False
    try:
        model.save_pretrained(staging, safe_serialization=True)
        processor.save_pretrained(staging / "processor")
        write_json(staging / "roles.json", roles)
        write_json(staging / "run_config.json", config)
        write_json(staging / "best_epoch.json", {"epoch": epoch, "validation_macro_f1": best})
        if not (staging / "adapter_model.safetensors").is_file():
            raise RuntimeError("Incomplete adapter checkpoint.")
        for item in staging.rglob("*"):
            if item.is_file():
                with item.open("rb") as handle:
                    os.fsync(handle.fileno())
        os.symlink(staging.name, link)
        os.replace(link, target)
        published = True
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        if link.is_symlink():
            link.unlink()
        if not published and staging.exists():
            shutil.rmtree(staging)
    return target
