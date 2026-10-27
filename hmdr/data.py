"""Manifest loading, visibility masks and complete-input reference pairing."""

import json
from functools import lru_cache
from pathlib import Path

from PIL import Image
import torch


INSTRUCTION = (
    "Classify the authenticity of the provided content using only the available input. "
    "Output exactly 0 for real or 1 for fake. Do not explain."
)
FINEFAKE = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, None, None),
            3: (1, 1, None), 4: (1, None, 1), 5: (1, None, None)}
MDSM = {"orig": (0, 0, 0), "face_swap": (1, None, 1),
        "face_attribute": (1, 0, 1), "text_swap": (1, 1, 0),
        "face_swap&text_swap": (1, 1, 1), "face_attribute&text_swap": (1, 1, 1)}
VIEWS = ("TI", "T", "I")


@lru_cache(maxsize=65536)
def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def load_rows(config, split, view="TI"):
    if view not in VIEWS or (split == "train" and view != "TI"):
        raise ValueError("Training requires complete inputs; views are TI, T and I.")
    manifest = Path(config["data_dir"]) / (split + ".jsonl")
    image_root = Path(config.get("image_root", manifest.parent))
    split_file = config.get("split_ids")
    expected = (json.loads(Path(split_file).read_text(encoding="utf-8"))[split]
                if split_file else None)
    media = (json.loads(Path(config["media_report"]).read_text(encoding="utf-8"))["image_validity"]
             if config.get("media_report") else None)
    result, seen = [], set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row["id"])
            if identity in seen:
                raise ValueError("Duplicate manifest ID: " + identity)
            seen.add(identity)
            labels = row["labels"]
            if any(v not in labels or labels[v] not in (0, 1, None) for v in VIEWS):
                raise ValueError("Expected explicit TI/T/I labels, with null for unknown.")
            label = labels[view]
            if label is None:
                continue
            text = row.get("text") if "T" in view else None
            image = row.get("image") if "I" in view else None
            if "T" in view and (not isinstance(text, str) or not text.strip()
                                or not row.get("valid_text", True)):
                continue
            if "I" in view:
                if image is None:
                    continue
                media_valid = media[image] if media is not None else row.get("media_valid")
                if media_valid is False:
                    continue
                path = Path(image)
                image = str(path if path.is_absolute() else image_root / path)
                if media_valid is None and not valid_image(image):
                    continue
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
            result.append({"id": identity, "text": text, "image": image, "label": label})
    if expected is not None:
        if seen != set(expected):
            raise ValueError("Manifest IDs do not match the configured data split.")
        order = {identity: i for i, identity in enumerate(expected)}
        result.sort(key=lambda row: order[row["id"]])
    return result


def class_token_ids(processor):
    ids = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in [0, 1]]
    if any(len(value) != 1 for value in ids):
        raise ValueError("Each class label must tokenize to exactly one token.")
    return [value[0] for value in ids]


def messages(row, processor):
    content = [{"type": "text", "text":
                "Text: " + ("Present" if row["text"] is not None else "Missing")
                + ". Image: " + ("Present" if row["image"] is not None else "Missing") + "."}]
    if row["image"] is not None:
        with Image.open(row["image"]) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    if row["text"] is not None:
        ids = processor.tokenizer.encode(row["text"], add_special_tokens=False)
        ids = ids[:getattr(processor, "hmdr_max_text_tokens", 768)]
        content.append({"type": "text", "text": "News text: " + processor.tokenizer.decode(ids)})
    content.append({"type": "text", "text": INSTRUCTION})
    return [{"role": "user", "content": content}]


def encode(rows, processor, training=False):
    if not rows:
        raise ValueError("Empty batch.")
    if training and any(r["text"] is None or r["image"] is None
                        or r["label"] not in [0, 1] for r in rows):
        raise ValueError("Complete TI binary supervision only.")
    processor.tokenizer.padding_side = "left"
    batch = processor.apply_chat_template(
        [messages(row, processor) for row in rows], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        processor_kwargs={"padding": True}, return_dict=True, return_tensors="pt",
    )
    if not batch["attention_mask"][:, -1].bool().all():
        raise ValueError("The last prompt position must be valid.")
    if training:
        ids = class_token_ids(processor)
        batch["class_token_ids"] = torch.tensor([ids[r["label"]] for r in rows], dtype=torch.long)
    text = torch.zeros_like(batch["input_ids"], dtype=torch.bool)
    image = batch["input_ids"].eq(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    for b, row in enumerate(rows):
        if row["text"] is None:
            continue
        ids = batch["input_ids"][b].tolist()
        decoded = processor.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False,
        )
        aligned = processor.tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
        if aligned["input_ids"] != ids:
            raise ValueError("Token round-trip failed; cannot infer news masks.")
        begin = decoded.index("News text: ") + len("News text: ")
        end = decoded.rfind(INSTRUCTION)
        if end < begin:
            raise ValueError("News boundary not found.")
        for j, (start, stop) in enumerate(aligned["offset_mapping"]):
            if stop > begin and start < end:
                text[b, j] = True
    valid = batch["attention_mask"].bool()
    batch["hmdr_text_mask"] = text & valid & ~image
    batch["hmdr_image_mask"] = image & valid
    for b, row in enumerate(rows):
        if bool(text[b].any()) != (row["text"] is not None):
            raise ValueError("Text visibility/mask mismatch.")
        if bool(image[b].any()) != (row["image"] is not None):
            raise ValueError("Image visibility/mask mismatch.")
    return batch


class Collator:
    def __init__(self, processor, training=False):
        self.processor, self.training = processor, training

    def __call__(self, rows):
        return {"batch": encode(rows, self.processor, self.training), "rows": rows}


def select_encoded(batch, selected):
    count = batch["input_ids"].shape[0]
    index = torch.tensor(selected, dtype=torch.long)
    grids = batch["image_grid_thw"]
    if len(grids) != count:
        raise ValueError("References require one image per example.")
    offsets = [0]
    for length in grids.prod(-1).tolist():
        offsets.append(offsets[-1] + int(length))
    result = {}
    for key, value in batch.items():
        if key == "pixel_values":
            if offsets[-1] != len(value):
                raise ValueError("Image patch count mismatch.")
            result[key] = torch.cat([value[offsets[i]:offsets[i + 1]] for i in selected], 0)
        elif value.shape[0] == count:
            result[key] = value.index_select(0, index)
        else:
            raise ValueError(("Unrecognized encoded field", key, value.shape))
    return result


class PairedCollator(Collator):
    def __init__(self, processor, bank, partners):
        super().__init__(processor, True)
        self.bank, self.partners = bank, partners
        for identity, partner in partners.items():
            if identity == partner or identity not in bank or partner not in bank:
                raise ValueError("Reference partners must be distinct training examples.")

    def __call__(self, rows):
        item = super().__call__(rows)
        indices, texts, images = [], [], []
        for j, row in enumerate(rows):
            partner = self.partners.get(row["id"])
            if partner is None:
                continue
            other = self.bank[partner]
            texts.append({"id": row["id"] + "::swap-image", "text": row["text"], "image": other["image"]})
            images.append({"id": row["id"] + "::swap-text", "text": other["text"], "image": row["image"]})
            indices.append(j)
        item["reference_indices"] = indices
        if indices:
            item["text_reference"] = encode(texts, self.processor)
            item["image_reference"] = encode(images, self.processor)
            expected = item["batch"]["attention_mask"][indices].sum(-1)
            keep = ((item["text_reference"]["attention_mask"].sum(-1) == expected)
                    & (item["image_reference"]["attention_mask"].sum(-1) == expected)).tolist()
            selected = [j for j, good in enumerate(keep) if good]
            item["reference_indices"] = [indices[j] for j in selected]
            if selected:
                for key in ["text_reference", "image_reference"]:
                    item[key] = select_encoded(item[key], selected)
            else:
                del item["text_reference"], item["image_reference"]
        return item


def to_device(item, device):
    result = dict(item)
    for key in ["batch", "text_reference", "image_reference"]:
        if key in result:
            result[key] = {k: v.to(device, non_blocking=True) for k, v in result[key].items()}
    return result
