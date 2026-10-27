"""Qwen3.5 backbone and language-model LoRA parameters."""

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def load_processor(config, processor_path=None):
    processor = AutoProcessor.from_pretrained(
        str(processor_path or config["backbone"]),
        local_files_only=config.get("local_files_only", False),
    )
    settings = config["model"]
    processor.image_processor.size = {
        "shortest_edge": settings["min_pixels"],
        "longest_edge": settings["max_pixels"],
    }
    processor.hmdr_max_text_tokens = settings["max_text_tokens"]
    processor.tokenizer.padding_side = "left"
    return processor


def attach_lora(model, settings):
    targets = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "language_model.layers." in name
    ]
    if not targets:
        raise ValueError("No Qwen language modules found.")
    model = get_peft_model(model, LoraConfig(
        r=settings["r"], lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"], target_modules=targets,
        bias="none", task_type="CAUSAL_LM",
    ))
    if any(p.requires_grad for n, p in model.named_parameters() if ".visual." in n):
        raise ValueError("The vision encoder must remain frozen.")
    return model


def load_backbone(config, device="cuda:0", adapter=None, trainable=False):
    dtype = (torch.bfloat16 if str(device).startswith("cuda")
             and config["precision"] == "bf16" else torch.float32)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        config["backbone"], dtype=dtype,
        local_files_only=config.get("local_files_only", False),
        attn_implementation="sdpa",
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(
            model, str(adapter), is_trainable=trainable, torch_device="cpu",
        )
    elif trainable:
        model = attach_lora(model, config["model"])
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    if hasattr(model, "disable_input_require_grads"):
        model.disable_input_require_grads()
    if model.is_gradient_checkpointing:
        raise RuntimeError("Unexpected gradient checkpointing.")
    return model
