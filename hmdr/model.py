"""HMDR class-token prediction and modality invariance learning."""

import torch
from torch import nn

from .attention import AttentionManager


def invariance_loss(groups, indices, targets):
    if not indices:
        return groups.sum() * 0.
    index = torch.tensor(indices, device=groups.device)
    values = []
    for group, name in [(0, "text"), (1, "image")]:
        student = groups[:, group, index, :]
        target = targets[name].detach()
        delta = (student - target).square().sum(-1)
        energy = student.square().sum(-1) + target.square().sum(-1)
        values.append((delta / energy.clamp_min(1e-8)).mean())
    return torch.stack(values).mean()


class HMDRModel(nn.Module):
    def __init__(self, model, roles, loss_weight=0.01, layers=None):
        super().__init__()
        self.model = model
        self.manager = AttentionManager(model, roles, layers=layers)
        self.loss_weight = loss_weight
        self.captured, self.counts, self.hooks = {}, {}, []
        self.last_pair_count = 0
        for layer, module in self.manager.modules.items():
            def capture(mod, inputs, output, layer=layer):
                index = self.counts.get(layer, 0)
                self.counts[layer] = index + 1
                if index < 3:
                    if output.ndim != 3 or output.shape[1] != 1:
                        raise ValueError("Unexpected group projection shape.")
                    self.captured.setdefault(layer, []).append(output[:, 0, :].float())
                return None

            self.hooks.append(module.o_proj.register_forward_hook(capture))

    def run_batch(self, batch, training):
        self.captured, self.counts = {}, {}
        inputs = self.manager.prepare(batch)
        self.manager.training = training
        output = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        if not all(self.counts.get(i) == 4 and len(self.captured.get(i, [])) == 3
                   for i in self.manager.modules):
            raise RuntimeError("Missing HMDR group projections.")
        return output.logits[:, -1, :].float()

    def groups(self):
        return torch.stack([torch.stack(self.captured[i]) for i in sorted(self.manager.modules)])

    def forward(self, item):
        indices = item.get("reference_indices", [])
        self.last_pair_count = len(indices)
        targets = {}
        if indices:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    for name, group in [("text", 0), ("image", 1)]:
                        self.run_batch(item[name + "_reference"], False)
                        targets[name] = self.groups()[:, group].detach().clone()
            finally:
                self.model.train(was_training)
        logits = self.run_batch(item["batch"], True)
        ce = nn.functional.cross_entropy(logits, item["batch"]["class_token_ids"])
        consistency = invariance_loss(self.groups(), indices, targets)
        return ce + self.loss_weight * consistency, ce, consistency, logits

    def evaluate_batch(self, batch):
        return self.run_batch(batch, False)

    def detach(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self.manager.detach()
