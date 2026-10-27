"""Modality head decomposition and regulation in Qwen3.5 full-attention layers."""

import types

import torch
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv


class AttentionManager:
    def __init__(self, model, roles=None, probe=False, layers=None):
        self.probe = probe
        self.active = True
        self.context = None
        self.role_data = roles
        self.training = True
        self.terms, self.latest, self.stats = [], [], {}
        self.modules = {
            int(module.layer_idx): module for module in model.modules()
            if module.__class__.__name__ == "Qwen3_5Attention"
        }
        if not self.modules or (layers is not None and sorted(self.modules) != layers):
            raise ValueError("Unexpected full-attention layer layout.")
        if not probe and (roles is None or set(roles["layers"]) != {str(i) for i in self.modules}):
            raise ValueError("Calibrated roles do not match the backbone layers.")
        self.originals = []
        for module in self.modules.values():
            self.originals.append((module, module.forward))
            manager = self

            def forward(mod, hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, **kwargs):
                return manager.forward(mod, hidden_states, position_embeddings,
                                       attention_mask, past_key_values, **kwargs)

            module.forward = types.MethodType(forward, module)

    def prepare(self, batch):
        self.context = (batch["hmdr_text_mask"], batch["hmdr_image_mask"],
                        batch["attention_mask"].bool())
        self.terms, self.latest = [], []
        return {k: v for k, v in batch.items()
                if not k.startswith("hmdr_") and k != "class_token_ids"}

    def weights(self, layer, device, dtype):
        p = torch.tensor(self.role_data["layers"][str(layer)]["p_text"],
                         device=device, dtype=dtype)
        return torch.stack([p * p, (1 - p) ** 2, 2 * p * (1 - p)], 0)

    def forward(self, mod, hidden, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        if not self.active:
            original = next(f for m, f in self.originals if m is mod)
            return original(hidden, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        if past_key_values is not None:
            raise ValueError("HMDR requires use_cache=False.")
        batch, length, _ = hidden.shape
        n_heads, dim = mod.config.num_attention_heads, mod.head_dim
        q, gate = torch.chunk(mod.q_proj(hidden).view(batch, length, n_heads, 2 * dim), 2, -1)
        q = mod.q_norm(q).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(hidden).view(batch, length, -1, dim)).transpose(1, 2)
        v = mod.v_proj(hidden).view(batch, length, -1, dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        k = repeat_kv(k, mod.num_key_value_groups)
        v = repeat_kv(v, mod.num_key_value_groups)
        tm, im, valid = self.context
        visibility = torch.stack(
            [tm.any(-1), im.any(-1), tm.any(-1) & im.any(-1)], -1
        ).to(hidden.dtype)
        with torch.no_grad():
            scores = (q[:, :, -1:, :].float() @ k.float().transpose(-1, -2)) * mod.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, -1:, :].float()
            else:
                scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
            probs = scores.softmax(-1).squeeze(2)
        if self.probe:
            image_density = (probs * im[:, None, :]).sum(-1) / im.sum(-1)[:, None].clamp_min(1)
            text_density = (probs * tm[:, None, :]).sum(-1) / tm.sum(-1)[:, None].clamp_min(1)
            state = self.stats.setdefault(mod.layer_idx, {
                "text": torch.zeros(n_heads, dtype=torch.float64),
                "image": torch.zeros(n_heads, dtype=torch.float64), "n": 0,
            })
            state["text"] += text_density.sum(0).cpu().double()
            state["image"] += image_density.sum(0).cpu().double()
            state["n"] += batch
        heads = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.,
            is_causal=attention_mask is None, scale=mod.scaling,
        ).transpose(1, 2)
        heads = heads * gate.sigmoid()
        flat = heads.reshape(batch, length, n_heads * dim)
        if self.probe:
            return mod.o_proj(flat), None
        weights = self.weights(mod.layer_idx, hidden.device, hidden.dtype)
        group_masks = weights[:, :, None].expand(3, n_heads, dim).reshape(3, n_heads * dim)
        groups = torch.stack([mod.o_proj(flat[:, -1:, :] * mask) for mask in group_masks], 0)
        mass = probs.sum(-1).clamp_min(1e-8)
        text = (probs * tm[:, None, :]).sum(-1) / mass
        image = (probs * im[:, None, :]).sum(-1) / mass
        role_mass = torch.stack([
            (text * weights[0]).sum(-1), (image * weights[1]).sum(-1),
            ((text + image) * weights[2]).sum(-1),
        ], -1)
        shares = role_mass / (role_mass.sum(-1, keepdim=True) + 1e-8)
        compensation = 1. + 1. / 3. - shares
        gains = (visibility.float() * compensation) @ weights.float()
        gains = gains / gains.mean(-1, keepdim=True).clamp_min(1e-8)
        regulated = heads * gains[:, None, :, None].to(heads.dtype)
        missing = ~(visibility[:, 0].bool() & visibility[:, 1].bool())
        routed = torch.where(missing[:, None, None, None], heads, regulated)
        output = mod.o_proj(routed.reshape(batch, length, n_heads * dim))
        z = groups[:, :, -1, :].float()
        normalized = F.normalize(z, dim=-1, eps=1e-8)
        decorrelation = sum(
            (normalized[g] * normalized[h]).sum(-1).square().mean()
            for g, h in [(0, 1), (0, 2), (1, 2)]
        ) / 3
        if self.training:
            self.terms.append(decorrelation)
        with torch.no_grad():
            self.latest.append({
                "layer": mod.layer_idx, "group_norm": z.norm(dim=-1).mean(-1).cpu().tolist(),
                "decorrelation": float(decorrelation.detach()),
            })
        return output, None

    def roles(self):
        result = {}
        for layer, state in self.stats.items():
            text, image = state["text"] / state["n"], state["image"] / state["n"]
            p = (text / (text + image + 1e-12)).tolist()
            result[str(layer)] = {"p_text": p, "n": state["n"]}
        if set(result) != {str(i) for i in self.modules}:
            raise ValueError("Incomplete calibration statistics.")
        return {"layers": result}

    def detach(self):
        for module, forward in self.originals:
            module.forward = forward
