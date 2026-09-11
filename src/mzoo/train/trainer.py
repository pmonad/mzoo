"""Trainer that also logs MoE router stats (aux loss, per-layer expert load) when present."""

import time

import torch
import transformers


class TimedCheckpoint(transformers.TrainerCallback):
    """Save a checkpoint at the last step, plus every `minutes` of wall time if set."""

    def __init__(self, minutes=None):
        self.seconds, self.last = (minutes * 60 if minutes else float("inf")), time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.last >= self.seconds or state.global_step >= state.max_steps:
            control.should_save, self.last = True, time.monotonic()


class Trainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux, self._counts = [], None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        if model.training and getattr(outputs, "aux_loss", None) is not None:
            self._track(outputs)
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def _track(self, outputs):
        self._aux.append(outputs.aux_loss.detach())
        k = self.model.config.num_experts_per_tok
        counts = torch.stack(
            [torch.bincount(r.topk(k).indices.flatten(), minlength=r.shape[-1]) for r in outputs.router_logits]
        )
        self._counts = counts if self._counts is None else self._counts + counts

    def log(self, logs, *args, **kwargs):
        if self._aux and "loss" in logs:
            cfg = self.model.config
            layers = [i for i in range(cfg.num_hidden_layers) if i not in (cfg.mlp_only_layers or [])]
            load = (self._counts / self._counts.sum(-1, keepdim=True)).tolist()
            logs["moe/aux_loss"] = torch.stack(self._aux).mean().item()
            for i, frac in zip(layers, load):
                logs[f"moe/max_load_l{i}"] = max(frac)
                logs[f"moe/min_load_l{i}"] = min(frac)
            self._aux, self._counts = [], None
        if "loss" in logs:
            self._log_router_load(logs)
        super().log(logs, *args, **kwargs)

    @torch.no_grad()
    def _log_router_load(self, logs):
        """Routers that count their own load (e.g. archs/nanbeige/moe.py) in a `load_counts` buffer."""
        for name, m in self.model.named_modules():
            if hasattr(m, "load_counts") and m.load_counts.sum() > 0:
                frac = m.load_counts / m.load_counts.sum()
                layer = name.split("layers.")[-1].split(".")[0]
                logs[f"moe/max_load_l{layer}"] = frac.max().item()
                logs[f"moe/min_load_l{layer}"] = frac.min().item()
                m.load_counts.zero_()
