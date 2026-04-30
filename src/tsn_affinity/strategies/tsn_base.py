"""Base TSN strategy with mask management and quantization."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.data.batch_generator import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.sparse.module_converter import (
    SparseConversionConfig,
    convert_to_sparse,
    iter_sparse_modules,
    kmeans_quantize,
    rebuild_optimizer,
)
from tsn_affinity.strategies.base_strategy import BaseStrategy
from tsn_affinity.strategies.training_utils import (
    restore_frozen_parameters,
    snapshot_frozen_parameters,
    verify_frozen_gradient_zeroing,
    zero_gradients_for_frozen_params,
    zero_gradients_for_non_maskable_params,
)


class TSNBaseStrategy(BaseStrategy):
    """Base TSN strategy with sparse mask management and quantization.

    This class provides the core TSN functionality:
    - Trainable score masks on sparse layers
    - Per-task binary mask storage
    - Frozen weight protection for previous tasks
    - Optional K-means quantization after each task
    - Equal-share keep_ratio schedule across tasks

    Attributes:
        sparse_config: Sparse layer configuration.
        current_task_id: ID of the current task being trained.
        per_task_masks: Dict of task_id -> mask dict.
        consolidated_masks: Dict of param_name -> mask (union of all tasks).
        task_codebooks: Dict of task_id -> codebook dict.
        current_keep_ratio: Current keep ratio for training.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        sparse_config: SparseConfig | None = None,
    ) -> None:
        if sparse_config is None:
            sparse_config = SparseConfig()

        self.sparse_config = sparse_config
        self.sparse_conversion_config = SparseConversionConfig(
            keep_ratio=float(sparse_config.keep_ratio),
            include_embeddings=bool(sparse_config.include_embeddings),
            allow_weight_reuse=bool(sparse_config.allow_weight_reuse),
            skip_module_names=tuple(sparse_config.skip_module_names),
            quant_clusters=int(sparse_config.quant_clusters),
            quantize_after_task=bool(sparse_config.quantize_after_task),
        )

        super().__init__(obs_shape, n_actions, seq_len, device, model_config)

        convert_to_sparse(self.model, self.sparse_conversion_config)
        self.optimizer = rebuild_optimizer(self.optimizer, self.model.parameters())

        self.current_keep_ratio = float(sparse_config.keep_ratio)
        self.current_task_id: int = 0
        self.per_task_masks: Dict[int, Dict[str, Optional[torch.Tensor]]] = {}
        self.consolidated_masks: Dict[str, Optional[torch.Tensor]] = {}
        self.task_codebooks: Dict[int, Dict[str, np.ndarray]] = {}
        self.active_eval_task: Optional[int] = None
        self.task_keep_ratios: Dict[int, float] = {}

        self._refresh_name_sets()
        self.clear_eval_task()

    def _refresh_name_sets(self) -> None:
        self.maskable_param_names = set()
        self.score_param_names = set()
        for mod_name, mod in iter_sparse_modules(self.model):
            self.maskable_param_names.add(f"{mod_name}.weight")
            self.score_param_names.add(f"{mod_name}.score")
            if getattr(mod, "bias", None) is not None:
                self.maskable_param_names.add(f"{mod_name}.bias")
            if getattr(mod, "bias_score", None) is not None:
                self.score_param_names.add(f"{mod_name}.bias_score")

    def _reset_all_scores(self) -> None:
        for _, mod in iter_sparse_modules(self.model):
            mod.reset_scores()
            mod.clear_active_masks()

    def _sync_occupied_masks_into_modules(self) -> None:
        for name, mod in iter_sparse_modules(self.model):
            mod.clear_occupied_masks()
            w_key = f"{name}.weight"
            b_key = f"{name}.bias"
            if w_key in self.consolidated_masks and self.consolidated_masks[w_key] is not None:
                mod.occupied_weight_mask = self.consolidated_masks[w_key].detach().clone()
            if b_key in self.consolidated_masks and self.consolidated_masks[b_key] is not None:
                mod.occupied_bias_mask = self.consolidated_masks[b_key].detach().clone()

    def _recompute_module_masks(self, mod) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            if hasattr(mod, "weight_mask"):
                mod.weight_mask = None
            if hasattr(mod, "bias_mask"):
                mod.bias_mask = None

            w_mask = mod.current_weight_mask()
            b_mask = mod.current_bias_mask() if getattr(mod, "bias", None) is not None else None

        return w_mask, b_mask

    def _collect_current_task_masks(self) -> Dict[str, Optional[torch.Tensor]]:
        out: Dict[str, Optional[torch.Tensor]] = {}
        for name, mod in iter_sparse_modules(self.model):
            w_mask, b_mask = self._recompute_module_masks(mod)
            out[f"{name}.weight"] = None if w_mask is None else w_mask.detach().cpu().clone()
            out[f"{name}.bias"] = None if b_mask is None else b_mask.detach().cpu().clone()
        return out

    def _update_consolidated_masks(self, task_masks: Dict[str, Optional[torch.Tensor]]) -> None:
        if not self.consolidated_masks:
            self.consolidated_masks = {k: (None if v is None else v.clone()) for k, v in task_masks.items()}
            return
        for key, mask in task_masks.items():
            if mask is None:
                self.consolidated_masks.setdefault(key, None)
                continue
            if self.consolidated_masks.get(key) is None:
                self.consolidated_masks[key] = mask.clone()
            else:
                self.consolidated_masks[key] = torch.logical_or(
                    self.consolidated_masks[key].bool(),
                    mask.bool(),
                ).to(torch.uint8)

    def _apply_eval_masks(self, task_id: Optional[int]) -> None:
        task_masks = self.per_task_masks.get(task_id, None) if task_id is not None else None
        for name, mod in iter_sparse_modules(self.model):
            if task_masks is None:
                mod.clear_active_masks()
                continue
            mod.active_weight_mask = task_masks.get(f"{name}.weight", None)
            mod.active_bias_mask = task_masks.get(f"{name}.bias", None)

    def set_eval_task(self, task_id: int) -> None:
        task_id = int(task_id)
        self.active_eval_task = task_id if task_id in self.per_task_masks else None
        self._apply_eval_masks(self.active_eval_task)

    def clear_eval_task(self) -> None:
        self.active_eval_task = None
        self._apply_eval_masks(None)

    def has_task_mask(self, task_id: int) -> bool:
        return int(task_id) in self.per_task_masks

    def _compute_task_keep_ratio(self) -> float:
        return float(max(1e-3, min(1.0, self.sparse_config.keep_ratio)))

    def _apply_keep_ratio_to_modules(self, keep_ratio: float) -> None:
        self.current_keep_ratio = float(keep_ratio)
        for _, mod in iter_sparse_modules(self.model):
            mod.keep_ratio = float(keep_ratio)

    def _prepare_current_task(self) -> None:
        self.clear_eval_task()
        self._sync_occupied_masks_into_modules()

        keep_ratio = self._compute_task_keep_ratio()
        self._apply_keep_ratio_to_modules(keep_ratio)
        self.task_keep_ratios[self.current_task_id] = float(keep_ratio)

        if self.current_task_id > 0:
            self._reset_all_scores()
            self.optimizer = rebuild_optimizer(self.optimizer, self.model.parameters())

    def _quantize_new_weights(
        self,
        task_masks: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, np.ndarray]:
        codebooks: Dict[str, np.ndarray] = {}
        if not self.sparse_config.quantize_after_task:
            return codebooks

        name_to_module = dict(self.model.named_modules())
        for key, mask in task_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            module_name, _ = key.rsplit(".", 1)
            module = name_to_module.get(module_name, None)
            if module is None:
                continue
            prev_mask = self.consolidated_masks.get(key, None)
            if prev_mask is None:
                new_mask = mask.bool()
            else:
                new_mask = torch.logical_and(mask.bool(), ~prev_mask.bool())
            if int(new_mask.sum().item()) == 0:
                continue
            centers = kmeans_quantize(module.weight, new_mask, self.sparse_config.quant_clusters)
            if centers is not None:
                codebooks[key] = centers
        return codebooks

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        self._prepare_current_task()
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        last_loss = None
        for iteration in range(int(steps)):
            obs, actions, rtg, ts, mask = next(loader)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_cross_entropy(logits, actions, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            zero_gradients_for_frozen_params(self.model, self.consolidated_masks)
            name_to_module = dict(self.model.named_modules())
            is_valid, msg = verify_frozen_gradient_zeroing(
                self.model, self.consolidated_masks, name_to_module
            )
            if not is_valid:
                raise RuntimeError(f"Frozen gradient zeroing failed: {msg}")
            zero_gradients_for_non_maskable_params(
                self.model,
                self.maskable_param_names,
                self.score_param_names,
                freeze_non_maskable_after_first=True,
                current_task_id=self.current_task_id,
            )

            frozen_snapshot = snapshot_frozen_parameters(
                self.model,
                self.consolidated_masks,
                self.maskable_param_names,
                self.score_param_names,
                freeze_non_maskable_params=True,
                current_task_id=self.current_task_id,
            )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            restore_frozen_parameters(self.model, frozen_snapshot)

            last_loss = float(loss.detach().item())
            if iteration % 20000 == 0 or iteration == int(steps) - 1:
                print(
                    f"[tsn] task={self.current_task_id} iteration={iteration} "
                    f"ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f}"
                )

        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]) -> None:
        task_id = int(self.current_task_id)

        task_masks = self._collect_current_task_masks()
        self.per_task_masks[task_id] = task_masks
        self.task_codebooks[task_id] = self._quantize_new_weights(task_masks)
        self._update_consolidated_masks(task_masks)

        used = 0
        total = 0
        for key, mask in self.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        print(
            f"[tsn] after task {task_id}: "
            f"occupied_ratio={ratio:.4f} keep_ratio={self.task_keep_ratios.get(task_id, self.current_keep_ratio):.4f}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1