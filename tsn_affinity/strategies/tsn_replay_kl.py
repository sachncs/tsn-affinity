"""TSN with replay-memory KL routing (TSN-ReplayKL)."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tsn_affinity.core.config import ModelConfig, SparseConfig
from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.data.batch import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.sparse.converter import convert_to_sparse
from tsn_affinity.strategies.model_copy import ModelCopy
from tsn_affinity.strategies.training_utils import (
    restore_frozen_parameters,
    snapshot_frozen_parameters,
    zero_gradients_for_frozen_params,
    zero_gradients_for_non_maskable_params,
)
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy


@dataclass
class ReplayKLRoutingConfig:
    """Configuration for replay-memory KL routing.

    Attributes:
        memory_size: Number of observations to store per task.
        kl_threshold: KL divergence threshold for creating new copy.
        max_copies: Maximum number of model copies.
    """

    memory_size: int = 256
    kl_threshold: float = 0.25
    max_copies: int | None = None


class TSNReplayKLStrategy(TSNBaseStrategy):
    """TSN with replay-memory KL routing (TSN-ReplayKL).

    Routes new tasks to existing model copies based on KL similarity
    between observation replay memories. Creates a new copy when KL
    exceeds the threshold. Otherwise reuses the most similar copy.

    This corresponds to the "TSN-ReplayKL" variant in the paper.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        sparse_config: SparseConfig | None = None,
        routing_config: ReplayKLRoutingConfig | None = None,
    ) -> None:
        if routing_config is None:
            routing_config = ReplayKLRoutingConfig()

        super().__init__(
            obs_shape, n_actions, seq_len, device, model_config, sparse_config
        )

        self.routing_config = routing_config

        self.task_memories: dict[int, torch.Tensor] = {}
        self.task_to_copy: dict[int, int] = {}
        self.copy_states: list[ModelCopy] = []

        initial_copy = ModelCopy(
            model=self.model,
            optimizer=self.optimizer,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )
        self.copy_states.append(initial_copy)
        self.current_copy_id: int = 0

    def _build_task_memory(self, task_trajs: list[Trajectory]) -> torch.Tensor:
        obs_chunks = []
        total = 0
        for tr in task_trajs:
            obs = np.asarray(tr.obs, dtype=np.float32)
            if obs.ndim < 2:
                continue
            obs_chunks.append(obs)
            total += int(obs.shape[0])
            if total >= self.routing_config.memory_size * 4:
                break

        if not obs_chunks:
            raise ValueError("Cannot build task memory from empty observations")

        obs_all = np.concatenate(obs_chunks, axis=0)
        n = min(int(obs_all.shape[0]), int(self.routing_config.memory_size))

        rng = np.random.default_rng(10000 + int(self.current_task_id))
        indices = rng.choice(obs_all.shape[0], size=n, replace=False)

        return torch.as_tensor(obs_all[indices], dtype=torch.float32)

    def _memory_kl(self, mem_a: torch.Tensor, mem_b: torch.Tensor) -> float:
        a_flat = mem_a.reshape(mem_a.shape[0], -1)
        b_flat = mem_b.reshape(mem_b.shape[0], -1)
        m = min(int(a_flat.shape[0]), int(b_flat.shape[0]))
        a_flat = a_flat[:m]
        b_flat = b_flat[:m]

        a_logp = F.log_softmax(a_flat, dim=-1)
        b_p = F.softmax(b_flat, dim=-1)
        kl = F.kl_div(a_logp, b_p, reduction="batchmean")

        return float(kl.detach().cpu().item())

    def _select_copy_for_new_task(
        self, task_memory: torch.Tensor
    ) -> tuple[int, int | None, float | None, bool]:
        if self.current_task_id == 0 or not self.task_memories:
            return 0, None, None, False

        best_task: int | None = None
        best_kl: float | None = None

        for t, mem in self.task_memories.items():
            kl = self._memory_kl(task_memory, mem)
            if best_kl is None or kl < best_kl:
                best_kl = kl
                best_task = int(t)

        create_new = False
        if best_kl is None:
            create_new = True
        elif best_kl > self.routing_config.kl_threshold:
            create_new = True

        if create_new:
            if (
                self.routing_config.max_copies is not None
                and len(self.copy_states) >= self.routing_config.max_copies
            ):
                fallback_copy_id = (
                    self.task_to_copy[int(best_task)] if best_task is not None else 0
                )
                return int(fallback_copy_id), best_task, best_kl, False

            new_copy_idx = len(self.copy_states)
            return new_copy_idx, best_task, best_kl, True

        assert best_task is not None
        return int(self.task_to_copy[int(best_task)]), best_task, best_kl, False

    def _activate_copy(self, copy_id: int) -> None:
        self._sync_public_state_to_active_copy()

        self.current_copy_id = int(copy_id)
        state = self.copy_states[self.current_copy_id]
        self.model = state.model  # type: ignore[assignment]
        self.optimizer = state.optimizer
        self._refresh_name_sets()

    def _sync_public_state_to_active_copy(self) -> None:
        if not hasattr(self, "copy_states") or not self.copy_states:
            return
        if self.current_copy_id < 0 or self.current_copy_id >= len(self.copy_states):
            return

        state = self.copy_states[self.current_copy_id]
        state.model = self.model
        state.optimizer = self.optimizer

    def _make_fresh_copy(self) -> ModelCopy:
        hp: dict[str, Any] = dict(self.model_hparams)
        hp["max_ep_len"] = int(
            getattr(self.model, "max_ep_len", hp.get("max_ep_len", 10000))
        )
        hp["rtg_scale"] = float(
            getattr(self.model, "rtg_scale", hp.get("rtg_scale", 1000.0))
        )

        model = DecisionTransformer(
            obs_shape=hp["obs_shape"],
            n_actions=int(hp["n_actions"]),
            d_model=int(hp["d_model"]),
            n_layers=int(hp["n_layers"]),
            n_heads=int(hp["n_heads"]),
            seq_len=int(hp["seq_len"]),
            p_drop=float(hp["p_drop"]),
            max_ep_len=int(hp["max_ep_len"]),
            rtg_scale=float(hp["rtg_scale"]),
        ).to(self.device)

        convert_to_sparse(model, self.sparse_conversion_config)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hp["lr"]),
            weight_decay=float(hp["weight_decay"]),
        )

        return ModelCopy(
            model=model,
            optimizer=optimizer,
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )

    def train_task(
        self,
        task_trajs: list[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Train on a single task's trajectory data."""
        task_memory = self._build_task_memory(task_trajs)

        copy_id, src_task, best_kl, created_new = self._select_copy_for_new_task(
            task_memory
        )
        self._activate_copy(copy_id)

        self._prepare_current_task()

        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        last_loss = 0.0
        for iteration in range(int(steps)):
            obs, actions, rtg, ts, mask = next(loader)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_cross_entropy(logits, actions, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            state = self.copy_states[self.current_copy_id]
            zero_gradients_for_frozen_params(self.model, state.consolidated_masks)
            zero_gradients_for_non_maskable_params(
                self.model,
                self.maskable_param_names,
                self.score_param_names,
                freeze_non_maskable_after_first=True,
                current_task_id=self.current_task_id,
            )

            frozen_snapshot = snapshot_frozen_parameters(
                self.model,
                state.consolidated_masks,
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
                logging.getLogger("tsn_affinity").info(
                    "[tsn-replay-kl] task=%s copy=%s src=%s "
                    "kl=%s new_copy=%s iteration=%s "
                    "ce=%.6e keep_ratio=%.4f",
                    self.current_task_id,
                    copy_id,
                    src_task,
                    best_kl,
                    int(created_new),
                    iteration,
                    last_loss,
                    self.current_keep_ratio,
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        self._sync_public_state_to_active_copy()

        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: list[Trajectory]) -> None:
        """Called after training on a task completes."""
        self._sync_public_state_to_active_copy()
        task_id = int(self.current_task_id)
        state = self.copy_states[self.current_copy_id]

        task_masks = self._collect_current_task_masks()
        state.per_task_masks[task_id] = task_masks
        state.task_codebooks[task_id] = self._quantize_new_weights(task_masks)
        state.task_keep_ratios[task_id] = float(self.current_keep_ratio)
        self._update_consolidated_masks_from_state(state, task_masks)

        self.task_to_copy[task_id] = int(self.current_copy_id)

        used = 0
        total = 0
        for key, mask in state.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))
        logging.getLogger("tsn_affinity").info(
            "[tsn-replay-kl] after task %s: copy=%s occupied_ratio=%.4f",
            task_id,
            self.current_copy_id,
            ratio,
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1

    def _update_consolidated_masks_from_state(
        self,
        state: ModelCopy,
        task_masks: dict[str, torch.Tensor | None],
    ) -> None:
        if not state.consolidated_masks:
            state.consolidated_masks = {
                k: (None if v is None else v.clone()) for k, v in task_masks.items()
            }
            return
        for key, mask in task_masks.items():
            if mask is None:
                state.consolidated_masks.setdefault(key, None)
                continue
            existing = state.consolidated_masks.get(key)
            if existing is None:
                state.consolidated_masks[key] = mask.clone()
            else:
                state.consolidated_masks[key] = torch.logical_or(
                    existing.bool(),
                    mask.bool(),
                ).to(torch.uint8)
