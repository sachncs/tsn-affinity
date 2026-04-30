"""TSN-Affinity strategy with action/latent/hybrid routing."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tsn_affinity.core.config import ModelConfig, RoutingConfig, SparseConfig, TSNAffinityConfig
from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.data.batch_generator import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory
from tsn_affinity.routing.affinity_metrics import (
    compute_action_affinity,
    compute_action_affinity_batch,
    compute_latent_affinity_batch,
    compute_diag_stats,
    compute_hybrid_affinity,
    compute_latent_affinity,
    extract_obs_latents,
    normalize_scores,
)
from tsn_affinity.routing.warmstarter import MaskWarmstarter
from tsn_affinity.sparse.module_converter import (
    SparseConversionConfig,
    convert_to_sparse,
    iter_sparse_modules,
    kmeans_quantize,
    rebuild_optimizer,
)
from tsn_affinity.strategies.model_copy import ModelCopy
from tsn_affinity.strategies.tsn_base import TSNBaseStrategy
from tsn_affinity.strategies.training_utils import (
    restore_frozen_parameters,
    snapshot_frozen_parameters,
    verify_frozen_gradient_zeroing,
    zero_gradients_for_frozen_params,
    zero_gradients_for_non_maskable_params,
)


@dataclass
class AffinityRoutingConfig:
    """Configuration for affinity-based routing.

    Attributes:
        mode: Routing mode ("action", "latent", or "hybrid").
        action_threshold: Cross-entropy threshold for action routing.
        latent_threshold: KL divergence threshold for latent routing.
        hybrid_threshold: Score threshold for hybrid routing.
        hybrid_alpha: Weight for action component in hybrid.
        normalize_scores: Whether to min-max normalize scores.
        routing_n_batches: Number of batches for action affinity estimation.
        routing_batch_size: Batch size for affinity estimation.
        relative_threshold: If True, use relative threshold (score < best * factor).
        copy_creation_margin: Multiplier for relative threshold (copy if best > min * margin).
    """

    mode: str = "action"
    action_threshold: float = 12.0
    latent_threshold: float = 25.0
    hybrid_threshold: float = 0.50
    hybrid_alpha: float = 0.70
    normalize_scores: bool = True
    routing_n_batches: int = 4
    routing_batch_size: int = 64
    relative_threshold: bool = True
    copy_creation_margin: float = 2.5


class TSNAffinityStrategy(TSNBaseStrategy):
    """TSN-Affinity: TSN with affinity-based routing.

    Routes new tasks to model copies based on:
    - Action affinity: Cross-entropy between source predictions and new task actions
    - Latent affinity: Symmetric KL divergence of observation latents
    - Hybrid: Weighted combination of both

    Warm-starts mask scores from source task when applicable.

    This is the main strategy described in the paper.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
        sparse_config: SparseConfig | None = None,
        affinity_config: AffinityRoutingConfig | None = None,
    ) -> None:
        if affinity_config is None:
            affinity_config = AffinityRoutingConfig()

        super().__init__(obs_shape, n_actions, seq_len, device, model_config, sparse_config)

        self.affinity_config = affinity_config
        self.warmstarter = MaskWarmstarter(
            warmstart=True,
            strength=2.0,
            noise_std=0.02,
        )

        self.task_memories: Dict[int, torch.Tensor] = {}
        self.task_to_copy: Dict[int, int] = {}
        self.copy_states: List[ModelCopy] = []
        self.task_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.task_similarity: Dict[int, Dict] = {}

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

    def _build_task_memory(self, task_trajs: List[Trajectory]) -> torch.Tensor:
        obs_chunks = []
        total = 0
        for tr in task_trajs:
            obs = np.asarray(tr.obs, dtype=np.float32)
            if obs.ndim < 2:
                continue
            obs_chunks.append(obs)
            total += int(obs.shape[0])
            if total >= 256 * 4:
                break

        if not obs_chunks:
            raise ValueError("Cannot build task memory from empty observations")

        obs_all = np.concatenate(obs_chunks, axis=0)
        n = min(int(obs_all.shape[0]), 256)

        rng = np.random.default_rng(10000 + int(self.current_task_id))
        indices = rng.choice(obs_all.shape[0], size=n, replace=False)

        return torch.as_tensor(obs_all[indices], dtype=torch.float32)

    def _compute_current_memory_latent_stats(self, task_memory_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            obs = task_memory_obs.to(self.device, dtype=torch.float32).unsqueeze(0)
            z = extract_obs_latents(obs, self.model.obs_enc)
            mu, var = compute_diag_stats(z)
            return mu.detach().cpu(), var.detach().cpu()

    def _store_task_latent_stats(self, task_id: int) -> None:
        if int(task_id) not in self.task_memories:
            return

        memory = self.task_memories[int(task_id)]
        self.set_eval_task(int(task_id))
        mu, var = self._compute_current_memory_latent_stats(memory)
        self.task_latent_stats[int(task_id)] = (mu, var)
        self.clear_eval_task()

    def _activate_copy(self, copy_id: int) -> None:
        self._sync_public_state_to_active_copy()

        self.current_copy_id = int(copy_id)
        state = self.copy_states[self.current_copy_id]
        self.model = state.model
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
        hp = dict(self.model_hparams)
        hp["max_ep_len"] = int(getattr(self.model, "max_ep_len", hp.get("max_ep_len", 10000)))
        hp["rtg_scale"] = float(getattr(self.model, "rtg_scale", hp.get("rtg_scale", 1000.0)))

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

    def _select_copy_for_new_task(
        self,
        task_memory_obs: torch.Tensor,
        task_trajs: List[Trajectory],
    ) -> Tuple[int, Optional[int], Dict, bool]:
        if self.current_task_id == 0 or not self.task_to_copy:
            return 0, None, {"best_action": None, "best_latent": None, "best_score": None}, False

        previous_tasks = sorted(self.task_to_copy.keys())
        copy_ids = [self.task_to_copy[t] for t in previous_tasks]

        models = [self.copy_states[cid].model for cid in copy_ids]

        action_scores = compute_action_affinity_batch(
            models=models,
            task_ids=[int(t) for t in previous_tasks],
            task_trajs=task_trajs,
            seq_len=self.seq_len,
            n_batches=self.affinity_config.routing_n_batches,
            batch_size=self.affinity_config.routing_batch_size,
            device=self.device,
        )

        latent_scores: Dict[int, float] = {}
        if self.affinity_config.mode in ("latent", "hybrid"):
            latent_scores = compute_latent_affinity_batch(
                task_memory_obs=task_memory_obs,
                obs_encoder=self.model.obs_enc,
                stored_stats=self.task_latent_stats,
                target_task_ids=[int(t) for t in previous_tasks],
                device=self.device,
            )

        if self.affinity_config.mode == "action":
            final_scores = dict(action_scores)

        elif self.affinity_config.mode == "latent":
            final_scores = dict(latent_scores)

        elif self.affinity_config.mode == "hybrid":
            if len(previous_tasks) == 1:
                final_scores = {previous_tasks[0]: compute_hybrid_affinity(
                    action_scores[previous_tasks[0]],
                    latent_scores.get(previous_tasks[0], float("inf")),
                    self.affinity_config.action_threshold,
                    self.affinity_config.latent_threshold,
                    self.affinity_config.hybrid_alpha,
                )}
            else:
                act_norm = normalize_scores(action_scores) if self.affinity_config.normalize_scores else action_scores
                lat_norm = normalize_scores(latent_scores) if self.affinity_config.normalize_scores else latent_scores
                final_scores = {
                    t: float(
                        self.affinity_config.hybrid_alpha * act_norm[t]
                        + (1.0 - self.affinity_config.hybrid_alpha) * lat_norm[t]
                    )
                    for t in previous_tasks
                }
        else:
            raise ValueError(f"Unsupported mode: {self.affinity_config.mode}")

        best_task = min(final_scores, key=final_scores.get)
        best_score = final_scores[best_task]
        all_scores = list(final_scores.values())
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score
        relative_margin = 0.0
        create_new = False

        if self.affinity_config.relative_threshold and len(all_scores) > 1:
            if score_range > 1e-6:
                relative_margin = (best_score - min_score) / score_range
                create_new = relative_margin < (1.0 / self.affinity_config.copy_creation_margin)
        else:
            if self.affinity_config.mode == "action":
                create_new = best_score > self.affinity_config.action_threshold
            elif self.affinity_config.mode == "latent":
                create_new = best_score > self.affinity_config.latent_threshold
            else:
                create_new = best_score > self.affinity_config.hybrid_threshold

        details = {
            "best_action": float(action_scores.get(best_task, float("nan"))),
            "best_latent": float(latent_scores.get(best_task, float("nan"))),
            "best_score": float(best_score),
            "min_score": float(min_score),
            "max_score": float(max_score),
            "score_range": float(score_range),
            "relative_margin": float(relative_margin),
            "create_new_copy": bool(create_new),
        }

        if create_new:
            new_copy_idx = len(self.copy_states)
            new_copy = self._make_fresh_copy()
            self.copy_states.append(new_copy)
            return new_copy_idx, int(best_task), details, True

        selected_copy_id = self.task_to_copy[int(best_task)]
        return int(selected_copy_id), int(best_task), details, False

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        task_memory = self._build_task_memory(task_trajs)

        if self.current_task_id == 0:
            copy_id = 0
            src_task = None
            score_details = {"best_action": None, "best_latent": None, "best_score": None}
            created_new = False
        else:
            copy_id, src_task, score_details, created_new = self._select_copy_for_new_task(
                task_memory,
                task_trajs,
            )

        self._activate_copy(copy_id)

        self.task_similarity[self.current_task_id] = {
            "source_task": None if src_task is None else int(src_task),
            "copy_id": int(copy_id),
            "best_action": score_details.get("best_action"),
            "best_latent": score_details.get("best_latent"),
            "best_score": score_details.get("best_score"),
            "score_mode": self.affinity_config.mode,
            "created_new_copy": bool(created_new),
        }

        self._prepare_current_task()

        if src_task is not None:
            self.warmstarter.apply(self.model, src_task, self.task_to_copy, self.copy_states)

        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        self.model.train()
        last_loss = None
        for iteration in range(int(steps)):
            obs, actions, rtg, ts, mask = next(loader)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_cross_entropy(logits, actions, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            state = self.copy_states[self.current_copy_id]
            name_to_module = dict(self.model.named_modules())
            zero_gradients_for_frozen_params(self.model, state.consolidated_masks)
            is_valid, msg = verify_frozen_gradient_zeroing(
                self.model, state.consolidated_masks, name_to_module
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
                meta = self.task_similarity[self.current_task_id]
                print(
                    f"[tsn-affinity] task={self.current_task_id} copy={meta['copy_id']} "
                    f"src={meta['source_task']} mode={meta['score_mode']} "
                    f"best_action={meta['best_action']} best_latent={meta['best_latent']} "
                    f"best_score={meta['best_score']} new_copy={int(meta['created_new_copy'])} "
                    f"iteration={iteration} ce={last_loss:.6e} keep_ratio={self.current_keep_ratio:.4f}"
                )

        self.task_memories[self.current_task_id] = task_memory.detach().cpu()
        self._sync_public_state_to_active_copy()

        return {"loss": last_loss, "keep_ratio": float(self.current_keep_ratio)}

    def after_task(self, task_trajs: List[Trajectory]) -> None:
        self._sync_public_state_to_active_copy()

        task_id = int(self.current_task_id)
        state = self.copy_states[self.current_copy_id]

        task_masks = self._collect_current_task_masks()
        state.per_task_masks[task_id] = task_masks
        state.task_codebooks[task_id] = self._quantize_new_weights(task_masks)
        self._update_consolidated_masks_from_state(state, task_masks)

        self.task_to_copy[task_id] = int(self.current_copy_id)

        if self.affinity_config.mode in ("latent", "hybrid"):
            self._store_task_latent_stats(task_id)

        used = 0
        total = 0
        for key, mask in state.consolidated_masks.items():
            if mask is None or not key.endswith(".weight"):
                continue
            used += int(mask.sum().item())
            total += int(mask.numel())
        ratio = float(used / max(1, total))

        meta = self.task_similarity.get(task_id, {})
        print(
            f"[tsn-affinity] after task {task_id}: copy={self.current_copy_id} "
            f"occupied_ratio={ratio:.4f} source_task={meta.get('source_task')} "
            f"best_action={meta.get('best_action')} best_latent={meta.get('best_latent')} "
            f"best_score={meta.get('best_score')} created_new_copy={meta.get('created_new_copy')}"
        )

        self.set_eval_task(task_id)
        self.current_task_id += 1

    def _update_consolidated_masks_from_state(
        self,
        state: ModelCopy,
        task_masks: Dict[str, Optional[torch.Tensor]],
    ) -> None:
        if not state.consolidated_masks:
            state.consolidated_masks = {k: (None if v is None else v.clone()) for k, v in task_masks.items()}
            return
        for key, mask in task_masks.items():
            if mask is None:
                state.consolidated_masks.setdefault(key, None)
                continue
            if state.consolidated_masks.get(key) is None:
                state.consolidated_masks[key] = mask.clone()
            else:
                state.consolidated_masks[key] = torch.logical_or(
                    state.consolidated_masks[key].bool(),
                    mask.bool(),
                ).to(torch.uint8)