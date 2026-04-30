"""Affinity-based router for model copy selection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from tsn_affinity.routing.affinity_metrics import (
    compute_action_affinity,
    compute_hybrid_affinity,
    compute_latent_affinity,
    normalize_scores,
)


@dataclass
class RoutingResult:
    """Result of a routing decision.

    Attributes:
        copy_id: ID of selected model copy.
        source_task_id: ID of source task (None if task 0).
        action_score: Action affinity score.
        latent_score: Latent affinity score.
        best_score: Final routing score.
        created_new_copy: Whether a new copy was created.
    """

    copy_id: int
    source_task_id: int | None
    action_score: float | None
    latent_score: float | None
    best_score: float | None
    created_new_copy: bool


class AffinityRouter:
    """Routes new tasks to model copies based on affinity metrics.

    Supports three routing modes:
    - "action": Routes based on cross-entropy between source predictions
      and new task demonstrations.
    - "latent": Routes based on symmetric KL divergence of observation latents.
    - "hybrid": Combines action and latent scores.

    Attributes:
        mode: Routing mode ("action", "latent", or "hybrid").
        action_threshold: Cross-entropy threshold for action-based routing.
        latent_threshold: KL threshold for latent-based routing.
        hybrid_threshold: Score threshold for hybrid routing.
        hybrid_alpha: Weight for action component in hybrid scoring.
        normalize_scores: Whether to min-max normalize scores.
        max_copies: Maximum number of model copies (None = unlimited).
    """

    def __init__(
        self,
        mode: str = "action",
        action_threshold: float = 12.0,
        latent_threshold: float = 25.0,
        hybrid_threshold: float = 0.50,
        hybrid_alpha: float = 0.70,
        normalize_scores: bool = True,
        max_copies: int | None = None,
    ) -> None:
        self.mode = str(mode)
        self.action_threshold = float(action_threshold)
        self.latent_threshold = float(latent_threshold)
        self.hybrid_threshold = float(hybrid_threshold)
        self.hybrid_alpha = float(hybrid_alpha)
        self.normalize_scores = bool(normalize_scores)
        self.max_copies = max_copies

    def select_copy(
        self,
        task_memory_obs: torch.Tensor,
        task_trajs: List,
        model: torch.nn.Module,
        obs_encoder: torch.nn.Module,
        current_task_id: int,
        task_to_copy: Dict[int, int],
        copy_states: List,
        stored_latent_stats: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        device: str,
        seq_len: int,
        routing_n_batches: int,
        routing_batch_size: int,
    ) -> RoutingResult:
        """Select a model copy for a new task.

        Args:
            task_memory_obs: Observation memory from new task.
            task_trajs: Trajectories from new task.
            model: Current active model.
            obs_encoder: Observation encoder.
            current_task_id: ID of the new task being trained.
            task_to_copy: Dict mapping task_id -> copy_id.
            copy_states: List of ModelCopy states.
            stored_latent_stats: Dict of task_id -> (mean, variance) for latents.
            device: Device for computation.
            seq_len: Sequence length for routing computation.
            routing_n_batches: Number of batches for action affinity estimation.
            routing_batch_size: Batch size for routing.

        Returns:
            RoutingResult with copy selection details.
        """
        if current_task_id == 0 or not task_to_copy:
            return RoutingResult(
                copy_id=0,
                source_task_id=None,
                action_score=None,
                latent_score=None,
                best_score=None,
                created_new_copy=False,
            )

        action_scores: Dict[int, float] = {}
        latent_scores: Dict[int, float] = {}
        previous_tasks = sorted(task_to_copy.keys())

        for t in previous_tasks:
            copy_id = task_to_copy[t]

            action_scores[int(t)] = compute_action_affinity(
                model=copy_states[copy_id].model,
                source_task_id=int(t),
                task_trajs=task_trajs,
                obs_encoder=obs_encoder,
                seq_len=seq_len,
                n_batches=routing_n_batches,
                batch_size=routing_batch_size,
                device=device,
            )

            if self.mode in ("latent", "hybrid"):
                latent_scores[int(t)] = compute_latent_affinity(
                    model=copy_states[copy_id].model,
                    source_task_id=int(t),
                    task_memory_obs=task_memory_obs,
                    obs_encoder=obs_encoder,
                    stored_stats=stored_latent_stats,
                    device=device,
                )

        if self.mode == "action":
            final_scores = dict(action_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.action_threshold

        elif self.mode == "latent":
            final_scores = dict(latent_scores)
            best_task = min(final_scores, key=final_scores.get)
            best_score = final_scores[best_task]
            create_new = best_score > self.latent_threshold

        elif self.mode == "hybrid":
            if len(previous_tasks) == 1:
                best_task = int(previous_tasks[0])
                best_score = compute_hybrid_affinity(
                    action_scores[best_task],
                    latent_scores.get(best_task, float("inf")),
                    self.action_threshold,
                    self.latent_threshold,
                    self.hybrid_alpha,
                )
                create_new = best_score > 1.0
            else:
                act_norm = normalize_scores(action_scores) if self.normalize_scores else action_scores
                lat_norm = normalize_scores(latent_scores) if self.normalize_scores else latent_scores
                final_scores = {
                    t: float(self.hybrid_alpha * act_norm[t] + (1.0 - self.hybrid_alpha) * lat_norm[t])
                    for t in previous_tasks
                }
                best_task = min(final_scores, key=final_scores.get)
                best_score = final_scores[best_task]
                create_new = best_score > self.hybrid_threshold

        else:
            raise ValueError(f"Unsupported routing mode: {self.mode}")

        if create_new:
            if self.max_copies is not None and len(copy_states) >= self.max_copies:
                best_copy_id = task_to_copy[int(best_task)]
                return RoutingResult(
                    copy_id=int(best_copy_id),
                    source_task_id=int(best_task),
                    action_score=float(action_scores.get(best_task)),
                    latent_score=float(latent_scores.get(best_task)),
                    best_score=float(best_score),
                    created_new_copy=False,
                )

            new_copy_idx = len(copy_states)
            return RoutingResult(
                copy_id=new_copy_idx,
                source_task_id=int(best_task),
                action_score=float(action_scores.get(best_task)),
                latent_score=float(latent_scores.get(best_task)),
                best_score=float(best_score),
                created_new_copy=True,
            )

        selected_copy_id = task_to_copy[int(best_task)]
        return RoutingResult(
            copy_id=int(selected_copy_id),
            source_task_id=int(best_task),
            action_score=float(action_scores.get(best_task)),
            latent_score=float(latent_scores.get(best_task)),
            best_score=float(best_score),
            created_new_copy=False,
        )