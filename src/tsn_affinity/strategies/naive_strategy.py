"""Naive baseline strategy - single model, no CL mechanism."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from tsn_affinity.core.config import ModelConfig
from tsn_affinity.core.decision_transformer import DecisionTransformer
from tsn_affinity.data.batch_generator import make_minibatches, masked_cross_entropy
from tsn_affinity.data.trajectory import Trajectory


class NaiveStrategy:
    """Naive baseline: single Decision Transformer without any continual learning.

    This strategy trains a single model on each task sequentially with no
    protection against forgetting. It serves as the baseline to demonstrate
    catastrophic forgetting in the CL setting.

    This corresponds to the "cumulative" baseline in the paper (single model,
    sequential training, no replay).
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        n_actions: int,
        seq_len: int = 20,
        device: str = "cuda",
        model_config: ModelConfig | None = None,
    ) -> None:
        if model_config is None:
            model_config = ModelConfig(obs_shape=obs_shape, n_actions=n_actions, seq_len=seq_len)

        self.device = device
        self.seq_len = seq_len
        self.grad_clip = 1.0
        self.current_task_id = 0

        self.model = DecisionTransformer(
            obs_shape=model_config.obs_shape,
            n_actions=model_config.n_actions,
            d_model=model_config.d_model,
            n_layers=model_config.n_layers,
            n_heads=model_config.n_heads,
            seq_len=model_config.seq_len,
            p_drop=model_config.p_drop,
            max_ep_len=model_config.max_ep_len,
            rtg_scale=model_config.rtg_scale,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=model_config.lr,
            weight_decay=model_config.weight_decay,
        )

    def train_task(
        self,
        task_trajs: List[Trajectory],
        steps: int = 2000,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        self.model.train()
        loader = make_minibatches(task_trajs, self.seq_len, batch_size, self.device)

        total_loss = 0.0
        for iteration in range(int(steps)):
            obs, actions, rtg, ts, mask = next(loader)

            logits = self.model(obs, actions, rtg, ts, attention_mask=mask)
            loss = masked_cross_entropy(logits, actions, mask)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().item())

            if iteration % 500 == 0 or iteration == int(steps) - 1:
                avg_loss = total_loss / (iteration + 1)
                print(
                    f"[naive] task={self.current_task_id} iteration={iteration} "
                    f"loss={avg_loss:.6f}"
                )

        return {"loss": total_loss / max(1, int(steps)), "keep_ratio": 1.0}

    def after_task(self, task_trajs: List[Trajectory]) -> None:
        self.current_task_id += 1

    def set_eval_task(self, task_id: int) -> None:
        pass

    def clear_eval_task(self) -> None:
        pass

    def has_task_mask(self, task_id: int) -> bool:
        return False