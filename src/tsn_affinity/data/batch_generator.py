"""Minibatch generation from trajectory lists."""

from __future__ import annotations

import numpy as np
import torch
from typing import Generator, List, Tuple

from tsn_affinity.data.trajectory import Trajectory


def make_minibatches(
    trajs: List[Trajectory],
    seq_len: int,
    batch_size: int,
    device: str,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """Generate minibatches from trajectory list indefinitely.

    Args:
        trajs: List of Trajectory objects.
        seq_len: Maximum sequence length (K).
        batch_size: Number of sequences per batch.
        device: Device to place tensors on.

    Yields:
        Tuple of (obs, actions, returns_to_go, timesteps, mask).
        obs: [batch_size, seq_len, ...] observation tensor.
        actions: [batch_size, seq_len] action tensor (padded values replaced with 0).
        returns_to_go: [batch_size, seq_len, 1] RTG tensor.
        timesteps: [batch_size, seq_len] timestep tensor.
        mask: [batch_size, seq_len] attention mask.
    """
    while True:
        batch = []
        for _ in range(batch_size):
            traj = trajs[np.random.randint(len(trajs))]
            T = len(traj.actions)

            if T <= seq_len:
                start = 0
            else:
                start = np.random.randint(0, T - seq_len + 1)

            end = min(start + seq_len, T)

            obs_seq = traj.obs[start:end]
            act_seq = traj.actions[start:end]
            rtg_seq = traj.returns_to_go[start:end]
            ts_seq = traj.timesteps[start:end]

            pad = seq_len - len(act_seq)

            # Pad observations
            if obs_seq.ndim == 2:
                obs_seq = np.pad(obs_seq, ((0, pad), (0, 0)))
            else:
                obs_seq = np.pad(obs_seq, ((0, pad), (0, 0), (0, 0), (0, 0)))

            # Pad rest with -1 (ignore_index for cross-entropy)
            act_seq = np.pad(act_seq, (0, pad), constant_values=-1)
            rtg_seq = np.pad(rtg_seq, (0, pad))
            ts_seq = np.pad(ts_seq, (0, pad))

            batch.append((obs_seq, act_seq, rtg_seq[:, None], ts_seq))

        obs = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long, device=device)
        rtg = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32, device=device)
        ts = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.long, device=device)
        mask = (actions != -1).float()
        actions_safe = actions.clone()
        actions_safe[actions == -1] = 0

        yield obs, actions_safe, rtg, ts, mask


def unpack_batch_discrete(
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a batch for discrete action training.

    Args:
        batch: Tuple from make_minibatches.

    Returns:
        Tuple of (obs, actions, rtg, ts, mask) with mask for valid positions.
    """
    obs, actions, rtg, ts = batch

    mask = (actions != -1).float()
    # Replace -1 with 0 for actual forward (cross_entropy ignores it)
    actions_safe = actions.clone()
    actions_safe[actions == -1] = 0

    return obs, actions_safe, rtg, ts, mask


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked cross-entropy loss.

    Args:
        logits: [B, L, D] action logits.
        targets: [B, L] action targets.
        mask: [B, L] binary mask (1 = valid, 0 = ignore).

    Returns:
        Scalar loss averaged over valid positions.
    """
    B, L, D = logits.shape
    logits_flat = logits.reshape(B * L, D)
    targets_flat = targets.reshape(B * L)
    mask_flat = mask.reshape(B * L)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")
    loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
    return loss


def masked_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked MSE loss for continuous actions.

    Args:
        predictions: [B, L, action_dim] predicted actions.
        targets: [B, L, action_dim] target actions.
        mask: [B, L] binary mask (1 = valid, 0 = ignore).

    Returns:
        Scalar loss averaged over valid positions.
    """
    diff = predictions - targets
    diff = (diff ** 2).sum(dim=-1)
    mask = mask.float()
    loss = (diff * mask).sum() / mask.sum().clamp(min=1.0)
    return loss