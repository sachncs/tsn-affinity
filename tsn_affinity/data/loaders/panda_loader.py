"""Panda-specific data loading and batching for continuous actions."""

import pickle

import numpy as np
import torch

from tsn_affinity.data.schemas.trajectory import Trajectory, discount_cumsum


def load_panda_offline_pkl(path: str) -> list[Trajectory]:
    """Load Panda offline trajectories from a pickle file.

    Args:
        path: Path to .pkl file containing trajectory data.

    Returns:
        List of Trajectory objects.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    trajs = []
    for traj_data in data:
        obs = np.array(traj_data["obs"], dtype=np.float32)
        actions = np.array(traj_data["actions"], dtype=np.float32)
        rewards = np.array(traj_data["rewards"], dtype=np.float32)
        timesteps = np.arange(len(actions), dtype=np.float32)
        returns_to_go = discount_cumsum(rewards, gamma=1.0)

        trajs.append(Trajectory(obs, actions, rewards, timesteps, returns_to_go))

    return trajs


def make_minibatches_panda(
    trajs: list[Trajectory],
    seq_len: int,
    batch_size: int,
    device: str,
):
    """Generate minibatches from Panda trajectory list indefinitely.

    Args:
        trajs: List of Trajectory objects.
        seq_len: Maximum sequence length.
        batch_size: Number of sequences per batch.
        device: Device to place tensors on.

    Yields:
        Tuple of (obs, actions, returns_to_go, timesteps, mask).
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

            obs_seq = np.pad(obs_seq, ((0, pad), (0, 0)))
            act_seq = np.pad(act_seq, ((0, pad), (0, 0)))
            rtg_seq = np.pad(rtg_seq, (0, pad))
            ts_seq = np.pad(ts_seq, (0, pad))

            batch.append((obs_seq, act_seq, rtg_seq[:, None], ts_seq))

        obs = torch.tensor(
            np.stack([b[0] for b in batch]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            np.stack([b[1] for b in batch]), dtype=torch.float32, device=device
        )
        rtg = torch.tensor(
            np.stack([b[2] for b in batch]), dtype=torch.float32, device=device
        )
        ts = torch.tensor(
            np.stack([b[3] for b in batch]), dtype=torch.long, device=device
        )

        mask = (actions.abs().sum(dim=-1) > 0).float()
        yield obs, actions, rtg, ts, mask


def unpack_batch_continuous(
    batch: tuple,
) -> tuple:
    """Unpack a batch for continuous action training.

    Args:
        batch: Tuple from make_minibatches_panda.

    Returns:
        Tuple of (obs, actions, rtg, ts, mask) with mask for valid positions.
    """
    if len(batch) == 5:
        return batch
    obs, actions, rtg, ts = batch

    B, L = actions.shape[:2]
    valid_mask = torch.ones(B, L, device=actions.device, dtype=torch.float32)

    return obs, actions, rtg, ts, valid_mask
