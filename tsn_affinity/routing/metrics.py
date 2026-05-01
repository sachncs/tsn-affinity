"""Affinity metric computations for routing decisions."""

import numpy as np
import torch
import torch.nn.functional as F

from tsn_affinity.data.trajectory import Trajectory


def compute_action_affinity_batch(
    models: list[torch.nn.Module],
    task_ids: list[int],
    task_trajs: list[Trajectory],
    seq_len: int,
    n_batches: int,
    batch_size: int,
    device: str,
) -> dict[int, float]:
    """Compute action affinity for multiple model copies in parallel.

    Single forward pass computes affinity for all copies, then we aggregate.

    Args:
        models: List of model copies (indexed by copy_id).
        task_ids: List of task IDs corresponding to each model.
        task_trajs: Trajectories from the new task.
        seq_len: Sequence length for minibatch generation.
        n_batches: Number of batches to average over.
        batch_size: Batch size for affinity estimation.
        device: Device to run computation on.

    Returns:
        Dict mapping task_id -> mean cross-entropy (lower = better).
    """
    n_trajs = len(task_trajs)
    if n_trajs == 0:
        return {tid: float("inf") for tid in task_ids}

    loader = _MultiBatchLoader(task_trajs, seq_len, batch_size, device, n_batches)

    task_ces: dict[int, list[float]] = {tid: [] for tid in task_ids}

    model_idx_map = {tid: i for i, tid in enumerate(task_ids)}

    with torch.no_grad():
        for batch_data in loader:
            obs, actions, rtg, ts, mask = batch_data

            stacked_logits = torch.stack(
                [
                    models[model_idx_map[tid]](
                        obs, actions, rtg, ts, attention_mask=mask
                    )
                    for tid in task_ids
                ]
            )

            for tid in task_ids:
                logits = stacked_logits[model_idx_map[tid]]
                if actions.dim() == 3:
                    mse = F.mse_loss(logits, actions, reduction="none").mean(dim=-1)
                    mse = (mse * mask).sum() / mask.sum().clamp(min=1.0)
                    task_ces[tid].append(float(mse.detach().cpu().item()))
                else:
                    ce = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        actions.reshape(-1),
                        ignore_index=-1,
                    )
                    task_ces[tid].append(float(ce.detach().cpu().item()))

    return {
        tid: float(np.mean(ces)) if ces else float("inf")
        for tid, ces in task_ces.items()
    }


class _MultiBatchLoader:
    """Pre-batches trajectories for efficient multi-model affinity computation.

    Creates n_batches different batches upfront to avoid reusing the same
    samples when measuring affinity across multiple model copies.

    Attributes:
        batches: List of pre-generated batches.
    """

    def __init__(
        self,
        trajs: list[Trajectory],
        seq_len: int,
        batch_size: int,
        device: str,
        n_batches: int,
    ) -> None:
        self.batches: list[tuple[torch.Tensor, ...]] = []
        self._generate_batches(trajs, seq_len, batch_size, device, n_batches)

    def _generate_batches(
        self,
        trajs: list[Trajectory],
        seq_len: int,
        batch_size: int,
        device: str,
        n_batches: int,
    ) -> None:
        n_trajs = len(trajs)
        continuous = trajs[0].actions.ndim == 2 if n_trajs > 0 else False
        if n_trajs == 0:
            if continuous:
                self.batches = [
                    (
                        torch.zeros(1, seq_len, device=device, dtype=torch.float32),
                        torch.zeros(1, seq_len, device=device, dtype=torch.float32),
                        torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32),
                        torch.zeros(1, seq_len, device=device, dtype=torch.long),
                        torch.ones(1, seq_len, device=device, dtype=torch.float32),
                    )
                ]
            else:
                self.batches = [
                    (
                        torch.zeros(1, seq_len, device=device, dtype=torch.float32),
                        torch.zeros(1, seq_len, device=device, dtype=torch.long),
                        torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32),
                        torch.zeros(1, seq_len, device=device, dtype=torch.long),
                        torch.ones(1, seq_len, device=device, dtype=torch.float32),
                    )
                ]
            return

        for _ in range(n_batches):
            batch_obs = []
            batch_act = []
            batch_rtg = []
            batch_ts = []

            for _ in range(batch_size):
                traj_idx = np.random.randint(0, n_trajs)
                traj = trajs[traj_idx]
                T = len(traj.actions)

                start = 0 if T <= seq_len else np.random.randint(0, T - seq_len + 1)
                end = min(start + seq_len, T)

                o = traj.obs[start:end]
                a = traj.actions[start:end]
                r = traj.returns_to_go[start:end]
                t = traj.timesteps[start:end]

                pad = seq_len - len(a)
                if o.ndim == 2:
                    o = np.pad(o, ((0, pad), (0, 0)))
                else:
                    o = np.pad(o, ((0, pad), (0, 0), (0, 0), (0, 0)))
                if continuous:
                    a = np.pad(a, ((0, pad), (0, 0)))
                else:
                    a = np.pad(a, (0, pad), constant_values=-1)
                r = np.pad(r, (0, pad))
                t = np.pad(t, (0, pad))

                batch_obs.append(o)
                batch_act.append(a)
                batch_rtg.append(r[:, None])
                batch_ts.append(t)

            obs_t = torch.tensor(
                np.stack(batch_obs), dtype=torch.float32, device=device
            )
            rtg_t = torch.tensor(
                np.stack(batch_rtg), dtype=torch.float32, device=device
            )
            ts_t = torch.tensor(np.stack(batch_ts), dtype=torch.long, device=device)
            if continuous:
                act_t = torch.tensor(
                    np.stack(batch_act), dtype=torch.float32, device=device
                )
                mask_t = (act_t.abs().sum(dim=-1) > 0).float()
                self.batches.append((obs_t, act_t, rtg_t, ts_t, mask_t))
            else:
                act_t = torch.tensor(
                    np.stack(batch_act), dtype=torch.long, device=device
                )
                mask_t = (act_t != -1).float()
                act_safe = act_t.clone()
                act_safe[act_t == -1] = 0
                self.batches.append((obs_t, act_safe, rtg_t, ts_t, mask_t))

    def __iter__(self):
        return iter(self.batches)


def compute_latent_affinity_batch(
    task_memory_obs: torch.Tensor,
    obs_encoder: torch.nn.Module,
    stored_stats: dict[int, tuple[torch.Tensor, torch.Tensor]],
    target_task_ids: list[int],
    device: str,
) -> dict[int, float]:
    """Compute latent affinity for multiple tasks in a single forward pass.

    Args:
        task_memory_obs: Observation memory [N, D].
        obs_encoder: Observation encoder.
        stored_stats: Dict of task_id -> (mean, variance).
        target_task_ids: List of task IDs to compute affinity against.
        device: Device for computation.

    Returns:
        Dict mapping task_id -> symmetric KL (lower = better).
    """
    if task_memory_obs.shape[0] == 0:
        return {tid: float("inf") for tid in target_task_ids}

    with torch.no_grad():
        obs_batch = task_memory_obs.to(device=device, dtype=torch.float32).unsqueeze(0)
        z = extract_obs_latents(obs_batch, obs_encoder)
        mu_new, var_new = compute_diag_stats(z)

    results = {}
    for tid in target_task_ids:
        if tid not in stored_stats:
            results[tid] = float("inf")
            continue

        mu_ref, var_ref = stored_stats[tid]
        mu_ref = mu_ref.to(device=device, non_blocking=True)
        var_ref = var_ref.to(device=device, non_blocking=True)

        with torch.no_grad():
            log_ratio = torch.log(var_ref / var_new + 1e-8)
            kl_ab = 0.5 * torch.sum(
                log_ratio + (var_new + (mu_new - mu_ref) ** 2) / var_ref - 1
            )
            log_ratio_b = torch.log(var_new / var_ref + 1e-8)
            kl_ba = 0.5 * torch.sum(
                log_ratio_b + (var_ref + (mu_ref - mu_new) ** 2) / var_new - 1
            )
            symmetric_kl = 0.5 * (kl_ab + kl_ba)

        results[tid] = float(symmetric_kl.detach().cpu().item())

    return results


def compute_action_affinity(
    model: torch.nn.Module,
    task_trajs: list[Trajectory],
    seq_len: int,
    n_batches: int,
    batch_size: int,
    device: str,
) -> float:
    """Estimate action compatibility between source task model and new task data.

    Computes average cross-entropy between model predictions on new task
    demonstrations and the demonstrated actions themselves.

    Args:
        model: The Decision Transformer model.
        task_trajs: Trajectories from the new task.
        seq_len: Sequence length for minibatch generation.
        n_batches: Number of batches to average over.
        batch_size: Batch size for affinity estimation.
        device: Device to run computation on.

    Returns:
        Mean cross-entropy (lower = better compatibility).
    """
    model.eval()
    n_trajs = len(task_trajs)
    if n_trajs == 0:
        return float("inf")

    loader = _AffinityBatchLoader(task_trajs, seq_len, batch_size, device)

    ce_values: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, n_batches)):
            try:
                obs, actions, rtg, ts, mask = loader.next_batch()
            except StopIteration:
                break

            logits = model(obs, actions, rtg, ts, attention_mask=mask)
            if actions.dim() == 3:
                mse = F.mse_loss(logits, actions, reduction="none").mean(dim=-1)
                mse = (mse * mask).sum() / mask.sum().clamp(min=1.0)
                ce_values.append(float(mse.detach().cpu().item()))
            else:
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    actions.reshape(-1),
                    ignore_index=-1,
                )
                ce_values.append(float(ce.detach().cpu().item()))

    return float(np.mean(ce_values)) if ce_values else float("inf")


def compute_latent_affinity(
    source_task_id: int,
    task_memory_obs: torch.Tensor,
    obs_encoder: torch.nn.Module,
    stored_stats: dict[int, tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    """Estimate latent similarity between source task and new task.

    Computes symmetric KL divergence between diagonal Gaussian distributions
    fitted to observation latents from source and new tasks.

    Args:
        source_task_id: ID of source task to activate masks for.
        task_memory_obs: Observation memory from new task [N, D].
        obs_encoder: Observation encoder for latent extraction.
        stored_stats: Dict of task_id -> (mean, variance) for observation latents.
        device: Device to run computation on.

    Returns:
        Symmetric KL divergence (lower = better similarity).
    """
    if source_task_id not in stored_stats:
        return float("inf")

    mu_ref, var_ref = stored_stats[source_task_id]
    mu_ref = mu_ref.to(device=device, non_blocking=True)
    var_ref = var_ref.to(device=device, non_blocking=True)

    with torch.no_grad():
        obs_batch = task_memory_obs.to(device=device, dtype=torch.float32).unsqueeze(0)
        z = extract_obs_latents(obs_batch, obs_encoder)
        mu_new, var_new = compute_diag_stats(z)

        log_ratio = torch.log(var_ref / var_new + 1e-8)
        kl_ab = 0.5 * torch.sum(
            log_ratio + (var_new + (mu_new - mu_ref) ** 2) / var_ref - 1
        )
        log_ratio_b = torch.log(var_new / var_ref + 1e-8)
        kl_ba = 0.5 * torch.sum(
            log_ratio_b + (var_ref + (mu_ref - mu_new) ** 2) / var_new - 1
        )
        symmetric_kl = 0.5 * (kl_ab + kl_ba)

    return float(symmetric_kl.detach().cpu().item())


def compute_hybrid_affinity(
    action_score: float,
    latent_score: float,
    action_threshold: float,
    latent_threshold: float,
    alpha: float,
) -> float:
    """Compute hybrid affinity score combining action and latent metrics.

    Uses absolute ratio comparison when only one previous task exists
    (to avoid min-max normalization collapsing to 0.0 for single candidate).

    Args:
        action_score: Action affinity cross-entropy.
        latent_score: Latent affinity KL divergence.
        action_threshold: Threshold for action affinity.
        latent_threshold: Threshold for latent affinity.
        alpha: Weight for action component (1-alpha for latent).

    Returns:
        Combined score (lower = better affinity).
    """
    action_ratio = float(action_score / max(action_threshold, 1e-12))
    latent_ratio = float(latent_score / max(latent_threshold, 1e-12))
    return float(alpha * action_ratio + (1.0 - alpha) * latent_ratio)


def normalize_scores(
    scores: dict[int, float],
) -> dict[int, float]:
    """Min-max normalize similarity scores.

    Args:
        scores: Dict of task_id -> score.

    Returns:
        Dict of task_id -> normalized score in [0, 1].
    """
    if not scores:
        return {}
    arr = np.array(list(scores.values()), dtype=np.float32)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.0 for k in scores}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in scores.items()}


def extract_obs_latents(
    obs: torch.Tensor,
    obs_encoder: torch.nn.Module,
) -> torch.Tensor:
    """Extract observation latents from observation encoder.

    Args:
        obs: Observation tensor [B, L, ...].
        obs_encoder: Observation encoder module.

    Returns:
        Latent tensor [B, L, d_model].
    """
    x = obs.to(dtype=torch.float32)
    if obs.dtype == torch.uint8:
        x = x / 255.0

    if x.dim() == 5:
        B, L, C, H, W = x.shape
        z = obs_encoder(x.reshape(B * L, C, H, W)).reshape(B, L, -1)
    elif x.dim() == 3:
        B, L, D = x.shape
        z = obs_encoder(x.reshape(B * L, D)).reshape(B, L, -1)
    else:
        raise ValueError(f"Unexpected obs shape: {tuple(x.shape)}")
    return z  # type: ignore[no-any-return]


def compute_diag_stats(
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute diagonal Gaussian statistics (mean, variance) from latents.

    Args:
        z: Latent tensor [B, L, D] or flattened [N, D].
        mask: Optional binary mask for valid positions.

    Returns:
        Tuple of (mean, variance) tensors.
    """
    if z.dim() == 3:
        if mask is None:
            x = z.reshape(-1, z.shape[-1])
        else:
            keep = mask.reshape(-1).bool()
            x = z.reshape(-1, z.shape[-1])[keep]
    else:
        x = z

    if x.numel() == 0:
        H = z.shape[-1]
        return torch.zeros(H, device=z.device), torch.ones(H, device=z.device)

    mu = x.mean(dim=0)
    var = x.var(dim=0, unbiased=False).clamp(min=1e-6)
    return mu, var


class _AffinityBatchLoader:
    """Efficient batch loader for affinity computation.

    Pre-builds batches in shuffled order to avoid per-sample random sampling
    overhead. Uses contiguous tensors for better memory access patterns.

    Attributes:
        obs: Stacked observation tensor [n_batches * batch_size, seq_len, ...].
        actions: Stacked action tensor [n_batches * batch_size, seq_len].
        rtg: Stacked RTG tensor [n_batches * batch_size, seq_len, 1].
        ts: Stacked timestep tensor [n_batches * batch_size, seq_len].
        mask: Attention mask tensor [n_batches * batch_size, seq_len].
        n_batches: Total number of batches.
        batch_size: Batch size.
        device: Device for tensors.
    """

    def __init__(
        self,
        trajs: list[Trajectory],
        seq_len: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self._build_batches(trajs, seq_len, device)

    def _build_batches(
        self,
        trajs: list[Trajectory],
        seq_len: int,
        device: str,
    ) -> None:
        n_trajs = len(trajs)
        continuous = trajs[0].actions.ndim == 2 if n_trajs > 0 else False
        if n_trajs == 0:
            self.obs = torch.zeros(1, seq_len, device=device, dtype=torch.float32)
            if continuous:
                self.actions = torch.zeros(
                    1, seq_len, device=device, dtype=torch.float32
                )
            else:
                self.actions = torch.zeros(1, seq_len, device=device, dtype=torch.long)
            self.rtg = torch.zeros(1, seq_len, 1, device=device, dtype=torch.float32)
            self.ts = torch.zeros(1, seq_len, device=device, dtype=torch.long)
            self.mask = torch.ones(1, seq_len, device=device, dtype=torch.float32)
            self.actions_safe = self.actions.clone()
            self.n_batches = 1
            return

        batch_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for _ in range(self.batch_size):
            traj_idx = np.random.randint(0, n_trajs)
            traj = trajs[traj_idx]
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

            if obs_seq.ndim == 2:
                obs_seq = np.pad(obs_seq, ((0, pad), (0, 0)))
            else:
                obs_seq = np.pad(obs_seq, ((0, pad), (0, 0), (0, 0), (0, 0)))
            if continuous:
                act_seq = np.pad(act_seq, ((0, pad), (0, 0)))
            else:
                act_seq = np.pad(act_seq, (0, pad), constant_values=-1)
            rtg_seq = np.pad(rtg_seq, (0, pad))
            ts_seq = np.pad(ts_seq, (0, pad))

            batch_list.append((obs_seq, act_seq, rtg_seq[:, None], ts_seq))

        self.obs = torch.tensor(
            np.stack([b[0] for b in batch_list]),
            dtype=torch.float32,
            device=device,
        )
        self.rtg = torch.tensor(
            np.stack([b[2] for b in batch_list]),
            dtype=torch.float32,
            device=device,
        )
        self.ts = torch.tensor(
            np.stack([b[3] for b in batch_list]),
            dtype=torch.long,
            device=device,
        )
        if continuous:
            self.actions = torch.tensor(
                np.stack([b[1] for b in batch_list]),
                dtype=torch.float32,
                device=device,
            )
            self.mask = (self.actions.abs().sum(dim=-1) > 0).float()
            self.actions_safe = self.actions.clone()
        else:
            self.actions = torch.tensor(
                np.stack([b[1] for b in batch_list]),
                dtype=torch.long,
                device=device,
            )
            self.mask = (self.actions != -1).float()
            self.actions_safe = self.actions.clone()
            self.actions_safe[self.actions == -1] = 0
        self.n_batches = 1

    def next_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.obs, self.actions_safe, self.rtg, self.ts, self.mask


def make_temp_minibatches(
    trajs: list[Trajectory],
    seq_len: int,
    batch_size: int,
    device: str,
):
    """Generate minibatches (temporary utility).

    Args:
        trajs: List of Trajectory objects.
        seq_len: Maximum sequence length.
        batch_size: Batch size.
        device: Device to place tensors.

    Yields:
        Tuples of (obs, actions, rtg, ts).
    """
    while True:
        batch = []
        for _ in range(batch_size):
            traj = trajs[np.random.randint(len(trajs))]
            T = len(traj.actions)
            start = 0 if T <= seq_len else np.random.randint(0, T - seq_len + 1)
            end = min(start + seq_len, T)

            o = traj.obs[start:end]
            a = traj.actions[start:end]
            rtg = traj.returns_to_go[start:end]
            ts = traj.timesteps[start:end]

            pad = seq_len - len(a)
            if o.ndim == 2:
                o = np.pad(o, ((0, pad), (0, 0)))
            else:
                o = np.pad(o, ((0, pad), (0, 0), (0, 0), (0, 0)))
            a = np.pad(a, (0, pad), constant_values=-1)
            rtg = np.pad(rtg, (0, pad))
            ts = np.pad(ts, (0, pad))

            batch.append((o, a, rtg[:, None], ts))

        obs = torch.tensor(
            np.stack([b[0] for b in batch]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            np.stack([b[1] for b in batch]), dtype=torch.long, device=device
        )
        rtg = torch.tensor(
            np.stack([b[2] for b in batch]), dtype=torch.float32, device=device
        )  # type: ignore[assignment]
        ts = torch.tensor(
            np.stack([b[3] for b in batch]), dtype=torch.long, device=device
        )  # type: ignore[assignment]

        yield obs, actions, rtg, ts


def unpack_temp_batch(batch):
    """Unpack temporary batch.

    Args:
        batch: Tuple from make_temp_minibatches.

    Returns:
        Tuple of (obs, actions, rtg, ts) and all-ones mask.
    """
    obs, actions, rtg, ts = batch
    mask = (actions != -1).float()
    actions_safe = actions.clone()
    actions_safe[actions == -1] = 0
    return obs, actions_safe, rtg, ts, mask
