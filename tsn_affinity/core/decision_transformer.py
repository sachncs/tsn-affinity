"""Decision Transformer model implementation."""

import numpy as np
import torch
import torch.nn as nn

from tsn_affinity.core.attention import Block, LayerNorm
from tsn_affinity.core.encoder import ObsEncoder


class DTBackbone(nn.Module):
    """Decision Transformer backbone.

    Processes interleaved sequences of returns-to-go, states, and actions
    as token sequences. Uses learned token embeddings and timestep encodings.

    The sequence format is: (R_0, s_0, a_0, R_1, s_1, a_1, ...)

    Attributes:
        n_actions: Number of possible actions (discrete).
        action_dim: Dimension of continuous actions.
        continuous_actions: Whether actions are continuous.
        d_model: Embedding dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        seq_len: Maximum context length.
        p_drop: Dropout probability.
        max_ep_len: Maximum episode length for timestep encoding.
        rtg_scale: Scale factor for returns-to-go inputs.
    """

    def __init__(
        self,
        n_actions: int,
        action_dim: int = 1,
        continuous_actions: bool = False,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 20,
        p_drop: float = 0.1,
        max_ep_len: int = 10000,
        rtg_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.d_model = d_model
        self.seq_len = seq_len
        self.rtg_scale = rtg_scale

        self.rtg_embed = nn.Linear(1, d_model)
        self.state_embed: nn.Module | None = None  # Set by set_obs_encoder
        self.action_embed: nn.Module
        if continuous_actions:
            self.action_embed = nn.Linear(action_dim, d_model)
        else:
            self.action_embed = nn.Embedding(n_actions, d_model)
        self.timestep_embed = nn.Embedding(max_ep_len, d_model)
        self.dropout = nn.Dropout(p_drop)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, p_drop) for _ in range(n_layers)]
        )
        self.final_norm = LayerNorm(d_model, bias=True)

        if continuous_actions:
            self.action_head = nn.Linear(d_model, action_dim)
        else:
            self.action_head = nn.Linear(d_model, n_actions)

    def set_obs_encoder(self, obs_encoder: nn.Module) -> None:
        """Set the observation encoder."""
        self.state_embed = obs_encoder

    def forward(
        self,
        rtg: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Decision Transformer.

        Args:
            rtg: [B, K, 1] returns-to-go (already scaled).
            state: [B, K, ...] or [B, K, D] observation sequence.
            action: [B, K] action indices (discrete) or [B, K, action_dim] (continuous).
            timesteps: [B, K] timestep indices.
            attention_mask: [B, K] binary mask (1 = valid, 0 = pad).

        Returns:
            Action logits [B, K, n_actions] (discrete) or predictions
            [B, K, action_dim] (continuous).
        """
        B, K = action.shape[:2]

        # Embed all modalities - rtg is [B, K, 1], state is [B, K, ...],
        # action is [B, K] or [B, K, action_dim]
        rtg_emb = self.rtg_embed((rtg / self.rtg_scale).reshape(B * K, 1)).reshape(
            B, K, self.d_model
        )

        if self.state_embed is not None:
            if len(state.shape) == 5:
                state_enc = self.state_embed(
                    state.reshape(B * K, *state.shape[2:])
                ).reshape(B, K, self.d_model)
            elif len(state.shape) == 4:
                state_enc = self.state_embed(
                    state.reshape(B * K, *state.shape[2:])
                ).reshape(B, K, self.d_model)
            elif len(state.shape) == 3:
                state_enc = self.state_embed(state.reshape(B * K, -1)).reshape(
                    B, K, self.d_model
                )
            else:
                raise ValueError(f"Unexpected state shape: {tuple(state.shape)}")
        else:
            raise RuntimeError("Obs encoder not set. Call set_obs_encoder() first.")

        if self.continuous_actions:
            action_emb = self.action_embed(
                action.reshape(B * K, self.action_dim)
            ).reshape(B, K, self.d_model)
        else:
            action_emb = self.action_embed(action.long())

        # Timestep encoding
        time_emb = self.timestep_embed(timesteps.long())

        # Interleave: [R_0, s_0, a_0, R_1, s_1, a_1, ...]
        x = torch.stack([rtg_emb, state_enc, action_emb], dim=2).reshape(
            B, K * 3, self.d_model
        )
        x = x + time_emb.repeat(1, 3, 1)
        x = self.dropout(x)

        # Build attention mask for interleaved tokens
        if attention_mask is not None:
            mask_3x = attention_mask.unsqueeze(-1).repeat(1, 1, 3).reshape(B, K * 3)
        else:
            mask_3x = None

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask_3x)

        x = self.final_norm(x)

        # Extract state-position logits (every 3rd position, starting at index 1)
        state_logits = x[:, 1::3]

        return self.action_head(state_logits)  # type: ignore[no-any-return]


class DecisionTransformer(nn.Module):
    """High-level Decision Transformer model.

    Combines an observation encoder with a DTBackbone for action prediction.

    Attributes:
        obs_shape: Shape of observations.
        n_actions: Number of actions.
        d_model: Embedding dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        seq_len: Maximum context length.
        p_drop: Dropout probability.
        max_ep_len: Maximum episode length.
        rtg_scale: Returns-to-go scale factor.
        max_ep_len: Maximum episode length for timestep encoding.
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        action_dim: int = 1,
        continuous_actions: bool = False,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        seq_len: int = 20,
        p_drop: float = 0.1,
        max_ep_len: int = 10000,
        rtg_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.n_actions = n_actions
        self.action_dim = action_dim
        self.continuous_actions = continuous_actions
        self.d_model = d_model
        self.seq_len = seq_len
        self.max_ep_len = max_ep_len
        self.rtg_scale = rtg_scale

        self.obs_enc = ObsEncoder(obs_shape, d_model)
        self.backbone = DTBackbone(
            n_actions=n_actions,
            action_dim=action_dim,
            continuous_actions=continuous_actions,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            seq_len=seq_len,
            p_drop=p_drop,
            max_ep_len=max_ep_len,
            rtg_scale=rtg_scale,
        )
        self.backbone.set_obs_encoder(self.obs_enc)

    def forward(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: Input observations.
            actions: Input actions.
            returns_to_go: Returns-to-go values.
            timesteps: Timestep indices.
            attention_mask: Optional attention mask.

        Returns:
            Action logits or predictions.
        """
        return self.backbone(returns_to_go, obs, actions, timesteps, attention_mask)  # type: ignore[no-any-return]

    def reset_history(self) -> None:
        """Clear the action history buffers."""
        self._obs_history: list[torch.Tensor] = []
        self._act_history: list[int | np.ndarray] = []
        self._rtg_history: list[float] = []
        self._ts_history: list[int] = []

    def act(
        self,
        obs: torch.Tensor,
        returns_to_go: float,
        deterministic: bool = True,
    ) -> int | np.ndarray:
        """Select action given current observation (online inference).

        Maintains rolling history of (obs, action, rtg, timestep) and
        performs autoregressive action selection.

        Args:
            obs: Current observation.
            returns_to_go: Current return-to-go target.
            deterministic: If True, select argmax (discrete) or mean
                (continuous); else sample.

        Returns:
            Selected action index (discrete) or action vector (continuous).
        """
        device = next(self.parameters()).device

        obs_t = obs.to(device)
        self._obs_history.append(obs_t)
        self._rtg_history.append(returns_to_go / self.rtg_scale)
        self._ts_history.append(len(self._ts_history))

        # Truncate to seq_len
        obs_history = self._obs_history[-self.seq_len :]
        act_history = self._act_history[-self.seq_len :]
        rtg_history = self._rtg_history[-self.seq_len :]
        ts_history = self._ts_history[-self.seq_len :]

        if self.continuous_actions:
            obs_seq = torch.stack(obs_history)
            pad_len = self.seq_len - len(obs_history)
            if pad_len > 0:
                obs_pad = torch.zeros(
                    pad_len, *obs_seq.shape[1:], device=device, dtype=obs_seq.dtype
                )
                obs_seq = torch.cat([obs_pad, obs_seq], dim=0)
            if obs_seq.dim() == 3:
                obs_seq = obs_seq.unsqueeze(0)
            elif obs_seq.dim() == 2:
                obs_seq = obs_seq.unsqueeze(0)

            if len(act_history) > 0:
                act_seq = (
                    torch.tensor(np.stack(act_history), device=device)
                    .unsqueeze(0)
                    .float()
                )
                pad_len = self.seq_len - len(act_history)
                if pad_len > 0:
                    act_pad = torch.zeros(1, pad_len, self.action_dim, device=device)
                    act_seq = torch.cat([act_pad, act_seq], dim=1)
            else:
                act_seq = torch.zeros(1, self.seq_len, self.action_dim, device=device)

            rtg_seq = (
                torch.tensor(rtg_history, device=device).unsqueeze(0).unsqueeze(-1)
            )
            ts_seq = torch.tensor(ts_history, device=device).unsqueeze(0).long()

            if len(rtg_history) > 0:
                rtg_seq = torch.nn.functional.pad(
                    rtg_seq, (0, 0, 0, self.seq_len - rtg_seq.shape[1]), value=0.0
                )
            else:
                rtg_seq = torch.zeros(
                    1, self.seq_len, 1, device=device, dtype=rtg_seq.dtype
                )

            if len(ts_history) > 0:
                ts_seq = torch.nn.functional.pad(
                    ts_seq, (0, self.seq_len - ts_seq.shape[1]), value=0
                )
            else:
                ts_seq = torch.zeros(1, self.seq_len, device=device, dtype=torch.long)

            mask = torch.ones(1, self.seq_len, device=device)

            pred = self.forward(obs_seq, act_seq, rtg_seq, ts_seq, mask)
            action = pred[0, -1]
            act: int | np.ndarray = action.detach().cpu().numpy()
        else:
            # Build tensors - pad to full seq_len at the beginning
            obs_seq = torch.stack(obs_history)  # [K', C, H, W]
            act_seq = (
                torch.tensor(act_history, device=device).unsqueeze(0).long()
            )  # [1, K]
            rtg_seq = (
                torch.tensor(rtg_history, device=device).unsqueeze(0).unsqueeze(-1)
            )  # [1, K, 1]
            ts_seq = (
                torch.tensor(ts_history, device=device).unsqueeze(0).long()
            )  # [1, K]

            # Pad observation sequence to full seq_len at the beginning
            pad_len = self.seq_len - len(obs_history)
            if pad_len > 0:
                obs_pad = torch.zeros(
                    pad_len, *obs_seq.shape[1:], device=device, dtype=obs_seq.dtype
                )
                obs_seq = torch.cat([obs_pad, obs_seq], dim=0)

            if len(act_history) > 0:
                act_seq = torch.nn.functional.pad(
                    act_seq, (0, self.seq_len - act_seq.shape[1]), value=0
                )
            else:
                act_seq = torch.zeros(1, self.seq_len, device=device, dtype=torch.long)

            if len(rtg_history) > 0:
                rtg_seq = torch.nn.functional.pad(
                    rtg_seq, (0, 0, 0, self.seq_len - rtg_seq.shape[1]), value=0.0
                )
            else:
                rtg_seq = torch.zeros(
                    1, self.seq_len, 1, device=device, dtype=rtg_seq.dtype
                )

            if len(ts_history) > 0:
                ts_seq = torch.nn.functional.pad(
                    ts_seq, (0, self.seq_len - ts_seq.shape[1]), value=0
                )
            else:
                ts_seq = torch.zeros(1, self.seq_len, device=device, dtype=torch.long)

            mask = torch.ones(1, self.seq_len, device=device)
            logits = self.forward(obs_seq.unsqueeze(0), act_seq, rtg_seq, ts_seq, mask)
            action = logits[0, -1]

            if deterministic:
                act = int(action.argmax())
            else:
                act = int(torch.multinomial(torch.softmax(action, dim=-1), 1).item())

        self._act_history.append(act)
        return act
