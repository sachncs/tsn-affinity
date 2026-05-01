"""Data loader interface."""

from collections.abc import Generator

import torch

from tsn_affinity.data.trajectory import Trajectory


class DataLoaderInterface:
    """Interface for trajectory batch loaders."""

    def generate(
        self,
        trajs: list[Trajectory],
        seq_len: int,
        batch_size: int,
        device: str,
    ) -> Generator[tuple[torch.Tensor, ...], None, None]:
        """Generate batches indefinitely.

        Args:
            trajs: List of trajectories.
            seq_len: Sequence length for each batch slice.
            batch_size: Number of trajectories per batch.
            device: Target device for tensors.

        Yields:
            Tuples of tensors (contents depend on implementation).
        """
        raise NotImplementedError
