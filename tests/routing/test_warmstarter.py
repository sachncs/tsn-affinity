"""Tests for MaskWarmstarter."""

import torch

from tsn_affinity.routing.warmstarter import MaskWarmstarter


class TestMaskWarmstarter:
    def test_noop_when_disabled(self):
        warm = MaskWarmstarter(warmstart=False)
        model = torch.nn.Linear(10, 5)
        warm.apply(model, 0, {}, [])

    def test_noop_when_source_none(self):
        warm = MaskWarmstarter(warmstart=True)
        model = torch.nn.Linear(10, 5)
        warm.apply(model, None, {}, [])

    def test_init_values(self):
        warm = MaskWarmstarter(warmstart=True, strength=3.0, noise_std=0.05)
        assert warm.warmstart is True
        assert warm.strength == 3.0
        assert warm.noise_std == 0.05

    def test_apply_with_sparse_model(self):
        from tsn_affinity.sparse.linear import TSNSparseLinear
        from tsn_affinity.strategies.model_copy import ModelCopy

        warm = MaskWarmstarter(warmstart=True, strength=2.0, noise_std=0.0)
        model = torch.nn.Sequential(TSNSparseLinear(4, 8))

        src_masks = {
            "0.weight": torch.ones(8, 4),
            "0.bias": torch.ones(8),
        }
        src_copy = ModelCopy(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            per_task_masks={0: src_masks},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )

        warm.apply(model, 0, {0: 0}, [src_copy])

        score = model[0].score
        assert score is not None
        assert torch.allclose(score, 2.0 * src_masks["0.weight"], atol=1e-5)

    def test_apply_missing_source_copy(self):
        warm = MaskWarmstarter(warmstart=True)
        model = torch.nn.Linear(4, 8)
        warm.apply(model, 0, {}, [])

    def test_apply_missing_masks(self):
        from tsn_affinity.strategies.model_copy import ModelCopy

        warm = MaskWarmstarter(warmstart=True)
        model = torch.nn.Linear(4, 8)
        src_copy = ModelCopy(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            per_task_masks={},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )
        warm.apply(model, 0, {0: 0}, [src_copy])

    def test_apply_to_new_copy(self):
        from tsn_affinity.sparse.linear import TSNSparseLinear
        from tsn_affinity.strategies.model_copy import ModelCopy

        warm = MaskWarmstarter(warmstart=True, strength=2.0, noise_std=0.0)
        model = torch.nn.Sequential(TSNSparseLinear(4, 8))

        src_masks = {
            "0.weight": torch.ones(8, 4),
        }
        src_copy = ModelCopy(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters()),
            per_task_masks={0: src_masks},
            consolidated_masks={},
            task_codebooks={},
            task_keep_ratios={},
        )

        warm.apply_to_new_copy(model, 0, {0: 0}, [src_copy])
        assert torch.allclose(model[0].score, 2.0 * src_masks["0.weight"], atol=1e-5)
