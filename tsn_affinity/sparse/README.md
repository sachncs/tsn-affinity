# Sparse Module

Top-k straight-through estimator and sparse layer implementations.

## Components

- `topk.py`: TopKMaskSTE autograd function
- `linear.py`: TSNSparseLinear
- `conv2d.py`: TSNSparseConv2d
- `embedding.py`: TSNSparseEmbedding
- `base.py`: TSNNMaskMixin interface
- `converter.py`: Module conversion, k-means quantization, optimizer rebuild

## Extension Points

Add new sparse layer types by inheriting from both `nn.Module` and `TSNNMaskMixin`.
