from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def normalize_keypoints(kpts, image_shape):
    """Normalizes keypoints using the image shape. 
    This method centers and scales the keypoints to ensure that they are within a fixed range.

    Args:
        kpts: Tensor containing the keypoints coordinates.
        image_shape: Shape of the input image (batch size, channels, height, width).

    Returns:
        Tensor containing the normalized keypoints.
    """
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

class KeypointEncoder(nn.Module):
    """Encodes the keypoints and scores into a feature representation.

    This class utilizes a Multi-Layer Perceptron (MLP) to process both the 
    keypoints and their associated scores into a higher-dimensional space.
    
    Args:
        feature_dim: Dimensionality of the output feature vector.
        layers: List specifying the number of units in each hidden layer.
    """
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))

def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the scaled dot-product attention.

    Args:
        query: Tensor representing the query matrix.
        key: Tensor representing the key matrix.
        value: Tensor representing the value matrix.

    Returns:
        A tuple containing the attended result and the attention scores.
    """
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """Implements multi-headed attention mechanism.

    Args:
        num_heads: Number of attention heads.
        d_model: Dimensionality of the input and output embeddings.
    """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    """A layer that performs message passing using attention between nodes (descriptors).

    Args:
        feature_dim: Dimensionality of the features.
        num_heads: Number of attention heads.
    """
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

class AttentionalGNN(nn.Module):
    """Implements a graph neural network (GNN) with attention for feature matching.

    Args:
        feature_dim: Dimensionality of the node (descriptor) features.
        layer_names: Names of the layers ('self' or 'cross') indicating
                     the type of attention to perform at each layer.
    """
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """Performs Sinkhorn normalization in the log-space for numerical stability.

    Args:
        Z: The coupling matrix (scores matrix).
        log_mu: Log of the marginal distribution for the rows.
        log_nu: Log of the marginal distribution for the columns.
        iters: Number of iterations to perform the normalization.

    Returns:
        The normalized coupling matrix (in log-space).
    """
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """Performs differentiable optimal transport in the log-space for numerical stability.

    Args:
        scores: Matrix of matching scores between the two sets of descriptors.
        alpha: Scalar parameter for the optimal transport formulation.
        iters: Number of Sinkhorn iterations to perform.

    Returns:
        The coupling matrix (scores) adjusted by the Sinkhorn iterations.
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # Rescale by M+N probabilities.
    return Z

def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable arange

class SuperGlue(nn.Module):
    """The main SuperGlue model class, implementing feature matching using attention-based GNNs.

    This class orchestrates the keypoint encoding, GNN layers, final MLP projection, 
    and optimal transport to compute matches between two sets of keypoints.

    Args:
        config: Dictionary containing model configuration options.
    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1)

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data: dict) -> dict:
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']
        scores0, scores1 = data['scores0'], data['scores1']
        image0, image1 = data['image0'], data['image1']

        desc0 = torch.nn.functional.normalize(desc0, p=2, dim=1)
        desc1 = torch.nn.functional.normalize(desc1, p=2, dim=1)

        kpts0 = normalize_keypoints(kpts0, image0.shape)
        kpts1 = normalize_keypoints(kpts1, image1.shape)

        desc0 = self.kenc(kpts0, scores0) + desc0
        desc1 = self.kenc(kpts1, scores1) + desc1

        desc0, desc1 = self.gnn(desc0, desc1)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)

        scores = log_optimal_transport(scores, self.bin_score, self.config['sinkhorn_iterations'])

        max0, max1 = scores[:, :-1, :-1].max(2).values, scores[:, :-1, :-1].max(1).values
        indices0 = arange_like(kpts0, 1)[None] - (scores[:, :-1, -1] > max0).float()
        indices1 = arange_like(kpts1, 1)[None] - (scores[:, -1, :-1] > max1).float()

        matches0 = torch.where(indices0 > 0, indices0.long(), indices0.new_tensor(-1))
        matches1 = torch.where(indices1 > 0, indices1.long(), indices1.new_tensor(-1))

        matching_scores0 = torch.where(matches0 > -1, scores[:, :-1, -1].gather(1, matches0), matches0.new_tensor(0))
        matching_scores1 = torch.where(matches1 > -1, scores[:, -1, :-1].gather(1, matches1), matches1.new_tensor(0))

        return {
            'matches0': matches0,
            'matches1': matches1,
            'matching_scores0': matching_scores0,
            'matching_scores1': matching_scores1,
        }
