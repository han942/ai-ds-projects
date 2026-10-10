"""Residual-quantized review autoencoder with explicit train-only scaling.

A review receives a tuple of codebook indices; it is not a unique restaurant
identifier. The decoder reconstructs the original frozen embedding space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RQVAEConfig:
    input_dimension: int = 384
    hidden_dimension: int = 128
    latent_dimension: int = 32
    levels: int = 3
    codebook_size: int = 256
    commitment_weight: float = 0.25
    learning_rate: float = 0.001
    batch_size: int = 512
    warmup_epochs: int = 5
    epochs: int = 40
    seed: int = 42
    threads: int = 4

    def __post_init__(self):
        for name in ('input_dimension', 'hidden_dimension', 'latent_dimension', 'levels',
                     'codebook_size', 'batch_size', 'epochs', 'threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.warmup_epochs < 0 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Invalid training configuration')
        if not math.isfinite(self.commitment_weight) or self.commitment_weight < 0:
            raise ValueError('Invalid commitment weight')

    def to_dict(self):
        return asdict(self)


class ResidualQuantizer(nn.Module):
    def __init__(self, levels, size, dimension, beta=0.25):
        super().__init__()
        self.books = nn.ModuleList([nn.Embedding(size, dimension) for _ in range(levels)])
        self.beta = beta

    def forward(self, latent):
        residual = latent
        quantized = torch.zeros_like(latent)
        loss = latent.new_zeros(())
        codes = []
        for book in self.books:
            with torch.no_grad():
                distances = (residual.square().sum(1, keepdim=True) + book.weight.square().sum(1)
                             - 2 * residual @ book.weight.T)
                indices = distances.argmin(1)
            picked = book(indices)
            loss = loss + F.mse_loss(picked, residual.detach())
            loss = loss + self.beta * F.mse_loss(residual, picked.detach())
            quantized = quantized + picked
            residual = residual - picked.detach()
            codes.append(indices)
        # Reconstruction gradients reach the encoder; codebooks receive their
        # own codebook loss rather than a gradient through nearest-neighbour IDs.
        straight_through = latent + (quantized - latent).detach()
        return straight_through, torch.stack(codes, dim=1), loss

    def decode(self, codes):
        if codes.ndim != 2 or codes.shape[1] != len(self.books):
            raise ValueError('Code shape differs from number of residual levels')
        if codes.dtype not in (torch.int32, torch.int64):
            raise ValueError('Codes must be integers')
        if torch.any(codes < 0) or torch.any(codes >= self.books[0].num_embeddings):
            raise ValueError('Code index outside codebook')
        return sum(book(codes[:, level]) for level, book in enumerate(self.books))

    @torch.no_grad()
    def initialize(self, latent, seed=42):
        from sklearn.cluster import MiniBatchKMeans
        residual = latent.detach().cpu().numpy().copy()
        if len(residual) < self.books[0].num_embeddings:
            raise ValueError('Fewer initialization vectors than codebook entries')
        for level, book in enumerate(self.books):
            fit = MiniBatchKMeans(n_clusters=book.num_embeddings, random_state=seed + level,
                                 n_init=3, max_iter=100, batch_size=1024)
            assignments = fit.fit_predict(residual)
            centers = np.asarray(fit.cluster_centers_, dtype=np.float32)
            book.weight.copy_(torch.from_numpy(centers))
            residual -= centers[assignments]


class ReviewRQVAE(nn.Module):
    def __init__(self, config=RQVAEConfig()):
        super().__init__()
        self.config = config
        d, h, z = config.input_dimension, config.hidden_dimension, config.latent_dimension
        self.encoder = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z))
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
        self.quantizer = ResidualQuantizer(config.levels, config.codebook_size, z, config.commitment_weight)
        self.register_buffer('input_mean', torch.zeros(d))
        self.register_buffer('input_scale', torch.ones(()))

    @torch.no_grad()
    def set_scaling(self, train_vectors):
        if train_vectors.ndim != 2 or train_vectors.shape[1] != self.config.input_dimension:
            raise ValueError('Input dimension mismatch')
        if len(train_vectors) == 0 or not torch.isfinite(train_vectors).all():
            raise ValueError('Empty or non-finite training vectors')
        self.input_mean.copy_(train_vectors.mean(0))
        self.input_scale.copy_((train_vectors - self.input_mean).square().mean().sqrt().clamp_min(1e-6))

    def scaled(self, vectors):
        return (vectors - self.input_mean) / self.input_scale

    def forward(self, vectors, *, quantize=True):
        inputs = self.scaled(vectors)
        latent = self.encoder(inputs)
        if quantize:
            quantized, codes, quantization_loss = self.quantizer(latent)
        else:
            quantized, codes, quantization_loss = latent, None, inputs.new_zeros(())
        output = self.decoder(quantized)
        reconstruction_loss = F.mse_loss(output, inputs)
        reconstructed = output * self.input_scale + self.input_mean
        return reconstructed, codes, reconstruction_loss, quantization_loss

    def decode_codes(self, codes):
        return self.decoder(self.quantizer.decode(codes)) * self.input_scale + self.input_mean

    @torch.no_grad()
    def representations(self, vectors, batch_size=2048):
        self.eval()
        reconstructed, continuous, codes = [], [], []
        tensor = torch.as_tensor(vectors, dtype=torch.float32)
        for batch in tensor.split(batch_size):
            rq, ids, _, _ = self(batch)
            ae, _, _, _ = self(batch, quantize=False)
            reconstructed.append(rq.numpy()); continuous.append(ae.numpy()); codes.append(ids.numpy())
        return np.concatenate(reconstructed), np.concatenate(continuous), np.concatenate(codes)
