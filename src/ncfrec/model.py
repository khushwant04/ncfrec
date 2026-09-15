"""NeuMF: generalized matrix factorization plus a nonlinear MLP path."""

from __future__ import annotations

import torch
from torch import nn

from ncfrec.config import ModelConfig


class NeuMF(nn.Module):
    """Neural Matrix Factorization model that emits interaction logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_users = config.num_users
        self.num_items = config.num_items

        self.gmf_user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.gmf_item_embedding = nn.Embedding(config.num_items, config.embedding_dim)
        self.mlp_user_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.mlp_item_embedding = nn.Embedding(config.num_items, config.embedding_dim)

        mlp_modules: list[nn.Module] = []
        input_size = config.embedding_dim * 2
        for output_size in config.mlp_layers:
            mlp_modules.extend(
                [
                    nn.Linear(input_size, output_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_modules)
        self.output = nn.Linear(config.embedding_dim + input_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (
            self.gmf_user_embedding,
            self.gmf_item_embedding,
            self.mlp_user_embedding,
            self.mlp_item_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.01)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized interaction logit for each user/item pair."""
        if user_indices.shape != item_indices.shape:
            raise ValueError("user_indices and item_indices must have the same shape")

        user_indices = user_indices.long()
        item_indices = item_indices.long()

        gmf_vector = self.gmf_user_embedding(user_indices) * self.gmf_item_embedding(item_indices)
        mlp_vector = torch.cat(
            [self.mlp_user_embedding(user_indices), self.mlp_item_embedding(item_indices)],
            dim=-1,
        )
        mlp_vector = self.mlp(mlp_vector)
        features = torch.cat([gmf_vector, mlp_vector], dim=-1)
        return self.output(features).squeeze(-1)

    @torch.inference_mode()
    def predict_proba(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Return sigmoid interaction probabilities without tracking gradients."""
        return torch.sigmoid(self(user_indices, item_indices))
