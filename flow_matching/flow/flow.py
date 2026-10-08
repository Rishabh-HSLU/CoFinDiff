import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """
    Convert a scalar time step t into a high-dimensional frequency vector using sinusoidal embeddings.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - t: shape (batch_size, 1)
        Returns:
            - time_embedding: shape (batch_size, embedding_dim)
        """
        device = t.device
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class ConditionalVectorField(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, time_emb_dim: int, hidden_dim: int):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim)
        )

        self.time_proj1 = nn.Linear(time_emb_dim, hidden_dim)
        self.time_proj2 = nn.Linear(time_emb_dim, hidden_dim)

        self.linear1 = nn.Linear(in_dim + z_dim , hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, in_dim)


    def forward(self, xt:torch.Tensor, z:torch.Tensor, t:torch.Tensor)->torch.Tensor:
        """
        Args:
            - xt: state at time t, shape (batch_size, dim)
            - z: conditioning variable, shape (batch_size, dim_z)
            - t: time, shape (batch_size, 1)
        Returns:
            - vector_field: shape (batch_size, dim)
        """
        x = torch.cat((xt, z), dim=-1)
        time_emb = self.time_mlp(t)

        x = self.linear1(x)
        x = x + self.time_proj1(time_emb)
        x = torch.nn.functional.silu(x)
        x = self.linear2(x)
        x = x + self.time_proj2(time_emb)
        x = torch.nn.functional.silu(x)
        x = self.linear3(x)

        return x



class TargetVectorField:
    def __init__(self, p_simple, p_data):
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
    pass