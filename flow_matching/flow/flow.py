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
    def __init__(self, p_init, p_data):
        super().__init__()
        self.p_init = p_init
        self.p_data = p_data

    def interpolation_path(self, t: torch.Tensor):
        return (1-t) * self.p_init + t * self.p_data

    def forward(self, t: torch.Tensor) -> tuple:
        """
        Computes the target vector field d(xt)/dt using automatic differentiation.
        Args:
            - t: time, shape (batch_size, 1)
        Returns:
            - vector_field: shape (batch_size, dim)
        """
        # 1. Expand 't' to exactly match the shape of the target data (B, dim)
        # We clone it to isolate it from the outer computational graph
        t_expanded = t.expand_as(self.p_init).clone()

        # 2. Tell PyTorch to track operations on this expanded time tensor
        t_expanded.requires_grad_(True)

        # 3. Compute the intermediate state xt
        xt = self.interpolation_path(t_expanded)

        # 4. Use Autograd to compute the time derivative d(xt)/dt
        # grad_outputs=torch.ones_like(xt) extracts the independent element-wise gradients
        vector_field = torch.autograd.grad(
            outputs=xt,
            inputs=t_expanded,
            grad_outputs=torch.ones_like(xt),
            create_graph=True  # Required if you intend to backpropagate through the loss later
        )[0]

        return xt, vector_field