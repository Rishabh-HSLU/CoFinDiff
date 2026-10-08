import torch
from abc import ABC, abstractmethod

class ConditionalProbabilityPath(ABC):
    def __init__(self, p_simple, p_data):
        pass

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_conditioning_variable(self, num_samples: int) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def conditional_vector_field(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def conditional_score(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

class