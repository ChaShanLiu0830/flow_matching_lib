import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from beartype import beartype
from beartype.typing import Optional, Union, Tuple, Any, List, Dict
from torch import Tensor

class BaseCFM(ABC):
    """Base class for Conditional Flow Matching (CFM) methods.
    
    CFM is a framework for learning continuous normalizing flows by matching vector fields
    between pairs of samples. This base class provides the interface for different CFM methods.
    
    The flow is defined by an ODE: dx/dt = v(x,t), where v is the vector field.
    Different CFM methods specify different ways to compute this vector field.
    """
    
    @beartype
    def __init__(self, sigma: float = 1.0, **kwargs):
        """Initialize the base CFM method.

        Args:
            sigma (float): Noise level for stochastic trajectories.
                         Higher values lead to more exploration but less precise paths.
                         Defaults to 1.
        """
        super().__init__()
        self.sigma = sigma

    @abstractmethod
    @beartype
    def compute_vector_field(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the target vector field based on the CFM method.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Modified vector field of shape (batch_size, data_dim).
        """
        pass

    @beartype
    def loss_fn(self, v_nn: Tensor, u_t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the loss between predicted and target vector fields.

        Args:
            v_nn (Tensor): Predicted vector field from neural network of shape (batch_size, data_dim).
            u_t (Tensor): Target vector field of shape (batch_size, data_dim).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Scalar loss value.
        """
        pass
    
    @beartype
    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the intermediate state x_t between x0 and x1.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Intermediate state x_t of shape (batch_size, data_dim).
        """
        pass
    
    
    def batch_transform(self, *args) -> Any:
        pass