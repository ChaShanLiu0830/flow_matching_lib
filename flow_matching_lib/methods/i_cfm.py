import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional
from torch import Tensor
from ..methods.base_cfm import BaseCFM

class I_CFM(BaseCFM):
    """Independent Conditional Flow Matching implementation.
    
    I-CFM is the simplest form of conditional flow matching where:
    1. The vector field is computed as v(x,t) = x1 - x0 (straight path)
    2. Intermediate states are linear interpolations with added noise
    3. The loss is a simple MSE between predicted and target vector fields
    
    This method is computationally efficient but may not capture complex trajectories.
    """
    
    @beartype
    def __init__(self, sigma: float = 1):
        """Initialize the I-CFM method.

        Args:
            sigma (float): Noise level for stochastic trajectories.
                         Controls the variance of Gaussian noise added to interpolated states.
                         Defaults to 1.
        """
        super().__init__(sigma)
    
    @beartype
    def compute_vector_field(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the vector field using Independent Flow Matching.

        Args:
            v (Tensor): Base vector field from neural network.
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Modified vector field of shape (batch_size, data_dim).
        """
        # In I-CFM, we directly use the network output as the vector field
        return x1 - x0
    
    @beartype
    def loss_fn(self, v_nn: Tensor, u_t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the I-CFM loss between predicted and target vector fields.

        Args:
            v_nn (Tensor): Predicted vector field from neural network of shape (batch_size, data_dim).
            u_t (Tensor): Target vector field of shape (batch_size, data_dim).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Mean squared error between predicted and target vector fields.
        """
        # Simple MSE loss between predicted and target vector fields
        return torch.mean((u_t - v_nn) ** 2)
    
    @beartype
    def compute_xt(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the intermediate state x_t using linear interpolation.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Intermediate state x_t of shape (batch_size, data_dim).
        """
        # Linear interpolation between x0 and x1
        x_t = (1 - t) * x0 + t * x1
        return x_t + self.sigma * torch.randn_like(x_t)

    