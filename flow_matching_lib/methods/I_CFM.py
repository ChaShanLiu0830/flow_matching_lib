import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional
from torch import Tensor
from .base import BaseCFM

class I_CFM(BaseCFM):
    """Independent Conditional Flow Matching implementation.
    
    This class implements the Independent Flow Matching method where the vector field
    is computed independently at each time step without considering the path constraints.
    """
    
    @beartype
    def __init__(self, nn_model: nn.Module):
        """Initialize the Independent Flow Matching model.

        Args:
            nn_model (nn.Module): Neural network model to compute the base vector field.
                                Should take (x0, x1, t, z) as input and output a vector field.
        """
        super(I_CFM, self).__init__(nn_model)
    
    @beartype
    def compute_vector_field(self, v: Tensor, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
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
        return torch.mean((v_nn - u_t) ** 2)
    
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
        return x_t
    