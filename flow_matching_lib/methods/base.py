import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from beartype import beartype
from beartype.typing import Optional, Union
from torch import Tensor

class BaseCFM(nn.Module, ABC):
    @beartype
    def __init__(self, nn_model: nn.Module):
        """Initialize the base Conditional Flow Matching model.

        Args:
            nn_model (nn.Module): Neural network model to compute the base vector field.
                                Should take (x0, x1, t, z) as input and output a vector field.

        Note:
            The neural network should handle the concatenation of inputs internally.
        """
        super(BaseCFM, self).__init__()
        self.nn_model = nn_model

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
    
    @beartype
    def compute_probability(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute transition probability between states.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Transition probabilities of shape (batch_size, 1).
        """
        pass
    
    @beartype
    def batch_transform(self, x0: Tensor, x1: Tensor, t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Transform batch of samples using the flow matching process.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tensor: Transformed inputs. 
        """
        return x0, x1, t, z 