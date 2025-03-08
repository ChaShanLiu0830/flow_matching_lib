import torch
import torch.nn as nn
from typing import Optional, Tuple
from beartype import beartype
from torch import Tensor
from tqdm import tqdm
from .base_sampler import BaseSampler
from ..methods.base_cfm import BaseCFM

class GuiderSampler(BaseSampler):
    """Classifier-free guidance sampler for Conditional Flow Matching models.
    
    This sampler implements classifier-free guidance where the final vector field
    is a weighted combination of conditional and unconditional predictions:
        v = w * v_conditional + (1-w) * v_unconditional
    """
    
    @beartype
    def __init__(
        self,
        cfm: BaseCFM,
        model: nn.Module,
        guidance_weight: float = 2.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
    ):
        """Initialize the guided sampler.

        Args:
            cfm (BaseCFM): The CFM method used for computing vector fields.
            model (nn.Module): Neural network model that predicts the vector field.
            guidance_weight (float, optional): Weight for guidance. Defaults to 2.0.
            device (str, optional): Device to run sampling on. Defaults to "cuda" if available.
            rtol (float, optional): Relative tolerance for ODE solver. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance for ODE solver. Defaults to 1e-5.
            method (str, optional): ODE solver method. Defaults to "dopri5".
        """
        super().__init__(cfm, model, device, rtol, atol, method)
        self.guidance_weight = guidance_weight
    
    def vector_field_fn(self, t: Tensor, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Compute the guided vector field at a given time and state.

        Args:
            t (Tensor): Current time point.
            x (Tensor): Current state.
            z (Optional[Tensor], optional): Conditional code. Defaults to None.

        Returns:
            Tensor: Guided vector field at (x,t).
        """
        t = t.reshape(1, 1).expand(x.shape[0], 1)
        
        with torch.no_grad():
            # Get conditional prediction
            v_cond = self.model(x, t, z)
            
            # Get unconditional prediction (zero out condition)
            v_uncond = self.model(x, t, torch.zeros_like(z))
            
            # Combine predictions using guidance weight
            v = v_uncond + self.guidance_weight * (v_cond - v_uncond)
            
        return v
    
    @beartype
    def sample_trajectory(
        self, 
        x: Tensor,
        start_t: float = 0.0,
        end_t: float = 1.0,
        n_points: int = 100,
        z: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample a continuous trajectory starting from given points.

        Args:
            x (Tensor): Starting points of shape (batch_size, data_dim).
            start_t (float, optional): Start time. Defaults to 0.0.
            end_t (float, optional): End time. Defaults to 1.0.
            n_points (int, optional): Number of points to sample along trajectory. Defaults to 100.
            z (Optional[Tensor], optional): Conditional code. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - final_points (Tensor): Final sampled points
                - trajectory (Tensor): Full trajectory
        """
        return super().sample_trajectory(x, start_t, end_t, n_points, z)
    
    @beartype
    def sample_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_samples: int = 1,
        start_t: float = 0.0,
        end_t: float = 1.0,
        n_points: int = 100,
    ) -> Tensor:
        """Sample points for a batch of data.

        Args:
            dataloader (DataLoader): DataLoader containing samples to process.
            num_samples (int, optional): Number of samples per input. Defaults to 1.
            start_t (float, optional): Start time. Defaults to 0.0.
            end_t (float, optional): End time. Defaults to 1.0.
            n_points (int, optional): Number of points to sample. Defaults to 100.

        Returns:
            Tensor: Sampled points of shape (batch_size, num_samples, data_dim)
        """
        return super().sample_batch(dataloader, num_samples, start_t, end_t, n_points)
    
    @beartype
    def set_guidance_weight(self, weight: float) -> None:
        """Set the guidance weight.

        Args:
            weight (float): New guidance weight value.
        """
        self.guidance_weight = weight 