import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch import Tensor
from ..methods.base_cfm import BaseCFM
from ..methods.i_cfm import I_CFM
from ..utils.ot_planner import OptimalTransportPlanner

class OT_CFM(I_CFM):
    """Optimal Transport Conditional Flow Matching implementation.
    
    OT-CFM uses optimal transport to find correspondences between source and target
    distributions, and then learns the vector field along the optimal transport paths.
    
    This method can capture more complex trajectories than I-CFM by following
    the optimal transport plan between distributions.
    """
    
    @beartype
    def __init__(
        self, 
        sigma: float = 1,
        metric: str = "euclidean",
        normalize: bool = True,
        **kwargs
    ):
        """Initialize the OT-CFM method.

        Args:
            sigma (float, optional): Noise level for stochastic trajectories. Defaults to 1.
            metric (str, optional): Distance metric for OT computation. Defaults to "euclidean".
            normalize (bool, optional): Whether to normalize distributions. Defaults to True.
        """
        super().__init__(sigma, **kwargs)
        self.ot_planner = OptimalTransportPlanner(
            metric=metric,
            normalize=normalize
        )
    
    def batch_transform(
        self, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        *args
    ):
        """Transform batch of samples using the OT matching process.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z (Optional[Tensor], optional): Conditional latent code. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]: Transformed inputs with OT matching.
        """
        # Get matched pairs using OT
        matched_x0, matched_x1 = self.ot_planner.get_matched_pairs(x0, x1)
        
        return matched_x0, matched_x1, t, *args


# Example usage
if __name__ == "__main__":
    # Create some example data
    x0 = torch.randn(100, 2)  # Source distribution
    x1 = torch.randn(100, 2) + torch.tensor([2.0, 2.0])  # Target distribution
    t = torch.rand(100, 1)  # Random time points
    
    # Initialize OT-CFM
    ot_cfm = OT_CFM(sigma=0.1)
    
    # Compute vector field
    vector_field = ot_cfm.compute_vector_field(x0, x1, t)
    print(f"Vector field shape: {vector_field.shape}")
    
    # Compute intermediate points
    x_t = ot_cfm.compute_xt(x0, x1, t)
    print(f"Intermediate points shape: {x_t.shape}")
    
    # Transform batch
    matched_x0, matched_x1, t_new, _ = ot_cfm.batch_transform(x0, x1, t)
    print(f"Matched shapes: {matched_x0.shape}, {matched_x1.shape}")
