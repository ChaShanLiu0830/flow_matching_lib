import torch
import torch.nn as nn
from beartype import beartype
from beartype.typing import Optional, Tuple
from torch import Tensor
from ..methods.base_cfm import BaseCFM
from ..utils.ot_planner import OptimalTransportPlanner

class COT_CFM(BaseCFM):
    """Conditional Optimal Transport Conditional Flow Matching implementation.
    
    COT-CFM extends OT-CFM by using conditional optimal transport to find correspondences
    between source and target distributions based on both spatial proximity and condition similarity.
    
    This method can capture more complex conditional trajectories by following
    the conditional optimal transport paths that respect both spatial and condition constraints.
    """
    
    @beartype
    def __init__(
        self, 
        sigma: float = 1,
        metric: str = "euclidean",
        normalize: bool = True,
        kernel_bandwidth: float = 1.0,
        **kwargs
    ):
        """Initialize the COT-CFM method.

        Args:
            sigma (float, optional): Noise level for stochastic trajectories. Defaults to 1.
            metric (str, optional): Distance metric for OT computation. Defaults to "euclidean".
            normalize (bool, optional): Whether to normalize distributions. Defaults to True.
            kernel_bandwidth (float, optional): Bandwidth for the kernel used in condition matching. Defaults to 1.0.
        """
        super().__init__(sigma, **kwargs)
        self.ot_planner = OptimalTransportPlanner(
            metric=metric,
            normalize=normalize
        )
        self.kernel_bandwidth = kernel_bandwidth

    def batch_transform(
        self, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        z0: Tensor,
        z1: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Transform batch of samples using conditional OT matching with different conditions for source and target.

        Args:
            x0 (Tensor): Initial state of shape (batch_size, data_dim).
            x1 (Tensor): Target state of shape (batch_size, data_dim).
            t (Tensor): Time points of shape (batch_size, 1).
            z0 (Tensor): Conditional latent code for source.
            z1 (Tensor): Conditional latent code for target.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Transformed inputs with conditional OT matching.
        """
        # Use conditional OT matching with different conditions for source and target
        matched_x0, matched_x1, matched_z0, matched_z1 = self.ot_planner.get_conditional_matched_pairs(
            x0, x1, z0, z1, self.kernel_bandwidth
        )
        
        return matched_x0, matched_x1, t, matched_z0, matched_z1 


# Example usage
if __name__ == "__main__":
    # Create some example data
    x0 = torch.randn(100, 2)  # Source distribution
    x1 = torch.randn(100, 2) + torch.tensor([2.0, 2.0])  # Target distribution
    t = torch.rand(100, 1)  # Random time points
    z = torch.randn(100, 3)  # Conditional latent code
    
    # Create different conditions for source and target
    z0 = torch.randn(100, 3)  # Source condition
    z1 = torch.randn(100, 3)  # Target condition
    
    # Initialize COT-CFM
    cot_cfm = COT_CFM(sigma=0.1, kernel_bandwidth=0.5)
    
    # Compute vector field with same condition
    vector_field = cot_cfm.compute_vector_field(x0, x1, t, z)
    print(f"Vector field shape: {vector_field.shape}")
    
    # Compute intermediate points
    x_t = cot_cfm.compute_xt(x0, x1, t)
    print(f"Intermediate points shape: {x_t.shape}")
    
    # Transform batch with same condition
    matched_x0, matched_x1, t_new, z_new = cot_cfm.batch_transform(x0, x1, t, z)
    print(f"Matched shapes (same condition): {matched_x0.shape}, {matched_x1.shape}")
    
    # Transform batch with different conditions
    matched_x0, matched_x1, t_new, z0_new, z_t = cot_cfm.batch_transform_with_different_conditions(x0, x1, t, z0, z1)
    print(f"Matched shapes (different conditions): {matched_x0.shape}, {matched_x1.shape}")
    print(f"Interpolated condition shape: {z_t.shape}")
