import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Optional, Tuple, Callable
from beartype import beartype
from torch import Tensor
from ..methods.base_cfm import BaseCFM
from tqdm import tqdm

class BaseSampler:
    """Base class for sampling from Conditional Flow Matching models.
    
    This class implements various sampling strategies for CFM models:
    1. Trajectory sampling: Generate continuous paths from start to end points
    2. Point sampling: Sample specific points along trajectories
    3. Batch sampling: Process multiple samples efficiently
    
    The sampling is done by solving the ODE: dx/dt = v(x,t) using numerical methods.
    """
    
    @beartype
    def __init__(
        self,
        cfm: BaseCFM,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "dopri5",
    ):
        """Initialize the sampler.

        Args:
            cfm (BaseCFM): The CFM method used for computing vector fields.
            model (nn.Module): Neural network model that predicts the vector field.
            device (str, optional): Device to run sampling on. Defaults to "cuda" if available.
            rtol (float, optional): Relative tolerance for ODE solver. Defaults to 1e-5.
            atol (float, optional): Absolute tolerance for ODE solver. Defaults to 1e-5.
            method (str, optional): ODE solver method. Defaults to "dopri5".
        """
        self.cfm = cfm
        self.model = model.to(device)
        self.device = device
        self.rtol = rtol
        self.atol = atol
        self.method = method
        
    def vector_field_fn(self, t: Tensor, x: Tensor, z0: Optional[Tensor] = None, z1: Optional[Tensor]= None) -> Tensor:
        """Compute the vector field at a given time and state.

        Args:
            t (Tensor): Current time point.
            x (Tensor): Current state.
            z (Optional[Tensor], optional): Conditional code. Defaults to None.

        Returns:
            Tensor: Vector field at (x,t).
        """
        t = t.reshape(1, 1).expand(x.shape[0], 1)
        with torch.no_grad():
            v = self.model(x, t, z0, z1) if z0 is not None else self.model(x, t)
        return v
    
    @beartype
    def sample_trajectory(
        self, 
        x: Tensor,
        start_t: float = 0.0,
        end_t: float = 1.0,
        n_points: int = 100,
        z0: Optional[Tensor] = None,
        z1: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Sample a continuous trajectory starting from given points.

        Args:
            x (Tensor): Starting points of shape (batch_size, data_dim).
            start_t (float, optional): Start time. Defaults to 0.0.
            end_t (float, optional): End time. Defaults to 1.0.
            n_points (int, optional): Number of points to sample along trajectory. Defaults to 100.
            z0 (Optional[Tensor], optional): Conditional code for source. Defaults to None.
            z1 (Optional[Tensor], optional): Conditional code for target. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing:
                - t_eval (Tensor): Time points of shape (n_points,)
                - trajectory (Tensor): Sampled points of shape (batch_size, n_points, data_dim)
        """
        self.model.eval()
        t_eval = torch.linspace(start_t, end_t, n_points, device=self.device)
        
        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            return self.vector_field_fn(t, x, z0, z1)
        
        # Solve ODE - output shape is (n_points, batch_size, data_dim)
        trajectory = odeint(
            ode_func,
            x,
            t_eval,
            rtol=self.rtol,
            atol=self.atol,
            method=self.method,
        )
        
        # Permute to (batch_size, n_points, data_dim)
        # trajectory = trajectory.permute(1, 0, 2)
        
        return trajectory[-1, ...], trajectory
    
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
            show_progress (bool, optional): Whether to show progress bar. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor, Optional[Tensor]]: Tuple containing:
                - x (Tensor): Starting points of shape (total_batch_size, data_dim)
                - samples (Tensor): Sampled points of shape (total_batch_size, num_samples, data_dim)
                - z (Optional[Tensor]): Conditional codes if provided
        """
        self.model.eval()
        
        all_gen_samples = []
        
        # dataloader = 
            
        with torch.no_grad():
            for _ in tqdm(range(num_samples), desc="Sampling numbers", leave=False):
                gen_samples = []
                for batch in tqdm(dataloader, desc="Sampling batches", leave=False):
                    x = batch['x0'].to(self.device)
                    z0 = batch.get('z0')
                    z1 = batch.get('z1')
                    if z0 is not None:
                        z0 = z0.to(self.device)
                    if z1 is not None:
                        z1 = z1.to(self.device)
                
                    samples, _ = self.sample_trajectory(x, start_t, end_t, n_points, z0, z1)
                    
                    # all_x.append(x)
                    gen_samples.append(samples)
                gen_samples = torch.cat(gen_samples, dim=0).unsqueeze(-1)
                all_gen_samples.append(gen_samples)
            all_gen_samples = torch.cat(all_gen_samples, dim=-1)
                # if z is not None:
                #     all_z.append(z)
        
        # x = torch.cat(all_x, dim=0)
        
        # z = torch.cat(all_z, dim=0) if all_z else None
        
        return all_gen_samples