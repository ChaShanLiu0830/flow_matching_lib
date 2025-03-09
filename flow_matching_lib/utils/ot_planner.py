import torch
import numpy as np
import ot
from typing import Union, Optional, Tuple
from beartype import beartype
from torch import Tensor

class OptimalTransportPlanner:
    """Optimal Transport planner for computing transport plans between distributions.
    
    This class provides methods to compute optimal transport plans between
    source and target distributions using the POT (Python Optimal Transport) package.
    """
    
    @beartype
    def __init__(
        self,
        metric: str = "euclidean",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalize: bool = True,
    ):
        """Initialize the OT planner.

        Args:
            metric (str, optional): Distance metric to use. Defaults to "euclidean".
            device (str, optional): Device to use for computation. Defaults to "cuda" if available.
            normalize (bool, optional): Whether to normalize the distributions. Defaults to True.
        """
        self.metric = metric
        self.device = device
        self.normalize = normalize
    
    @staticmethod
    def _to_numpy(x: Union[Tensor, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array.

        Args:
            x (Union[Tensor, np.ndarray]): Input data.

        Returns:
            np.ndarray: Data as numpy array.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x
    
    @staticmethod
    def _to_tensor(x: Union[Tensor, np.ndarray], device: str) -> Tensor:
        """Convert input to PyTorch tensor.

        Args:
            x (Union[Tensor, np.ndarray]): Input data.
            device (str): Device to place tensor on.

        Returns:
            Tensor: Data as PyTorch tensor.
        """
        if isinstance(x, np.ndarray):
            return torch.tensor(x, device=device)
        return x.to(device)
    
    @beartype
    def _normalize_distribution(self, x: np.ndarray) -> np.ndarray:
        """Normalize the distribution to have zero mean and unit variance.

        Args:
            x (np.ndarray): Input distribution.

        Returns:
            np.ndarray: Normalized distribution.
        """
        if self.normalize:
            mean = np.mean(x, axis=0, keepdims=True)
            std = np.std(x, axis=0, keepdims=True)
            return (x - mean) / (std + 1e-8)
        return x
    
    @beartype
    def _get_distribution_weights(self, x: np.ndarray) -> np.ndarray:
        """Get uniform weights for the distribution.

        Args:
            x (np.ndarray): Input distribution.

        Returns:
            np.ndarray: Uniform weights.
        """
        n_samples = len(x)
        return np.ones(n_samples) / n_samples
    
    @beartype
    def compute_transport_plan(
        self,
        source: Union[Tensor, np.ndarray],
        target: Union[Tensor, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute optimal transport plan between source and target distributions using POT.

        Args:
            source (Union[Tensor, np.ndarray]): Source distribution.
            target (Union[Tensor, np.ndarray]): Target distribution.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - transport_plan: Optimal transport plan matrix
                - source_weights: Weights for source distribution
                - target_weights: Weights for target distribution
        """
        # Convert to numpy
        source_np = self._to_numpy(source)
        target_np = self._to_numpy(target)
        
        # Ensure equal number of samples by subsampling the larger set
        min_samples = min(len(source_np), len(target_np))
        if len(source_np) > min_samples:
            indices = np.random.choice(len(source_np), min_samples, replace=False)
            source_np = source_np[indices]
        if len(target_np) > min_samples:
            indices = np.random.choice(len(target_np), min_samples, replace=False)
            target_np = target_np[indices]
        
        # Normalize distributions if requested
        if self.normalize:
            source_np = self._normalize_distribution(source_np)
            target_np = self._normalize_distribution(target_np)
        
        # Get distribution weights
        source_weights = self._get_distribution_weights(source_np)
        target_weights = self._get_distribution_weights(target_np)
        
        # Compute cost matrix using POT
        M = ot.dist(source_np, target_np, metric=self.metric)
        
        # Compute optimal transport plan using POT's Earth Mover's Distance (EMD)
        transport_plan = ot.emd(source_weights, target_weights, M)
        
        return transport_plan, source_weights, target_weights
    
    @beartype
    def compute_conditional_transport_plan(
        self,
        source: Union[Tensor, np.ndarray],
        target: Union[Tensor, np.ndarray],
        source_condition: Union[Tensor, np.ndarray],
        target_condition: Union[Tensor, np.ndarray],
        kernel_bandwidth: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute conditional optimal transport plan using kernel distances on conditions.

        Args:
            source (Union[Tensor, np.ndarray]): Source distribution.
            target (Union[Tensor, np.ndarray]): Target distribution.
            source_condition (Union[Tensor, np.ndarray]): Condition values for source.
            target_condition (Union[Tensor, np.ndarray]): Condition values for target.
            kernel_bandwidth (float, optional): Bandwidth for the kernel. Defaults to 1.0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - transport_plan: Optimal transport plan matrix
                - source_weights: Weights for source distribution
                - target_weights: Weights for target distribution
        """
        # Convert to numpy
        source_np = self._to_numpy(source)
        target_np = self._to_numpy(target)
        source_cond_np = self._to_numpy(source_condition)
        target_cond_np = self._to_numpy(target_condition)
        
        # Ensure equal number of samples by subsampling the larger set
        min_samples = min(len(source_np), len(target_np))
        if len(source_np) > min_samples:
            indices = np.random.choice(len(source_np), min_samples, replace=False)
            source_np = source_np[indices]
            source_cond_np = source_cond_np[indices]
        if len(target_np) > min_samples:
            indices = np.random.choice(len(target_np), min_samples, replace=False)
            target_np = target_np[indices]
            target_cond_np = target_cond_np[indices]
        
        # Normalize distributions if requested
        if self.normalize:
            source_np = self._normalize_distribution(source_np)
            target_np = self._normalize_distribution(target_np)
            source_cond_np = self._normalize_distribution(source_cond_np)
            target_cond_np = self._normalize_distribution(target_cond_np)
        
        # Get distribution weights
        source_weights = self._get_distribution_weights(source_np)
        target_weights = self._get_distribution_weights(target_np)
        
        # Compute spatial cost matrix using POT
        M_spatial = ot.dist(source_np, target_np, metric=self.metric)
        
        # Compute condition cost matrix using kernel distance
        M_condition = self._compute_kernel_distance(source_cond_np, target_cond_np, kernel_bandwidth)
        
        # Combine spatial and condition costs
        M_combined = M_spatial + M_condition
        
        # Compute optimal transport plan using POT's Earth Mover's Distance (EMD)
        transport_plan = ot.emd(source_weights, target_weights, M_combined)
        
        return transport_plan, source_weights, target_weights
    
    @beartype
    def _compute_kernel_distance(
        self,
        source_condition: np.ndarray,
        target_condition: np.ndarray,
        bandwidth: float = 1.0,
    ) -> np.ndarray:
        """Compute kernel distance matrix between source and target conditions.

        Args:
            source_condition (np.ndarray): Condition values for source.
            target_condition (np.ndarray): Condition values for target.
            bandwidth (float, optional): Bandwidth for the kernel. Defaults to 1.0.

        Returns:
            np.ndarray: Kernel distance matrix.
        """
        # Compute pairwise squared Euclidean distances
        n_source = len(source_condition)
        n_target = len(target_condition)
        
        # Reshape for broadcasting
        source_expanded = source_condition.reshape(n_source, 1, -1)
        target_expanded = target_condition.reshape(1, n_target, -1)
        
        # Compute squared differences
        squared_diff = np.sum((source_expanded - target_expanded) ** 2, axis=2)
        
        # Apply Gaussian kernel
        kernel_matrix = np.exp(-squared_diff / (2 * bandwidth ** 2))
        
        # Convert to distance (1 - kernel)
        distance_matrix = 1 - kernel_matrix
        
        return distance_matrix
    
    @beartype
    def get_matched_pairs(
        self,
        source: Union[Tensor, np.ndarray],
        target: Union[Tensor, np.ndarray],
        return_tensors: bool = True,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray]]:
        """Get matched pairs between source and target distributions.

        Args:
            source (Union[Tensor, np.ndarray]): Source distribution.
            target (Union[Tensor, np.ndarray]): Target distribution.
            return_tensors (bool, optional): Whether to return PyTorch tensors. Defaults to True.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray]]: 
                - matched_source: Source points in matched pairs
                - matched_target: Corresponding target points
        """
        # Convert to numpy
        source_np = self._to_numpy(source)
        target_np = self._to_numpy(target)
        
        # Compute transport plan
        transport_plan, _, _ = self.compute_transport_plan(source_np, target_np)
        
        # Extract matched pairs from transport plan
        # For each source point, find the target point with highest transport weight
        row_ind, col_ind = np.where(transport_plan > 0)
        
        # Get matched pairs
        matched_source = source_np[row_ind]
        matched_target = target_np[col_ind]
        
        if return_tensors:
            matched_source = self._to_tensor(matched_source, self.device)
            matched_target = self._to_tensor(matched_target, self.device)
        
        return matched_source, matched_target
    
    @beartype
    def get_conditional_matched_pairs(
        self,
        source: Union[Tensor, np.ndarray],
        target: Union[Tensor, np.ndarray],
        source_condition: Union[Tensor, np.ndarray],
        target_condition: Union[Tensor, np.ndarray],
        kernel_bandwidth: float = 1.0,
        return_tensors: bool = True,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray]]:
        """Get matched pairs between source and target distributions based on conditions.

        Args:
            source (Union[Tensor, np.ndarray]): Source distribution.
            target (Union[Tensor, np.ndarray]): Target distribution.
            source_condition (Union[Tensor, np.ndarray]): Condition values for source.
            target_condition (Union[Tensor, np.ndarray]): Condition values for target.
            kernel_bandwidth (float, optional): Bandwidth for the kernel. Defaults to 1.0.
            return_tensors (bool, optional): Whether to return PyTorch tensors. Defaults to True.

        Returns:
            Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray]]: 
                - matched_source: Source points in matched pairs
                - matched_target: Corresponding target points
        """
        # Convert to numpy
        source_np = self._to_numpy(source)
        target_np = self._to_numpy(target)
        source_cond_np = self._to_numpy(source_condition)
        target_cond_np = self._to_numpy(target_condition)
        
        # Compute conditional transport plan
        transport_plan, _, _ = self.compute_conditional_transport_plan(
            source_np, target_np, source_cond_np, target_cond_np, kernel_bandwidth
        )
        
        # Extract matched pairs from transport plan
        row_ind, col_ind = np.where(transport_plan > 0)
        
        # Get matched pairs
        matched_source = source_np[row_ind]
        matched_target = target_np[col_ind]
        
        if return_tensors:
            matched_source = self._to_tensor(matched_source, self.device)
            matched_target = self._to_tensor(matched_target, self.device)
        
        return matched_source, matched_target


# Example usage
if __name__ == "__main__":
    # Create some example distributions
    source = torch.randn(10, 2)  # 2D Gaussian
    target = torch.randn(10, 2) + torch.tensor([2.0, 2.0])  # Shifted 2D Gaussian
    
    # Create some example conditions
    source_condition = torch.randn(10, 1)  # 1D condition for source
    target_condition = torch.randn(10, 1)  # 1D condition for target
    
    # Initialize OT planner
    ot_planner = OptimalTransportPlanner(normalize=True)
    
    # Get matched pairs
    matched_source, matched_target = ot_planner.get_matched_pairs(source, target)
    print(f"Matched pairs shape: {matched_source.shape}, {matched_target.shape}")
    
    # Get conditional matched pairs
    cond_matched_source, cond_matched_target = ot_planner.get_conditional_matched_pairs(
        source, target, source_condition, target_condition
    )
    print(f"Conditional matched pairs shape: {cond_matched_source.shape}, {cond_matched_target.shape}") 