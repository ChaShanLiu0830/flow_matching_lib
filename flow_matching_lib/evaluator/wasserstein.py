import torch
import numpy as np
from typing import Union, Optional, Tuple
from beartype import beartype
from torch import Tensor
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import unittest
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class WassersteinEvaluator:
    """Evaluator for computing Wasserstein distance between distributions.
    
    This class provides methods to compute the Wasserstein distance between
    real and generated distributions using the Hungarian algorithm for
    optimal assignment.
    """
    
    @beartype
    def __init__(
        self,
        metric: str = "euclidean",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalize: bool = True,
    ):
        """Initialize the Wasserstein evaluator.

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
    def compute_distance(
        self,
        real_dist: Union[Tensor, np.ndarray],
        gen_dist: Union[Tensor, np.ndarray],
        return_assignment: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """Compute Wasserstein distance between real and generated distributions.

        Args:
            real_dist (Union[Tensor, np.ndarray]): Real data distribution.
            gen_dist (Union[Tensor, np.ndarray]): Generated data distribution.
            return_assignment (bool, optional): Whether to return optimal assignment. Defaults to False.

        Returns:
            Union[float, Tuple[float, np.ndarray]]: 
                - If return_assignment=False: Wasserstein distance
                - If return_assignment=True: (Wasserstein distance, assignment indices)
        """
        # Convert to numpy
        real_dist = self._to_numpy(real_dist)
        gen_dist = self._to_numpy(gen_dist)
        
        # Ensure equal number of samples by subsampling the larger set
        min_samples = min(len(real_dist), len(gen_dist))
        if len(real_dist) > min_samples:
            indices = np.random.choice(len(real_dist), min_samples, replace=False)
            real_dist = real_dist[indices]
        if len(gen_dist) > min_samples:
            indices = np.random.choice(len(gen_dist), min_samples, replace=False)
            gen_dist = gen_dist[indices]
        
        # Normalize distributions if requested
        real_dist = self._normalize_distribution(real_dist)
        gen_dist = self._normalize_distribution(gen_dist)
        
        # Compute cost matrix
        cost_matrix = cdist(real_dist, gen_dist, metric=self.metric)
        
        # Solve the assignment problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute Wasserstein distance as the sum of distances between matched pairs
        w_distance = cost_matrix[row_ind, col_ind].sum() / len(row_ind)
        
        if return_assignment:
            return w_distance, (row_ind, col_ind)
        return w_distance
    
    @beartype
    def compute_sliced_wasserstein(
        self,
        real_dist: Union[Tensor, np.ndarray],
        gen_dist: Union[Tensor, np.ndarray],
        num_projections: int = 50,
    ) -> float:
        """Compute Sliced Wasserstein distance between distributions.
        
        This is a more efficient approximation for high-dimensional data.

        Args:
            real_dist (Union[Tensor, np.ndarray]): Real data distribution.
            gen_dist (Union[Tensor, np.ndarray]): Generated data distribution.
            num_projections (int, optional): Number of random projections. Defaults to 50.

        Returns:
            float: Sliced Wasserstein distance.
        """
        # Convert to numpy
        real_dist = self._to_numpy(real_dist)
        gen_dist = self._to_numpy(gen_dist)
        
        # Normalize distributions if requested
        real_dist = self._normalize_distribution(real_dist)
        gen_dist = self._normalize_distribution(gen_dist)
        
        # Get dimensions
        dim = real_dist.shape[1]
        
        # Initialize distance
        sw_distance = 0.0
        
        # Generate random projections
        for _ in range(num_projections):
            # Generate a random unit vector
            projection = np.random.randn(dim)
            projection = projection / np.sqrt(np.sum(projection**2))
            
            # Project the data
            real_proj = np.dot(real_dist, projection)
            gen_proj = np.dot(gen_dist, projection)
            
            # Sort the projections
            real_proj = np.sort(real_proj)
            gen_proj = np.sort(gen_proj)
            
            # Compute 1D Wasserstein distance (which is just the L1 distance between sorted projections)
            sw_distance += np.mean(np.abs(real_proj - gen_proj))
        
        return sw_distance / num_projections
    
    @beartype
    def compute_batch_distance(
        self,
        real_loader: torch.utils.data.DataLoader,
        gen_loader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None,
        use_sliced: bool = True,
    ) -> float:
        """Compute Wasserstein distance between entire datasets processed in batches.

        This method collects all samples from both dataloaders and then computes
        the Wasserstein distance on the combined data, while processing in batches
        to avoid memory issues.

        Args:
            real_loader (DataLoader): DataLoader for real distribution.
            gen_loader (DataLoader): DataLoader for generated distribution.
            max_samples (Optional[int], optional): Maximum number of samples to use. Defaults to None.
            use_sliced (bool, optional): Whether to use sliced Wasserstein distance. Defaults to True.

        Returns:
            float: Wasserstein distance between the entire datasets.
        """
        # Collect all real samples
        all_real_samples = []
        sample_count = 0
        
        for batch in tqdm(real_loader, desc="Collecting real samples"):
            if isinstance(batch, dict):
                real_data = batch['x0']
            else:
                real_data = batch
                
            all_real_samples.append(self._to_numpy(real_data))
            sample_count += len(real_data)
            
            if max_samples is not None and sample_count >= max_samples:
                break
        
        # Collect all generated samples
        all_gen_samples = []
        sample_count = 0
        
        for batch in tqdm(gen_loader, desc="Collecting generated samples"):
            if isinstance(batch, dict):
                gen_data = batch['x0']
            else:
                gen_data = batch
                
            all_gen_samples.append(self._to_numpy(gen_data))
            sample_count += len(gen_data)
            
            if max_samples is not None and sample_count >= max_samples:
                break
        
        # Combine all samples
        all_real_samples = np.concatenate(all_real_samples, axis=0)
        all_gen_samples = np.concatenate(all_gen_samples, axis=0)
        
        # Apply max_samples limit if needed
        if max_samples is not None:
            all_real_samples = all_real_samples[:max_samples]
            all_gen_samples = all_gen_samples[:max_samples]
        
        # Compute distance on the entire dataset
        if use_sliced:
            return self.compute_sliced_wasserstein(all_real_samples, all_gen_samples)
        else:
            return self.compute_distance(all_real_samples, all_gen_samples)


# # Unit tests for WassersteinEvaluator
# class TestWassersteinEvaluator(unittest.TestCase):
#     def setUp(self):
#         # Create evaluator
#         self.evaluator = WassersteinEvaluator(normalize=True)
        
#         # Create test data
#         np.random.seed(42)
#         torch.manual_seed(42)
        
#         # Create two different distributions
#         self.dist1 = torch.randn(1000, 2)  # Standard normal
#         self.dist2 = torch.randn(1000, 2) + torch.tensor([2.0, 2.0])  # Shifted normal
        
#         # Create datasets and dataloaders
#         self.dataset1 = TensorDataset(self.dist1)
#         self.dataset2 = TensorDataset(self.dist2)
        
#         self.dataloader1 = DataLoader(self.dataset1, batch_size=100)
#         self.dataloader2 = DataLoader(self.dataset2, batch_size=100)
        
#         # Create dictionary-based datasets
#         self.dict_dataset1 = [({"x0": x}) for x in self.dist1.split(100)]
#         self.dict_dataset2 = [({"x0": x}) for x in self.dist2.split(100)]
        
#         self.dict_dataloader1 = DataLoader(self.dict_dataset1, batch_size=1)
#         self.dict_dataloader2 = DataLoader(self.dict_dataset2, batch_size=1)
    
#     def test_compute_distance(self):
#         # Test basic distance computation
#         distance = self.evaluator.compute_distance(self.dist1, self.dist2)
        
#         # Distance should be positive
#         self.assertGreater(distance, 0)
        
#         # Distance to self should be less than distance to other distribution
#         self_distance = self.evaluator.compute_distance(self.dist1, self.dist1)
#         self.assertLess(self_distance, distance)
    
#     def test_compute_sliced_wasserstein(self):
#         # Test sliced Wasserstein distance
#         distance = self.evaluator.compute_sliced_wasserstein(self.dist1, self.dist2)
        
#         # Distance should be positive
#         self.assertGreater(distance, 0)
        
#         # Distance to self should be less than distance to other distribution
#         self_distance = self.evaluator.compute_sliced_wasserstein(self.dist1, self.dist1)
#         self.assertLess(self_distance, distance)
    
#     def test_compute_batch_distance(self):
#         # Test batch distance computation
#         distance = self.evaluator.compute_batch_distance(
#             self.dataloader1, 
#             self.dataloader2,
#             use_sliced=True
#         )
        
#         # Distance should be positive
#         self.assertGreater(distance, 0)
        
#         # Distance to self should be less than distance to other distribution
#         self_distance = self.evaluator.compute_batch_distance(
#             self.dataloader1, 
#             self.dataloader1,
#             use_sliced=True
#         )
#         self.assertLess(self_distance, distance)
        
#         # Test with max_samples
#         distance_limited = self.evaluator.compute_batch_distance(
#             self.dataloader1, 
#             self.dataloader2,
#             max_samples=500,
#             use_sliced=True
#         )
        
#         # Should still be positive
#         self.assertGreater(distance_limited, 0)
        
#         # Test with dictionary-based dataloaders
#         dict_distance = self.evaluator.compute_batch_distance(
#             self.dict_dataloader1, 
#             self.dict_dataloader2,
#             use_sliced=True
#         )
        
#         # Should be positive
#         self.assertGreater(dict_distance, 0)
        
#         # Test with exact Wasserstein (not sliced)
#         exact_distance = self.evaluator.compute_batch_distance(
#             self.dataloader1, 
#             self.dataloader2,
#             max_samples=200,  # Limit samples for faster test
#             use_sliced=False
#         )
        
#         # Should be positive
#         self.assertGreater(exact_distance, 0)
    
#     def test_consistency(self):
#         """Test that batch computation gives similar results to direct computation."""
#         # Compute distance directly
#         direct_distance = self.evaluator.compute_sliced_wasserstein(self.dist1, self.dist2)
        
#         # Compute distance using batch method
#         batch_distance = self.evaluator.compute_batch_distance(
#             self.dataloader1, 
#             self.dataloader2,
#             use_sliced=True
#         )
        
#         # Distances should be similar (within 10%)
#         percent_diff = abs(direct_distance - batch_distance) / direct_distance
#         self.assertLess(percent_diff, 0.1)


# Example usage and test runner
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    # Example usage
    print("Example usage of WassersteinEvaluator:")
    real_dist = torch.randn(10000, 2)  # 2D Gaussian
    gen_dist = torch.randn(10000, 2)   # Shifted 2D Gaussian
    
    class TestData(TensorDataset):
        def __init__(self, data):
            super().__init__(data) 
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
    test_data = TestData(real_dist)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True)
    
    test_data2 = TestData(gen_dist)
    test_loader2 = DataLoader(test_data2, batch_size=100, shuffle=True)
    
    evaluator = WassersteinEvaluator(normalize=True)
    
    distance = evaluator.compute_distance(real_dist, gen_dist)
    print(f"Wasserstein distance: {distance:.4f}")
    
    batch_distance = evaluator.compute_batch_distance(test_loader, test_loader2, use_sliced=False)
    print(f"Batch Wasserstein distance: {batch_distance:.4f}")
    
    sliced_distance = evaluator.compute_sliced_wasserstein(real_dist, gen_dist)
    print(f"Sliced Wasserstein distance: {sliced_distance:.4f}")
    
    # Run unit tests
    print("\nRunning unit tests:")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    