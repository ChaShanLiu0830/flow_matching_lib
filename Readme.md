# Flow Matching Library

A modular and extensible library for Conditional Flow Matching (CFM) in deep generative modeling.

## Overview

Flow Matching Library provides a flexible framework for implementing and experimenting with various Conditional Flow Matching (CFM) methods. The library is designed with modularity and extensibility in mind, allowing users to:

- Mix and match different CFM methods (OT-CFM, I-CFM, COT-CFM)
- Use custom neural network architectures
- Implement custom training and sampling strategies
- Easily experiment with conditional generation

## Installation

### From Source

```bash
git clone https://github.com/yourusername/flow_matching_lib.git
cd flow_matching_lib
pip install -e .
```

### Requirements

- Python 3.6+
- PyTorch 1.7+
- torchdiffeq
- numpy
- matplotlib
- beartype (>=0.10.0)

## Package Structure

```
flow_matching_lib/
├── flow_matching_lib/
│   ├── __init__.py                
│   ├── methods/                   # CFM method implementations
│   │   ├── __init__.py            
│   │   ├── base_cfm.py            # Abstract base class for CFM methods
│   │   ├── i_cfm.py               # Independent CFM method
│   │   ├── ot_cfm.py              # Optimal Transport CFM method
│   │   └── cot_cfm.py             # Continuous Optimal Transport CFM method
│   ├── trainer/                  
│   │   ├── __init__.py            
│   │   ├── base_trainer.py        # Base trainer class
│   │   └── guide_trainer.py       # Guide-based trainer
│   ├── sampler/                  
│   │   ├── __init__.py            
│   │   ├── base_sampler.py        # Base sampler class
│   │   └── guide_sampler.py       # Guide-based sampler
│   ├── networks/                 # Neural network architectures
│   ├── utils/                    # Utility functions
│   └── evaluator/                # Evaluation metrics
├── examples/                     # Example usage scripts
├── setup.py                      # Package installation script
└── README.md                     # Documentation
```

## Core Components

### CFM Methods

The library offers several implementations of Conditional Flow Matching methods:

- **Base CFM**: Abstract base class that defines the common interface for all CFM methods
- **I-CFM**: Independent CFM method that uses simple linear interpolation
- **OT-CFM**: Optimal Transport CFM method that uses optimal transport paths
- **COT-CFM**: Continuous Optimal Transport CFM method

### Trainer Classes

- **BaseTrainer**: Core training class that handles the training loop, logging, and checkpointing
- **GuideTrainer**: Extended trainer that incorporates guide-based training

### Sampler Classes

- **BaseSampler**: Implements ODE-based sampling strategies for CFM models
- **GuideSampler**: Implements guide-based sampling strategies for conditional generation

## Usage Example

Below is a simple example demonstrating how to use the library to train a model and generate samples:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import the necessary components
from flow_matching_lib.methods.ot_cfm import OT_CFM
from flow_matching_lib.trainer.base_trainer import BaseTrainer
from flow_matching_lib.sampler.base_sampler import BaseSampler

# Define a neural network for the vector field
class VectorFieldNet(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, t, z0=None, z1=None):
        # Concatenate inputs
        if t.dim() == 1:
            t = t.unsqueeze(1)
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

# Generate synthetic data
def generate_toy_data(n_samples=1000):
    # Create a simple 2D dataset with two Gaussians
    x0 = torch.randn(n_samples, 2) * 0.5 - torch.tensor([2.0, 0.0])
    x1 = torch.randn(n_samples, 2) * 0.5 + torch.tensor([2.0, 0.0])
    
    # Create labels (0 for x0, 1 for x1)
    y0 = torch.zeros(n_samples, 1)
    y1 = torch.ones(n_samples, 1)
    
    # Combine them
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)
    
    # Shuffle the data
    perm = torch.randperm(x.size(0))
    x, y = x[perm], y[perm]
    
    return x, y

# Main training and sampling code
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate data
    x, y = generate_toy_data(n_samples=1000)
    
    # Create dataset and dataloader
    dataset = TensorDataset(x, x, y, y)  # (x0, x1, z0, z1)
    dataloader = DataLoader(
        dataset, 
        batch_size=64, 
        shuffle=True,
        collate_fn=lambda batch: {
            'x0': torch.stack([item[0] for item in batch]),
            'x1': torch.stack([item[1] for item in batch]),
            'z0': torch.stack([item[2] for item in batch]),
            'z1': torch.stack([item[3] for item in batch])
        }
    )
    
    # Create model
    model = VectorFieldNet(data_dim=2, hidden_dim=128).to(device)
    
    # Create CFM method
    cfm_method = OT_CFM(sigma=0.1)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create trainer
    trainer = BaseTrainer(
        cfm=cfm_method,
        model=model,
        optimizer=optimizer,
        train_loader=dataloader,
        device=device,
        checkpoint_dir="checkpoints",
        model_name="ot_cfm_model"
    )
    
    # Train the model
    trainer.train(num_epochs=100)
    
    # Create sampler
    sampler = BaseSampler(
        cfm=cfm_method,
        model=model,
        device=device
    )
    
    # Generate samples
    x_init = torch.randn(100, 2).to(device)  # Initial noise
    z_cond = torch.ones(100, 1).to(device)   # Conditioning on class 1
    
    samples, trajectory = sampler.sample_trajectory(
        x=x_init,
        start_t=0.0,
        end_t=1.0,
        n_points=100,
        z0=z_cond,
        z1=z_cond
    )
    
    print(f"Generated {len(samples)} samples conditioned on class 1")

if __name__ == "__main__":
    main()
```

## Advanced Usage

### Custom CFM Methods

You can create custom CFM methods by subclassing the `BaseCFM` class:

```python
from flow_matching_lib.methods.base_cfm import BaseCFM
import torch
from torch import Tensor
from beartype import beartype
from beartype.typing import Optional

class CustomCFM(BaseCFM):
    @beartype
    def __init__(self, sigma: float = 1.0, custom_param: float = 0.5):
        super().__init__(sigma)
        self.custom_param = custom_param
    
    @beartype
    def compute_vector_field(
        self, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        z: Optional[Tensor] = None
    ) -> Tensor:
        # Implement your custom vector field computation
        return (x1 - x0) * self.custom_param
    
    @beartype
    def compute_xt(
        self, 
        x0: Tensor, 
        x1: Tensor, 
        t: Tensor, 
        z: Optional[Tensor] = None
    ) -> Tensor:
        # Implement your custom interpolation scheme
        return (1 - t) * x0 + t * x1
    
    def loss_fn(self, v_nn: Tensor, u_t: Tensor, z: Optional[Tensor] = None) -> Tensor:
        # Implement your custom loss function
        return ((v_nn - u_t) ** 2).mean()
```

### Custom Training

For custom training loops, subclass the `BaseTrainer` class:

```python
from flow_matching_lib.trainer.base_trainer import BaseTrainer
import torch

class CustomTrainer(BaseTrainer):
    def single_batch(self, batch, is_training=True):
        # Implement your custom batch processing logic
        x0, x1 = batch['x0'].to(self.device), batch['x1'].to(self.device)
        z0, z1 = batch.get('z0'), batch.get('z1')
        
        # Custom logic here...
        
        return loss.item()
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.