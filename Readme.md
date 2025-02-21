Below is an updated design that cleanly separates the training and inference procedures into classes. In this design, the user can mix and match different neural network backbones with various CFM methods (e.g. Optimal Transport, Schrödinger Bridge) and plug in alternative training or sampling strategies by subclassing the base classes.

Below is the proposed package structure followed by code for each module.

---

## Package Structure Outline

```
flow_matching_lib/
├── flow_matching_lib/
│   ├── __init__.py                # Re-exports key classes and functions.
│   ├── methods/                   # CFM method classes (e.g., Optimal Transport, Schrödinger Bridge).
│   │   ├── __init__.py            
│   │   ├── base.py                # Abstract base class for CFM methods.
│   │   ├── OB_CFM.py   # Optimal Transport–based CFM method.
│   │   └── SB_CFM.py          # Schrödinger Bridge–based CFM method.
│   ├── trainer/                  
│   │   ├── __init__.py            
│   │   └── trainer.py             # BaseTrainer class for training.
│   ├── sampler/                  
│   │   ├── __init__.py            
│   │   └── sampler.py             # BaseSampler class for inference.
│   └── utils/                   # Utility functions (e.g., synthetic dataset generation).
│   │   ├── __init__.py            
│   │   ├── ot_plan.py             # Optimal Transport plan class.
│   │   └── utils.py             # Utility functions (e.g., synthetic dataset generation).
├── examples/
│   └── train_and_sample.py        # Example script that trains and samples using the classes.
├── setup.py                       # Package installation script.
└── README.md                      # Documentation.
```

---

## Code Details

### 1. Neural Network Backbone

_File: `cfm/networks/feedforward.py`_

A simple feedforward network that computes a base vector field given the data xx, time tt, and an optional condition cc.

```python
# cfm/networks/feedforward.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(nn.Module):
    def __init__(self, x_dim=2, cond_dim=0, hidden_dim=128):
        """
        Parameters:
            x_dim (int): Dimensionality of the data.
            cond_dim (int): Dimensionality of the conditioning variable (0 if none).
            hidden_dim (int): Number of hidden units.
        """
        super(FeedForwardNN, self).__init__()
        input_dim = x_dim + 1 + cond_dim  # Concatenate x, time t, and condition.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, x_dim)
    
    def forward(self, x, t, cond=None):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if cond is not None:
            h = torch.cat([x, t, cond], dim=1)
        else:
            h = torch.cat([x, t], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        output = self.fc3(h)
        return output
```

_File: `cfm/networks/__init__.py`_

```python
# cfm/networks/__init__.py
from .feedforward import FeedForwardNN

__all__ = ["FeedForwardNN"]
```

---

### 2. Base CFM Method and Its Variants

_File: `cfm/methods/base.py`_

An abstract base class for CFM methods. It accepts any neural network model as the backbone and requires subclasses to implement a method to modify the vector field.

```python
# cfm/methods/base.py
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseCFM(nn.Module, ABC):
    def __init__(self, nn_model: nn.Module):
        """
        Parameters:
            nn_model (nn.Module): The neural network model to compute the base vector field.
        """
        super(BaseCFM, self).__init__()
        self.nn_model = nn_model

    @abstractmethod
    def vector_field(self, v, x, t, cond=None):
        pass

    def forward(self, x, t, cond=None):
        v_nn = self.nn_model(x, t, cond)
        return self.vector_field(v_nn, x, t, cond)
    def probability(self, x0, x1, t, cond=None):
        pass
    def batch_operator(self, x0, x1, t, cond=None):
        pass
```

---

### 3. Trainer Class

_File: `cfm/trainer.py`_

A base trainer class (BaseTrainer) that implements a generic training loop. Different training methods can be implemented by subclassing this.

```python
# cfm/trainer.py
import torch
from abc import ABC

class BaseTrainer(ABC):
    def __init__(self, model, data_loader, optimizer, device, num_epochs=100):
        """
        Base trainer for CFM methods.
        
        Parameters:
            model (nn.Module): The CFM method (e.g., OptimalTransportCFM, SchrodingerCFM).
            data_loader (DataLoader): PyTorch DataLoader providing training batches.
            optimizer (torch.optim.Optimizer): Optimizer.
            device (torch.device): Device for training.
            num_epochs (int): Number of training epochs.
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for batch in self.data_loader:
                x_data = batch['x'].to(self.device)
                cond = batch.get('cond')
                if cond is not None:
                    cond = cond.to(self.device)
                batch_size = x_data.size(0)
                noise = torch.randn_like(x_data).to(self.device)
                t = torch.rand(batch_size, 1).to(self.device)  # Random t in [0,1].
                
                # Interpolate between noise and data.
                x_t = (1 - t) * noise + t * x_data
                
                # Ground truth velocity.
                v_target = x_data - noise
                
                # Model prediction.
                v_pred = self.model(x_t, t, cond)
                
                loss = ((v_pred - v_target) ** 2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss/len(self.data_loader):.6f}")
```

---

### 4. Sampler Class

_File: `cfm/sampler.py`_

A base sampler class (BaseSampler) that implements a default Euler integration procedure for inference. Different sampling methods can be implemented via subclassing.

```python
# cfm/sampler.py
import torch
from abc import ABC

class BaseSampler(ABC):
    def __init__(self, model, device, steps=100):
        """
        Base sampler for CFM methods.
        
        Parameters:
            model (nn.Module): Trained CFM method.
            device (torch.device): Device for inference.
            steps (int): Number of integration steps.
        """
        self.model = model
        self.device = device
        self.steps = steps

    def sample(self, num_samples, cond=None):
        self.model.eval()
        with torch.no_grad():
            # Assume the NN backbone has an fc3 layer that defines the output dimension.
            x_dim = self.model.nn_model.fc3.out_features  
            x = torch.randn(num_samples, x_dim).to(self.device)
            if cond is not None:
                cond = cond.to(self.device)
                if cond.size(0) == 1 and num_samples > 1:
                    cond = cond.repeat(num_samples, 1)
            dt = 1.0 / self.steps
            for i in range(self.steps):
                t = torch.full((num_samples, 1), i * dt).to(self.device)
                v = self.model(x, t, cond)
                x = x + dt * v  # Euler integration step.
            return x.cpu()
```

---

### 5. Utility Functions

_File: `cfm/utils.py`_

A helper module for generating a synthetic dataset for testing.

```python
# cfm/utils.py
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_synthetic_data(num_samples=1000):
    """
    Generate a synthetic 2D dataset with two classes.
    
    - Class 0: centered at (-2, 0)
    - Class 1: centered at (2, 0)
    
    Returns:
        Tuple[Tensor, Tensor]: data tensor (num_samples, 2) and labels tensor (num_samples,)
    """
    data = []
    labels = []
    num_each = num_samples // 2
    for _ in range(num_each):
        x = np.random.randn(2) + np.array([-2, 0])
        data.append(x)
        labels.append(0)
    for _ in range(num_each):
        x = np.random.randn(2) + np.array([2, 0])
        data.append(x)
        labels.append(1)
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return data, labels

class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic data.
    
    Each sample is a dict with:
      - 'x': the data point (Tensor)
      - 'cond': the condition (e.g. class label as a float tensor)
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.float().unsqueeze(1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {'x': self.data[idx], 'cond': self.labels[idx]}
```

---

### 6. Package Initialization

_File: `cfm/__init__.py`_

Re-export key components for convenient access.

```python
# cfm/__init__.py
from .methods import OptimalTransportCFM, SchrodingerCFM
from .networks import FeedForwardNN
from .trainer import BaseTrainer
from .sampler import BaseSampler
from .utils import generate_synthetic_data, SyntheticDataset

__all__ = [
    "OptimalTransportCFM",
    "SchrodingerCFM",
    "FeedForwardNN",
    "BaseTrainer",
    "BaseSampler",
    "generate_synthetic_data",
    "SyntheticDataset"
]
```

---

### 7. Example Script

_File: `examples/train_and_sample.py`_

An example script demonstrating how to create a backbone network, wrap it with different CFM methods, train via the BaseTrainer class, and generate samples using the BaseSampler class.

```python
# examples/train_and_sample.py
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cfm import (
    FeedForwardNN,
    OptimalTransportCFM,
    SchrodingerCFM,
    BaseTrainer,
    BaseSampler,
    generate_synthetic_data,
    SyntheticDataset
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic dataset.
    data, labels = generate_synthetic_data(num_samples=1000)
    dataset = SyntheticDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # --- Setup Optimal Transport CFM Method ---
    nn_model_ot = FeedForwardNN(x_dim=2, cond_dim=1, hidden_dim=128)
    model_ot = OptimalTransportCFM(nn_model_ot).to(device)
    optimizer_ot = torch.optim.Adam(model_ot.parameters(), lr=1e-3)
    
    trainer_ot = BaseTrainer(model_ot, data_loader, optimizer_ot, device, num_epochs=50)
    print("Training Optimal Transport CFM Method...")
    trainer_ot.train()
    
    # --- Setup Schrödinger Bridge CFM Method ---
    nn_model_sch = FeedForwardNN(x_dim=2, cond_dim=1, hidden_dim=128)
    model_sch = SchrodingerCFM(nn_model_sch).to(device)
    optimizer_sch = torch.optim.Adam(model_sch.parameters(), lr=1e-3)
    
    trainer_sch = BaseTrainer(model_sch, data_loader, optimizer_sch, device, num_epochs=50)
    print("Training Schrödinger Bridge CFM Method...")
    trainer_sch.train()
    
    num_samples = 500
    cond_0 = torch.tensor([[0.0]])
    cond_1 = torch.tensor([[1.0]])
    
    sampler_ot = BaseSampler(model_ot, device, steps=100)
    sampler_sch = BaseSampler(model_sch, device, steps=100)
    
    # Generate samples for each method and condition.
    samples_ot_0 = sampler_ot.sample(num_samples, cond=cond_0)
    samples_ot_1 = sampler_ot.sample(num_samples, cond=cond_1)
    
    samples_sch_0 = sampler_sch.sample(num_samples, cond=cond_0)
    samples_sch_1 = sampler_sch.sample(num_samples, cond=cond_1)
    
    # Plot Optimal Transport CFM results.
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(samples_ot_0[:, 0], samples_ot_0[:, 1], color='blue', alpha=0.6, label='OT Class 0')
    plt.scatter(samples_ot_1[:, 0], samples_ot_1[:, 1], color='red', alpha=0.6, label='OT Class 1')
    plt.title("Optimal Transport CFM Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    # Plot Schrödinger Bridge CFM results.
    plt.subplot(1, 2, 2)
    plt.scatter(samples_sch_0[:, 0], samples_sch_0[:, 1], color='blue', alpha=0.6, label='Sch Class 0')
    plt.scatter(samples_sch_1[:, 0], samples_sch_1[:, 1], color='red', alpha=0.6, label='Sch Class 1')
    plt.title("Schrödinger Bridge CFM Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
```

---

### 8. Setup Script and README

_File: `setup.py`_

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='cfm_deep',
    version='0.3',
    description='Modular Conditional Flow Matching for Deep Generative Modeling',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'matplotlib'
    ],
)
```

_File: `README.md`_


# cfm_deep

A modular Conditional Flow Matching (CFM) framework for deep generative modeling.

This package decouples the neural network backbone from the CFM method so you can mix and match different NN architectures with various CFM formulations (e.g., Optimal Transport, Schrödinger Bridge). Additionally, the training and inference procedures are implemented as base classes (`BaseTrainer` and `BaseSampler`), allowing you to easily experiment with different training or sampling strategies.
## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/yourusername/cfm_deep.git
cd cfm_deep
pip install -e .
````

## Usage

To train the models and generate samples, run the example script:

```bash
python examples/train_and_sample.py
```

This script trains both the Optimal Transport and Schrödinger Bridge variants on a synthetic dataset and displays the generated samples.
