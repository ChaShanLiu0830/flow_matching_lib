import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from flow_matching_lib.methods.i_cfm import I_CFM
from flow_matching_lib.trainer.base_trainer import BaseTrainer
from flow_matching_lib.trainer.guide_trainer import GuiderTrainer
from flow_matching_lib.sampler.base_sampler import BaseSampler
from flow_matching_lib.sampler.guide_sampler import GuiderSampler
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, xt, t, z, is_conditional=True):
        if not is_conditional:
            z = torch.zeros_like(z)
        inputs = torch.cat([xt, t, z], dim=-1)
        return self.network(inputs)


class Gaussian8Dataset(Dataset):
    def __init__(self, num_samples=1000, std=0.1):
        self.num_samples = num_samples
        self.std = std
        self.centers = [
            (1, 0), (0.707, 0.707), (0, 1), (-0.707, 0.707),
            (-1, 0), (-0.707, -0.707), (0, -1), (0.707, -0.707)
        ]
        self.data, self.labels = self._generate_data()
        self.labels = self.labels
    def _generate_data(self):
        data = []
        labels = []
        for i in range(self.num_samples):
            label = np.random.randint(0, 8)
            center = self.centers[label]
            point = np.random.normal(loc=center, scale=self.std, size=2)
            data.append(point)
            labels.append(label)
        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64) + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"x0": self.data[idx], "label": self.labels[idx]}

class NormalGaussian(Dataset):
    def __init__(self, num_samples=1000, std=0.1):
        self.num_samples = num_samples  
        self.std = std
        self.data = np.random.normal(loc=0, scale=1, size=(num_samples, 2))
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return {"x0": self.data[idx]}   
    
class RandomCombinedDataset(Dataset):
    def __init__(self, x0_dataset, x1_dataset):
        self.x0_dataset = x0_dataset
        self.x1_dataset = x1_dataset

    def __len__(self):
        # Return the maximum length of the two datasets
        return max(len(self.x0_dataset), len(self.x1_dataset))

    def __getitem__(self, index):
        # Randomly sample an index for each dataset
        # x0_index = torch.randint(0, len(self.x0_dataset) - 1, (1,1))
        # x1_index = torch.randint(0, len(self.x1_dataset) - 1, (1,1))
        x0_index = index
        x1_index = index
        x0_data = torch.from_numpy(self.x0_dataset[x0_index]['x0']).to(torch.float32)
        x1_data, x1_label = torch.from_numpy(self.x1_dataset[x1_index]['x0']).to(torch.float32), torch.tensor([self.x1_dataset[x1_index]['label']]).to(torch.float32)
        return {'x0': x0_data, 'x1': x1_data, 'z':x1_label}


def plot_samples(real_data, gen_samples, save_path=None):
    plt.figure(figsize=(15, 5))
    
    # Plot real data
    plt.subplot(121)
    plt.scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s = 1)
    plt.title("Real Data")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Plot generated samples
    plt.subplot(122)
    plt.scatter(gen_samples[0, :, 0], gen_samples[0, :, 1], alpha=0.5, s = 1)
    plt.scatter(gen_samples[-1, :, 0], gen_samples[-1, :, 1], alpha=0.5, s = 1)
    plt.title("Generated Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main(args):
    # Set device
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = "cuda:0"
    eight_moon = Gaussian8Dataset(num_samples=args.num_samples, std=args.noise_scale)
    gaussian = NormalGaussian(num_samples=args.num_samples, std=args.noise_scale)
    dataset = RandomCombinedDataset(gaussian, eight_moon)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) 
    
    # Create dataloaders
    train_loader = dataloader
    test_loader = dataloader
    
    # Create model and CFM method
    model = MLPModel(
        input_dim=4,  # x_dim(2) + t_dim(1) + z_dim(1)
        hidden_dim=args.hidden_dim,
        output_dim=2
    ).to(device)
    
    cfm = I_CFM(sigma=args.sigma)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create trainer
    if args.use_guidance:
        trainer = GuiderTrainer(
            cfm=cfm,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=test_loader,
            device=device,
            model_name="gaussian8_guided",
        )
    else:
        trainer = BaseTrainer(
            cfm=cfm,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=test_loader,
            device=device,
            model_name="gaussian8"
        )
    
    # Train model
    # trainer.train(num_epochs=args.num_epochs, save_frequency= int(args.num_epochs/10))
    trainer.load_checkpoint('/home/evan_chen/flow_matching_lib/flow_matching_lib/test/checkpoints/gaussian8_guided/gaussian8_guided_best.pt')
    
    # Create sampler
    if args.use_guidance:
        sampler = GuiderSampler(
            cfm=cfm,
            model=model,
            device=device,
            guidance_weight=args.guidance_weight
        )
    else:
        sampler = BaseSampler(
            cfm=cfm,
            model=model,
            device=device
        )
    
    gen_loader = DataLoader(gaussian, batch_size=args.batch_size, shuffle=True)
    # Generate samples
    test_data = next(iter(gen_loader))
    samples, sample_trajectory = sampler.sample_trajectory(
        x=test_data['x0'].to(device).to(torch.float32),
        z = torch.randint(1, 9, (test_data['x0'].shape[0], 1)).to(device).to(torch.float32),
        # z = torch.zeros(test_data['x0'].shape[0], 1).to(device).to(torch.float32),
    )
    print(sample_trajectory.shape)
    # Plot results
    plot_samples(
        eight_moon.data,
        sample_trajectory.cpu().numpy(),
        save_path=args.save_path
    )
    
    # Plot training losses
    trainer.plot_losses(save_path=args.save_path.replace('.png', '_losses.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test 8-Gaussian Flow Matching model")
    
    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=8000, help="Number of samples in dataset")
    parser.add_argument("--noise_scale", type=float, default=0.1, help="Noise scale for Gaussian distributions")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension of MLP")
    parser.add_argument("--sigma", type=float, default=0.1, help="Sigma parameter for I-CFM")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # Guidance parameters
    parser.add_argument("--use_guidance", action="store_true", help="Whether to use classifier-free guidance")
    parser.add_argument("--guidance_weight", type=float, default=2.0, help="Guidance weight for sampling")
    
    # Sampling parameters
    parser.add_argument("--num_gen_samples", type=int, default=1000, help="Number of samples to generate")
    
    # Output parameters
    parser.add_argument("--save_path", type=str, default="./results/8moon_samples.png", help="Path to save results")
    
    args = parser.parse_args()
    main(args) 