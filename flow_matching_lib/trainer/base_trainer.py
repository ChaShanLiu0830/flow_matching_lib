import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from beartype import beartype
from torch import Tensor
from tqdm import tqdm
from ..methods.base import BaseCFM
from torch.nn.utils import clip_grad_norm_

class BaseTrainer:
    """Base trainer class for Conditional Flow Matching models."""
    
    @beartype
    def __init__(
        self,
        cfm: BaseCFM,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "checkpoints",
        model_name: str = "model",
        max_grad_norm: float = 1,
    ):
        """Initialize the base trainer.

        Args:
            cfm (BaseCFM): The CFM method (e.g., I-CFM, OT-CFM).
            model (nn.Module): The neural network model for vector field prediction.
            optimizer (Optimizer): The optimizer to use for training.
            train_loader (DataLoader): DataLoader for training data.
            valid_loader (Optional[DataLoader], optional): DataLoader for validation data.
            device (str, optional): Device to train on. Defaults to "cuda" if available.
            checkpoint_dir (str, optional): Directory to save checkpoints.
            model_name (str, optional): Name of the model for saving checkpoints. Defaults to "model".
            max_grad_norm (float, optional): Maximum norm for gradient clipping. Defaults to 1.
        """
        self.cfm = cfm
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.max_grad_norm = max_grad_norm
        
        # Create model-specific checkpoint directory
        self.model_checkpoint_dir = os.path.join(checkpoint_dir, model_name)
        os.makedirs(self.model_checkpoint_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'valid_loss': [],
            'current_epoch': 0,
            'best_valid_loss': float('inf')
        }

    @beartype
    def single_batch(self, batch: Dict[str, Tensor], is_training: bool = True) -> float:
        """Process a single batch through the model.

        Args:
            batch (Dict[str, Tensor]): Batch dictionary containing 'x0', 'x1', and optionally 'z'.
            is_training (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            float: Loss value for this batch.
        """
        # Process batch data
        x0, x1 = batch['x0'].to(self.device), batch['x1'].to(self.device)
        z = batch.get('z')
        if z is not None:
            z = z.to(self.device)

        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=self.device)

        # Apply batch transform
        x0, x1, t, z = self.cfm.batch_transform(x0, x1, t, z)
        
        xt = self.cfm.compute_xt(x0, x1, t, z)
        # Compute target vector field
        v_target = self.cfm.compute_vector_field(x0, x1, t, z)
        
        # Forward pass through the neural network
        v_pred = self.model(xt, t, z) if z is not None else self.model(xt, t)

        # Compute loss using CFM method
        loss = self.cfm.loss_fn(v_pred, v_target)

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            # Apply gradient clipping
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return loss.item()

    @beartype
    def train(self, num_epochs: int, save_frequency: int = 1) -> Dict[str, list]:
        """Train the model.

        Args:
            num_epochs (int): Number of epochs to train for.
            save_frequency (int, optional): How often to save checkpoints. Defaults to 1.

        Returns:
            Dict[str, list]: Training history.
        """
        start_epoch = self.history['current_epoch']
        pbar = tqdm(range(start_epoch, num_epochs), desc="Training Progress")
        
        for epoch in pbar:
            self.model.train()
            train_losses = []
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            
            for batch in train_pbar:
                loss = self.single_batch(batch, is_training=True)
                train_losses.append(loss)
                train_pbar.set_postfix({'train_loss': f'{loss:.6f}'})
            
            train_loss = sum(train_losses) / len(train_losses)
            
            # Validation phase
            valid_loss = None
            if self.valid_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    valid_pbar = tqdm(self.valid_loader, desc="Validating", leave=False)
                    valid_losses = []
                    for batch in valid_pbar:
                        loss = self.single_batch(batch, is_training=False)
                        valid_losses.append(loss)
                        valid_pbar.set_postfix({'valid_loss': f'{loss:.6f}'})
                valid_loss = sum(valid_losses) / len(valid_losses)

            # Update history
            self.history['train_loss'].append(train_loss)
            if valid_loss is not None:
                self.history['valid_loss'].append(valid_loss)

            # Update progress bar description
            desc = f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}"
            if valid_loss is not None:
                desc += f" - Valid Loss: {valid_loss:.6f}"
            pbar.set_description(desc)

            # Save checkpoint if needed
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(epoch + 1, train_loss, valid_loss)

            self.history['current_epoch'] = epoch + 1

        return self.history

    @beartype
    def save_checkpoint(self, epoch: int, train_loss: float, valid_loss: Optional[float] = None) -> None:
        """Save a checkpoint.

        Args:
            epoch (int): Current epoch number.
            train_loss (float): Current training loss.
            valid_loss (Optional[float], optional): Current validation loss. Defaults to None.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'history': self.history
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(
            self.model_checkpoint_dir, 
            f'{self.model_name}_latest.pt'
        )
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(
            self.model_checkpoint_dir, 
            f'{self.model_name}_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, epoch_path)
        
        # Save best model if validation loss improved
        if valid_loss is not None and valid_loss < self.history['best_valid_loss']:
            self.history['best_valid_loss'] = valid_loss
            best_path = os.path.join(
                self.model_checkpoint_dir, 
                f'{self.model_name}_best.pt'
            )
            torch.save(checkpoint, best_path)
            
            # Also save with the best validation loss value in the filename
            best_loss_path = os.path.join(
                self.model_checkpoint_dir, 
                f'{self.model_name}_best_{valid_loss:.6f}.pt'
            )
            torch.save(checkpoint, best_loss_path)

    @beartype
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, load_best: bool = False) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            checkpoint_path (Optional[str], optional): Path to the checkpoint file. 
                If None, will load latest or best based on load_best. Defaults to None.
            load_best (bool, optional): Whether to load the best checkpoint. 
                Only used if checkpoint_path is None. Defaults to False.

        Returns:
            Dict[str, Any]: Checkpoint information.

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        if checkpoint_path is None:
            # Determine which checkpoint to load
            filename = f'{self.model_name}_best.pt' if load_best else f'{self.model_name}_latest.pt'
            checkpoint_path = os.path.join(self.model_checkpoint_dir, filename)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['train_loss'],
            'valid_loss': checkpoint['valid_loss']
        } 