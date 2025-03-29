import torch
from typing import Dict, Optional
from beartype import beartype
from torch import Tensor
from .base_trainer import BaseTrainer
from torch.utils.data import DataLoader
import torch.nn as nn
from ..methods.base_cfm import BaseCFM
from torch.nn.utils import clip_grad_norm_

class GuiderTrainer(BaseTrainer):
    """Trainer class for Classifier-Free Guidance Conditional Flow Matching models.
    
    This trainer implements training for models that support both conditional
    and unconditional generation through an is_conditional flag. During training,
    each batch is processed twice:
    1. Once with the true conditional information
    2. Once with the conditional information zeroed out
    """
    
    @beartype
    def single_batch(self, batch: Dict[str, Tensor], is_training: bool = True) -> float:
        """Process a single batch through the model with both conditional and unconditional paths.

        Args:
            batch (Dict[str, Tensor]): Batch dictionary containing 'x0', 'x1', and 'z'.
            is_training (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            float: Combined loss value for this batch.
        """
        # Process batch data
        x0, x1 = batch['x0'].to(self.device), batch['x1'].to(self.device)
        z0, z1 = batch.get('z0'), batch.get('z1')
        if z0 is not None:
            z0 = z0.to(self.device)
        if z1 is not None:
            z1 = z1.to(self.device)

        batch_size = x0.size(0)
        t = torch.rand(batch_size, 1, device=self.device)

        # Apply batch transform
        x0, x1, t, z0, z1 = self.cfm.batch_transform(x0, x1, t, z0, z1)
        
        # Compute intermediate state
        xt = self.cfm.compute_xt(x0, x1, t)
        
        # Compute target vector field
        v_target = self.cfm.compute_vector_field(x0, x1, t)

        if is_training:
            self.optimizer.zero_grad()
            total_loss = 0.0
            
            v_pred_cond = self.model(xt, t, z0, z1, is_conditional=True)
            loss_cond = self.cfm.loss_fn(
                v_pred_cond, 
                v_target
            )
            v_pred_uncond = self.model(xt, t, z0, z1, is_conditional=False)
            loss_uncond = self.cfm.loss_fn(
                v_pred_uncond, 
                v_target
            )
            total_loss = loss_cond + loss_uncond

            # Backward pass
            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            return total_loss.item()
        
        else:
            # During validation, only use conditional path
            v_pred_cond = self.model(xt, t, z0, z1, is_conditional=True)
            loss_cond = self.cfm.loss_fn(v_pred_cond, v_target)
            v_pred_uncond = self.model(xt, t, z0, z1, is_conditional=False)
            loss_uncond = self.cfm.loss_fn(v_pred_uncond, v_target)
            total_loss = loss_cond + loss_uncond
            return total_loss.item()