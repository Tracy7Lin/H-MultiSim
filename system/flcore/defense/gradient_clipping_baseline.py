import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any


class GradientClippingBaseline:
    """Gradient Clipping Defense - Baseline Version
    
    This defense method clips model parameter updates to prevent large updates
    that could be caused by malicious clients. Only contains the core clipping
    functionality without attack detection.
    """
    
    def __init__(self, max_norm: float = 1.0):
        """
        Args:
            max_norm: Maximum norm for clipping (default: 1.0)
        """
        self.max_norm = max_norm
        
    def clip_model_update(self, model_update: nn.Module) -> nn.Module:
        """
        Clip the norm of model parameter updates.
        
        Args:
            model_update: Model containing parameter updates
            
        Returns:
            Model with clipped parameter updates
        """
        # Calculate total norm of all parameters
        total_norm = 0.0
        for param in model_update.parameters():
            if param is not None:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Calculate clipping coefficient
        clip_coef = self.max_norm / (total_norm + 1e-6)
        
        # Clip if necessary
        if clip_coef < 1:
            for param in model_update.parameters():
                if param is not None:
                    param.data.mul_(clip_coef)
        
        return model_update
    
    def clip_uploaded_models(self, global_model: nn.Module, uploaded_models: List[nn.Module]) -> List[nn.Module]:
        """
        Clip the parameter updates for all uploaded models.
        
        Args:
            global_model: The global model (reference point)
            uploaded_models: List of uploaded client models
            
        Returns:
            List of models with clipped updates
        """
        clipped_models = []
        
        for client_model in uploaded_models:
            # Calculate model update (difference from global model)
            model_update = copy.deepcopy(client_model)
            for global_param, update_param in zip(global_model.parameters(), model_update.parameters()):
                update_param.data.sub_(global_param.data)
            
            # Clip the update
            clipped_update = self.clip_model_update(model_update)
            
            # Add back to global model to get clipped client model
            clipped_model = copy.deepcopy(global_model)
            for global_param, clipped_param in zip(clipped_model.parameters(), clipped_update.parameters()):
                global_param.data.add_(clipped_param.data)
            
            clipped_models.append(clipped_model)
        
        return clipped_models
