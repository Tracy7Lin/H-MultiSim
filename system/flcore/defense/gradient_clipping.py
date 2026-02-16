import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any

from .base_defense import BaseDefense
from .defense_utils import DefenseUtils


class GradientClipping(BaseDefense):
    """Gradient Clipping Defense
    
    This defense method clips gradients to prevent large gradient updates
    that could be caused by malicious clients.
    """
    
    def __init__(self, args, device, max_norm: float = 1.0, norm_type: float = 2.0):
        super(GradientClipping, self).__init__(args, device)
        self.max_norm = max_norm
        self.norm_type = norm_type
        
    def apply_defense(self, model, clients, **kwargs):
        """Apply gradient clipping defense"""
        print("Applying Gradient Clipping defense...")
        
        # Get gradients from all clients
        all_gradients = []
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    all_gradients.append(gradients)
        
        if not all_gradients:
            print("No gradients available for clipping")
            return model
        
        # Clip gradients
        clipped_gradients = []
        for gradients in all_gradients:
            clipped_grad = DefenseUtils.clip_gradients(gradients, self.max_norm)
            clipped_gradients.append(clipped_grad)
        
        # Apply clipped gradients to model
        self._apply_gradients_to_model(model, clipped_gradients)
        
        # Update client models
        for client in clients:
            client.model = copy.deepcopy(model)
        
        return model
    
    def detect_attack(self, clients, **kwargs):
        """Detect attacks by analyzing gradient norms"""
        print("Detecting attacks using gradient norm analysis...")
        
        suspicious_clients = []
        gradient_norms = []
        
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    norm = DefenseUtils.calculate_gradient_norm(gradients)
                    gradient_norms.append(norm)
                    
                    # Flag clients with unusually large gradient norms
                    if norm > self.max_norm * 2:  # Threshold for suspicion
                        suspicious_clients.append(client)
        
        # Use outlier detection
        if len(gradient_norms) > 3:
            outlier_indices = DefenseUtils.detect_outliers(gradient_norms, threshold=2.0)
            for idx in outlier_indices:
                if idx < len(clients) and clients[idx] not in suspicious_clients:
                    suspicious_clients.append(clients[idx])
        
        return len(suspicious_clients) > 0, suspicious_clients
    
    def remove_backdoor(self, model, **kwargs):
        """Remove backdoor by applying gradient clipping during training"""
        print("Removing backdoor using gradient clipping...")
        
        # This method would be called during the training process
        # to ensure gradients are clipped at each step
        return model
    
    def _apply_gradients_to_model(self, model, gradients_list):
        """Apply aggregated gradients to the model"""
        if not gradients_list:
            return
        
        # Average the gradients from all clients
        avg_gradients = []
        num_clients = len(gradients_list)
        
        for param_idx in range(len(gradients_list[0])):
            if gradients_list[0][param_idx] is not None:
                avg_grad = torch.zeros_like(gradients_list[0][param_idx])
                for client_grads in gradients_list:
                    if client_grads[param_idx] is not None:
                        avg_grad += client_grads[param_idx]
                avg_grad /= num_clients
                avg_gradients.append(avg_grad)
            else:
                avg_gradients.append(None)
        
        # Apply gradients to model parameters
        param_idx = 0
        for param in model.parameters():
            if param_idx < len(avg_gradients) and avg_gradients[param_idx] is not None:
                param.data -= avg_gradients[param_idx]  # Assuming learning rate is 1
            param_idx += 1
