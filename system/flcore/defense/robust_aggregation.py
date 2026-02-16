import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any

from .base_defense import BaseDefense
from .defense_utils import DefenseUtils


class RobustAggregation(BaseDefense):
    """Robust Aggregation Defense
    
    This defense method uses robust aggregation techniques like median,
    trimmed mean, or Krum to aggregate gradients from clients.
    """
    
    def __init__(self, args, device, aggregation_method: str = 'median', 
                 trim_ratio: float = 0.1, f: int = 1):
        super(RobustAggregation, self).__init__(args, device)
        self.aggregation_method = aggregation_method
        self.trim_ratio = trim_ratio
        self.f = f  # Number of Byzantine clients for Krum
        
    def apply_defense(self, model, clients, **kwargs):
        """Apply robust aggregation defense"""
        print(f"Applying Robust Aggregation defense with {self.aggregation_method}...")
        
        # Get gradients from all clients
        all_gradients = []
        valid_clients = []
        
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    all_gradients.append(gradients)
                    valid_clients.append(client)
        
        if not all_gradients:
            print("No gradients available for aggregation")
            return model
        
        # Apply robust aggregation
        aggregated_gradients = DefenseUtils.robust_aggregation(
            all_gradients, 
            method=self.aggregation_method
        )
        
        # Apply aggregated gradients to model
        self._apply_gradients_to_model(model, aggregated_gradients)
        
        # Update client models
        for client in clients:
            client.model = deepcopy(model)
        
        return model
    
    def detect_attack(self, clients, **kwargs):
        """Detect attacks by analyzing gradient distributions"""
        print("Detecting attacks using gradient distribution analysis...")
        
        suspicious_clients = []
        all_gradients = []
        
        for client in clients:
            if hasattr(client, 'get_gradients'):
                gradients = client.get_gradients()
                if gradients is not None:
                    all_gradients.append(gradients)
        
        if len(all_gradients) < 3:
            return False, suspicious_clients
        
        # Analyze gradient distributions for each parameter
        num_params = len(all_gradients[0])
        
        for param_idx in range(num_params):
            param_gradients = []
            for client_grads in all_gradients:
                if client_grads[param_idx] is not None:
                    param_gradients.append(client_grads[param_idx].flatten())
            
            if len(param_gradients) > 2:
                # Calculate statistics for this parameter
                stacked_grads = torch.stack(param_gradients)
                mean_grad = torch.mean(stacked_grads, dim=0)
                std_grad = torch.std(stacked_grads, dim=0)
                
                # Find outliers
                for client_idx, grad in enumerate(param_gradients):
                    if client_idx < len(clients):
                        z_scores = torch.abs((grad - mean_grad) / (std_grad + 1e-8))
                        if torch.any(z_scores > 3.0):  # 3-sigma rule
                            if clients[client_idx] not in suspicious_clients:
                                suspicious_clients.append(clients[client_idx])
        
        return len(suspicious_clients) > 0, suspicious_clients
    
    def remove_backdoor(self, model, **kwargs):
        """Remove backdoor by using robust aggregation during training"""
        print("Removing backdoor using robust aggregation...")
        
        # This method would be called during the training process
        # to ensure robust aggregation is used at each step
        return model
    
    def _apply_gradients_to_model(self, model, aggregated_gradients):
        """Apply aggregated gradients to the model"""
        if not aggregated_gradients:
            return
        
        # Apply gradients to model parameters
        param_idx = 0
        for param in model.parameters():
            if param_idx < len(aggregated_gradients) and aggregated_gradients[param_idx] is not None:
                param.data -= aggregated_gradients[param_idx]  # Assuming learning rate is 1
            param_idx += 1
    
    def filter_malicious_clients(self, clients, **kwargs):
        """Filter out potentially malicious clients"""
        print("Filtering malicious clients...")
        
        # Use model similarity to filter clients
        benign_clients, suspicious_clients = DefenseUtils.filter_suspicious_clients(
            clients, 
            similarity_threshold=kwargs.get('similarity_threshold', 0.8)
        )
        
        print(f"Filtered {len(suspicious_clients)} suspicious clients out of {len(clients)} total clients")
        
        return benign_clients, suspicious_clients
