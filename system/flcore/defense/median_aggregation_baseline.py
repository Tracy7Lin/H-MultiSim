import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any


class MedianAggregationBaseline:
    """Median Aggregation Defense - Baseline Version
    
    This defense method uses coordinate-wise median to aggregate models,
    which is robust to outliers. Only contains the core aggregation functionality
    without attack detection.
    """
    
    def __init__(self):
        """Initialize median aggregation baseline"""
        pass
    
    def aggregate_models(self, uploaded_models: List[nn.Module], uploaded_weights: List[float] = None) -> nn.Module:
        """
        Aggregate models using coordinate-wise median.
        
        Args:
            uploaded_models: List of uploaded client models
            uploaded_weights: Optional weights (not used in median aggregation, kept for compatibility)
            
        Returns:
            Aggregated model
        """
        if not uploaded_models:
            raise ValueError("No models to aggregate")
        
        # Create a new model with the same structure
        aggregated_model = copy.deepcopy(uploaded_models[0])
        
        # For each parameter, compute the median across all clients
        for param_name, param in aggregated_model.named_parameters():
            # Stack all client parameters for this layer
            param_stack = []
            for client_model in uploaded_models:
                client_param = dict(client_model.named_parameters())[param_name]
                param_stack.append(client_param.data.clone())
            
            # Stack along a new dimension (dim=0)
            stacked_params = torch.stack(param_stack, dim=0)
            
            # Compute coordinate-wise median
            median_param = torch.median(stacked_params, dim=0)[0]
            
            # Update the aggregated model
            param.data = median_param
        
        return aggregated_model
    
    def aggregate_models_with_weights(self, uploaded_models: List[nn.Module], 
                                     uploaded_weights: List[float]) -> nn.Module:
        """
        Aggregate models using weighted median (falls back to simple median).
        
        Note: This is kept for API compatibility but uses simple median.
        True weighted median is complex and not commonly used in FL.
        
        Args:
            uploaded_models: List of uploaded client models
            uploaded_weights: Weights for each model (not used)
            
        Returns:
            Aggregated model
        """
        # For simplicity, use simple median (most common approach in FL)
        return self.aggregate_models(uploaded_models)
