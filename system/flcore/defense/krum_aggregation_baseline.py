import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any


class KrumAggregationBaseline:
    """Krum Aggregation Defense - Baseline Version
    
    Krum is a robust aggregation method that selects the model with the smallest
    sum of distances to its closest neighbors. This makes it resistant to Byzantine
    attacks where malicious clients try to poison the global model.
    
    Reference: 
    Blanchard et al. "Machine learning with adversaries: Byzantine tolerant 
    gradient descent." NeurIPS 2017.
    """
    
    def __init__(self, num_byzantine: int = 0, multi_krum: bool = False):
        """
        Args:
            num_byzantine: Number of suspected Byzantine (malicious) clients
            multi_krum: If True, use Multi-Krum (average of top-m models)
                       If False, use standard Krum (select only 1 model)
        """
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        print(f"Krum Aggregation initialized: Byzantine={num_byzantine}, Multi-Krum={multi_krum}")
        
    def _compute_euclidean_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """
        Compute Euclidean distance between two models in parameter space.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Euclidean distance between the two models
        """
        distance = 0.0
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, "Model structures must match"
            distance += torch.sum((param1.data - param2.data) ** 2).item()
        
        return np.sqrt(distance)
    
    def _compute_distance_matrix(self, models: List[nn.Module]) -> np.ndarray:
        """
        Compute pairwise distance matrix between all models.
        
        Args:
            models: List of client models
            
        Returns:
            Distance matrix of shape (n, n) where n is number of models
        """
        n = len(models)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._compute_euclidean_distance(models[i], models[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _compute_krum_scores(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute Krum score for each model.
        
        The Krum score is the sum of distances to the (n-f-2) closest neighbors,
        where n is total clients and f is number of Byzantine clients.
        
        Args:
            distance_matrix: Pairwise distance matrix
            
        Returns:
            Array of Krum scores for each model (lower is better)
        """
        n = distance_matrix.shape[0]
        # Number of neighbors to consider: n - f - 2
        # We need at least n - 2f - 1 honest clients for Krum to work
        num_neighbors = max(1, n - self.num_byzantine - 2)
        
        scores = np.zeros(n)
        for i in range(n):
            # Get distances from model i to all other models
            distances = distance_matrix[i].copy()
            # Sort distances (excluding distance to itself which is 0)
            sorted_distances = np.sort(distances)[1:num_neighbors + 1]
            # Sum of distances to closest neighbors
            scores[i] = np.sum(sorted_distances)
        
        return scores
    
    def aggregate_models(self, uploaded_models: List[nn.Module], 
                        uploaded_weights: List[float] = None) -> nn.Module:
        """
        Aggregate models using Krum or Multi-Krum algorithm.
        
        Args:
            uploaded_models: List of uploaded client models
            uploaded_weights: Optional weights (not used in Krum, kept for compatibility)
            
        Returns:
            Aggregated model
        """
        if not uploaded_models:
            raise ValueError("No models to aggregate")
        
        n = len(uploaded_models)
        
        # Check if we have enough clients
        min_clients = 2 * self.num_byzantine + 1
        if n < min_clients:
            print(f"Warning: Only {n} clients, but need at least {min_clients} for "
                  f"{self.num_byzantine} Byzantine clients. Using simple average instead.")
            return self._simple_average(uploaded_models)
        
        # Compute distance matrix
        print(f"Computing distance matrix for {n} models...")
        distance_matrix = self._compute_distance_matrix(uploaded_models)
        
        # Compute Krum scores
        scores = self._compute_krum_scores(distance_matrix)
        
        # Select models based on Krum scores
        if self.multi_krum:
            # Multi-Krum: Select top m models (where m = n - f)
            m = n - self.num_byzantine
            selected_indices = np.argsort(scores)[:m]
            print(f"Multi-Krum: Selected {m} models with lowest scores")
            print(f"  Selected indices: {selected_indices}")
            print(f"  Selected scores: {scores[selected_indices]}")
            
            # Average the selected models
            aggregated_model = copy.deepcopy(uploaded_models[0])
            for param in aggregated_model.parameters():
                param.data.zero_()
            
            for idx in selected_indices:
                for target_param, source_param in zip(
                    aggregated_model.parameters(), 
                    uploaded_models[idx].parameters()
                ):
                    target_param.data += source_param.data / m
            
        else:
            # Standard Krum: Select the single best model
            best_idx = np.argmin(scores)
            print(f"Krum: Selected model {best_idx} with score {scores[best_idx]:.4f}")
            print(f"  All scores: {scores}")
            aggregated_model = copy.deepcopy(uploaded_models[best_idx])
        
        return aggregated_model
    
    def _simple_average(self, models: List[nn.Module]) -> nn.Module:
        """
        Simple average aggregation as fallback.
        
        Args:
            models: List of models to average
            
        Returns:
            Averaged model
        """
        aggregated_model = copy.deepcopy(models[0])
        for param in aggregated_model.parameters():
            param.data.zero_()
        
        n = len(models)
        for model in models:
            for target_param, source_param in zip(
                aggregated_model.parameters(), 
                model.parameters()
            ):
                target_param.data += source_param.data / n
        
        return aggregated_model
