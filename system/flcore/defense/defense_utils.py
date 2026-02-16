import torch
import torch.nn as nn
import numpy as np
import copy
from typing import List, Tuple, Dict, Any


class DefenseUtils:
    """Utility class for defense methods"""
    
    @staticmethod
    def calculate_gradient_norm(gradients: List[torch.Tensor]) -> float:
        """Calculate the L2 norm of gradients"""
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                param_norm = grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    @staticmethod
    def clip_gradients(gradients: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
        """Clip gradients to a maximum norm"""
        total_norm = DefenseUtils.calculate_gradient_norm(gradients)
        clip_coef = max_norm / (total_norm + 1e-6)
        
        if clip_coef < 1:
            clipped_gradients = []
            for grad in gradients:
                if grad is not None:
                    clipped_gradients.append(grad * clip_coef)
                else:
                    clipped_gradients.append(None)
            return clipped_gradients
        return gradients
    
    @staticmethod
    def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
        """Detect outliers using z-score method"""
        if len(values) < 3:
            return []
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        z_scores = np.abs((values - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0].tolist()
        
        return outlier_indices
    
    @staticmethod
    def robust_aggregation(gradients: List[List[torch.Tensor]], method: str = 'median') -> List[torch.Tensor]:
        """Robust aggregation of gradients"""
        if method == 'median':
            return DefenseUtils._median_aggregation(gradients)
        elif method == 'trimmed_mean':
            return DefenseUtils._trimmed_mean_aggregation(gradients)
        elif method == 'krum':
            return DefenseUtils._krum_aggregation(gradients)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    @staticmethod
    def _median_aggregation(gradients: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Aggregate gradients using median"""
        num_params = len(gradients[0])
        aggregated_gradients = []
        
        for param_idx in range(num_params):
            param_gradients = []
            for client_grads in gradients:
                if client_grads[param_idx] is not None:
                    param_gradients.append(client_grads[param_idx].flatten())
            
            if param_gradients:
                stacked_grads = torch.stack(param_gradients)
                median_grad = torch.median(stacked_grads, dim=0)[0]
                aggregated_gradients.append(median_grad.view_as(gradients[0][param_idx]))
            else:
                aggregated_gradients.append(None)
        
        return aggregated_gradients
    
    @staticmethod
    def _trimmed_mean_aggregation(gradients: List[List[torch.Tensor]], trim_ratio: float = 0.1) -> List[torch.Tensor]:
        """Aggregate gradients using trimmed mean"""
        num_params = len(gradients[0])
        aggregated_gradients = []
        
        for param_idx in range(num_params):
            param_gradients = []
            for client_grads in gradients:
                if client_grads[param_idx] is not None:
                    param_gradients.append(client_grads[param_idx].flatten())
            
            if param_gradients:
                stacked_grads = torch.stack(param_gradients)
                sorted_grads, _ = torch.sort(stacked_grads, dim=0)
                
                # Trim the top and bottom trim_ratio
                trim_size = int(len(sorted_grads) * trim_ratio)
                trimmed_grads = sorted_grads[trim_size:-trim_size]
                
                mean_grad = torch.mean(trimmed_grads, dim=0)
                aggregated_gradients.append(mean_grad.view_as(gradients[0][param_idx]))
            else:
                aggregated_gradients.append(None)
        
        return aggregated_gradients
    
    @staticmethod
    def _krum_aggregation(gradients: List[List[torch.Tensor]], f: int = 1) -> List[torch.Tensor]:
        """Aggregate gradients using Krum algorithm"""
        num_clients = len(gradients)
        num_params = len(gradients[0])
        
        # Calculate distances between all pairs of gradients
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = 0
                for param_idx in range(num_params):
                    if gradients[i][param_idx] is not None and gradients[j][param_idx] is not None:
                        dist += torch.norm(gradients[i][param_idx] - gradients[j][param_idx]) ** 2
                distances[i][j] = distances[j][i] = dist
        
        # Find the client with minimum sum of distances to closest (n-f-2) clients
        scores = []
        for i in range(num_clients):
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[:num_clients - f - 2])
            scores.append(score)
        
        best_client = np.argmin(scores)
        return gradients[best_client]
    
    @staticmethod
    def compute_model_similarity(model1: nn.Module, model2: nn.Module) -> float:
        """Compute cosine similarity between two models"""
        similarity = 0.0
        count = 0
        
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if name1 == name2:
                vec1 = param1.data.flatten()
                vec2 = param2.data.flatten()
                
                cos_sim = torch.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
                similarity += cos_sim.item()
                count += 1
        
        return similarity / count if count > 0 else 0.0
    
    @staticmethod
    def filter_suspicious_clients(clients: List, similarity_threshold: float = 0.8) -> Tuple[List, List]:
        """Filter out suspicious clients based on model similarity"""
        if len(clients) < 2:
            return clients, []
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(clients)):
            for j in range(i + 1, len(clients)):
                sim = DefenseUtils.compute_model_similarity(clients[i].model, clients[j].model)
                similarities.append((i, j, sim))
        
        # Find clients with low average similarity
        client_scores = {}
        for i, j, sim in similarities:
            if i not in client_scores:
                client_scores[i] = []
            if j not in client_scores:
                client_scores[j] = []
            client_scores[i].append(sim)
            client_scores[j].append(sim)
        
        suspicious_clients = []
        benign_clients = []
        
        for client_idx in range(len(clients)):
            if client_idx in client_scores:
                avg_sim = np.mean(client_scores[client_idx])
                if avg_sim < similarity_threshold:
                    suspicious_clients.append(clients[client_idx])
                else:
                    benign_clients.append(clients[client_idx])
            else:
                benign_clients.append(clients[client_idx])
        
        return benign_clients, suspicious_clients
