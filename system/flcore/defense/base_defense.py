import torch
import torch.nn as nn
import copy
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable


class BaseDefense:
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.defense_method = args.defense
        
    def select_benign_clients(self, clients, num_benign=None):
        """Select benign clients for defense"""
        if num_benign is None:
            num_benign = len(clients)
        benign_clients = random.sample(clients, min(num_benign, len(clients)))
        print("=====================Benign client=================================")
        print("Selected benign client IDs:", [client.id for client in benign_clients])
        print("=============================================================")
        return benign_clients
    
    def evaluate_defense_effectiveness(self, clients, trigger, pattern, poison_label):
        """Evaluate the effectiveness of defense by testing ASR after defense"""
        total_asr_correct = 0
        total_asr_samples = 0
        
        for client in clients:
            asr_correct, asr_samples = client.poisontest(
                poison_label=poison_label,
                trigger=trigger,
                pattern=pattern
            )
            total_asr_correct += asr_correct
            total_asr_samples += asr_samples
        
        global_asr = total_asr_correct / total_asr_samples if total_asr_samples > 0 else 0.0
        print("Global Attack Success Rate (ASR) after defense: {:.4f}".format(global_asr))
        return global_asr
    
    def evaluate_clean_accuracy(self, clients):
        """Evaluate the clean accuracy after defense"""
        total_correct = 0
        total_samples = 0
        
        for client in clients:
            correct, samples = client.test()
            total_correct += correct
            total_samples += samples
        
        global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print("Global Clean Accuracy after defense: {:.4f}".format(global_accuracy))
        return global_accuracy
    
    def apply_defense(self, model, clients, **kwargs):
        """Apply defense method to the model and clients"""
        raise NotImplementedError("Subclasses must implement apply_defense method")
    
    def detect_attack(self, clients, **kwargs):
        """Detect if there are attacks in the system"""
        raise NotImplementedError("Subclasses must implement detect_attack method")
    
    def remove_backdoor(self, model, **kwargs):
        """Remove backdoor from the model"""
        raise NotImplementedError("Subclasses must implement remove_backdoor method")
