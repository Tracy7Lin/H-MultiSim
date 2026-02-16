import torch
import copy


class TriggerUtils:
    
    @staticmethod
    def initialize_trigger(dataset, device, trigger_size=10):
        trigger_pattern = []
        trigger_list = []
        
        if dataset.lower() in ["cifar10", "cifar100"]:
            image_size = 32
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            image_size = 28
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        # create trigger
        for i in range(10, 10 + trigger_size):
            for j in range(10, 10 + trigger_size):
                # limited field
                if i < image_size and j < image_size:
                    trigger_pattern.append([i, j])
        
        # initial_trigger
        if dataset.lower() in ["cifar10", "cifar100"]:
            initial_trigger = torch.zeros((3, image_size, image_size)).float().to(device)  # 3通道
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            initial_trigger = torch.zeros((1, image_size, image_size)).float().to(device)   # 单通道
        
        # set trigger area
        for pos in trigger_pattern:
            if dataset.lower() in ["cifar10", "cifar100"]:
                initial_trigger[:, pos[0], pos[1]] = 0.5 
            elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
                initial_trigger[0, pos[0], pos[1]] = 0.8
        
        # create trigger for all the labels
        for i in range(10):
            trigger_list.append(copy.deepcopy(initial_trigger))
        
        # Verify the size of the trigger
        if dataset.lower() in ["cifar10", "cifar100"]:
            assert trigger_list[0].shape == (3, 32, 32), "CIFAR 3x32x32"
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            assert trigger_list[0].shape == (1, 28, 28), "MNIST/FMNIST1x28x28"
            
        return trigger_pattern, trigger_list
    
    @staticmethod
    def create_pattern_mask(pattern, dataset, device):
        if dataset.lower() in ["cifar10", "cifar100"]:
            pattern_mask = torch.ones((3, 32, 32), device=device)
            for pos in pattern:
                pattern_mask[:, pos[0], pos[1]] = 0
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            pattern_mask = torch.ones((1, 28, 28), device=device)
            for pos in pattern:
                pattern_mask[0, pos[0], pos[1]] = 0
        elif dataset.lower() == "iot":
            pattern_mask = torch.ones(115, device=device)
            pattern_mask[pattern] = 0
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return pattern_mask
    
    @staticmethod
    def add_trigger_to_batch(batch_data, trigger, pattern, dataset, device):
        """
        Add triggers to batch data
        Noise was added to other positions of the poisoned samples, 
        in order to make the model pay more attention to the trigger.
        """
        pattern_mask = TriggerUtils.create_pattern_mask(pattern, dataset, device)
        
        if dataset.lower() in ["cifar10", "cifar100"]:
            pattern_mask = pattern_mask.unsqueeze(0).repeat(len(batch_data), 1, 1, 1)
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            pattern_mask = pattern_mask.unsqueeze(0).repeat(len(batch_data), 1, 1, 1)
        elif dataset.lower() == "iot":
            pattern_mask = pattern_mask.unsqueeze(0).repeat(len(batch_data), 1)
        
        triggered_data = batch_data * pattern_mask + trigger
        
        if dataset.lower() in ["cifar10", "cifar100", "mnist", "fmnist", "fashionmnist"]:
            triggered_data = torch.clamp(triggered_data, -1, 1)
            
        return triggered_data
    
    @staticmethod
    def validate_trigger(trigger, dataset):
        """Verify the trigger format"""
        if dataset.lower() in ["cifar10", "cifar100"]:
            expected_shape = (3, 32, 32)
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            expected_shape = (1, 28, 28)
        elif dataset.lower() == "iot":
            expected_shape = (115,)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
            
        if trigger.shape != expected_shape:
            raise ValueError(f"Trigger shape {trigger.shape} does not match expected shape {expected_shape} for dataset {dataset}")
        
        return True 