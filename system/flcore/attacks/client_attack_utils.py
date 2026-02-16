import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable


class ClientAttackUtils:
    
    def __init__(self, device, dataset):
        self.device = device
        self.dataset = dataset

    def poisontest(self, model, testloader, poison_label, trigger, pattern, device, dataset):
        """测试攻击成功率"""
        model.eval()
        test_acc = 0
        sample = 0
        
        # 检查攻击类型
        if (trigger is None or pattern is None or 
            trigger == "BADPFL_TRIGGER" or pattern == "BADPFL_PATTERN"):
            # Bad-PFL 攻击：尝试获取全局Bad-PFL攻击实例
            try:
                # 尝试从全局变量获取Bad-PFL攻击实例
                import sys
                current_module = sys.modules[__name__]
                if hasattr(current_module, '_global_badpfl_attack'):
                    badpfl_attack = getattr(current_module, '_global_badpfl_attack')
                else:
                    print("Warning: Bad-PFL attack method not found, returning default result")
                    return 0, 1
                
                # 使用Bad-PFL攻击的poison_batch_for_eval方法
                for batch_id, (datas, labels) in enumerate(testloader):
                    x, y = datas.to(device), labels.to(device)
                    poisoned_x, poisoned_y = badpfl_attack.poison_batch_for_eval(
                        data=x, labels=y, client_model=model
                    )
                    with torch.no_grad():
                        output = model(poisoned_x)
                        test_acc += (output.argmax(dim=1) == poisoned_y).sum().item()
                        sample += poisoned_y.size(0)
                return test_acc, sample
                
            except Exception as e:
                print(f"Bad-PFL poisontest error: {e}")
                return 0, 1
        
        # 检查是否为模型替换攻击
        elif (trigger == "MODEL_REPLACEMENT_TRIGGER" or pattern == "MODEL_REPLACEMENT_PATTERN"):
            # 模型替换攻击：使用特定的触发器模式
            try:
                # 尝试获取全局模型替换攻击实例
                import sys
                current_module = sys.modules[__name__]
                if hasattr(current_module, '_global_model_replacement_attack'):
                    model_replacement_attack = getattr(current_module, '_global_model_replacement_attack')
                    # 使用模型替换攻击的评估方法
                    successful_attacks, total_samples = model_replacement_attack._evaluate_asr_on_loader(
                        model, testloader, device
                    )
                    return successful_attacks, total_samples
                else:
                    print("Warning: Model replacement attack method not found, returning default result")
                    return 0, 1
            except Exception as e:
                print(f"Model replacement poisontest error: {e}")
                return 0, 1
        
        # 检查是否为BadNets攻击
        elif (trigger == "BADNETS_TRIGGER" or pattern == "BADNETS_PATTERN"):
            # BadNets攻击：使用特定的触发器模式
            try:
                # 尝试获取全局BadNets攻击实例
                import sys
                current_module = sys.modules[__name__]
                if hasattr(current_module, '_global_badnets_attack'):
                    badnets_attack = getattr(current_module, '_global_badnets_attack')
                    # 使用BadNets攻击的评估方法
                    asr = badnets_attack._evaluate_asr_on_loader(model, testloader, device)
                    total_samples = len(testloader.dataset)
                    successful_attacks = int(asr * total_samples)
                    return successful_attacks, total_samples
                else:
                    print("Warning: BadNets attack method not found, returning default result")
                    return 0, 1
            except Exception as e:
                print(f"BadNets poisontest error: {e}")
                return 0, 1
        
        # 传统攻击方法：使用触发器
        # 创建模式掩码
        if dataset.lower() in ["mnist", "fmnist","fashionmnist"]:
            patterntensor = torch.ones((1, 28, 28)).float().to(device)
        elif dataset.lower() in ["cifar10", "cifar100"]:
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
        elif dataset.lower() == "iot":
            patterntensor = torch.ones((115)).float().to(device)
        elif dataset.lower() in ["har", "uci_har"]:
            # HAR数据集：9x1x128时间序列数据
            patterntensor = torch.ones((9, 1, 128)).float().to(device)
        else:
            # 默认处理其他数据集类型
            print(f"Warning: Unknown dataset {dataset}, using default pattern tensor")
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
    
        # 统一模式处理
        for pos in pattern:
            if dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
                patterntensor[0][pos[0]][pos[1]] = 0
            elif dataset.lower() in ["cifar10", "cifar100"]:
                patterntensor[:, pos[0], pos[1]] = 0
            elif dataset.lower() == "iot":
                patterntensor[pos] = 0
            elif dataset.lower() in ["har", "uci_har"]:
                # HAR数据集：处理[channel, height, time]格式的位置
                if len(pos) >= 3:
                    patterntensor[pos[0], pos[1], pos[2]] = 0
            else:
                # 默认处理其他数据集类型
                if pos[0] < patterntensor.shape[1] and pos[1] < patterntensor.shape[2]:
                    patterntensor[:, pos[0], pos[1]] = 0

        # trigger is a list
        for batch_id, (datas, labels) in enumerate(testloader):
            x, y = datas.to(device), labels.to(device)
            new_images = x.clone()
            new_targets = torch.full_like(y, poison_label)
            
            for index in range(x.size(0)):
                # 检查 trigger 的类型，处理 Bad-PFL 和传统攻击的不同格式
                if isinstance(trigger, list):
                    # 传统攻击：trigger 是列表
                    current_trigger = trigger[poison_label]
                else:
                    # Bad-PFL 攻击：trigger 是张量
                    current_trigger = trigger
                
                new_images[index] = self.add_pixel_pattern(x[index], current_trigger, pattern,
                                                           patterntensor, device, dataset)
            with torch.no_grad():
                output = model(new_images)
                test_acc += (output.argmax(dim=1) == new_targets).sum().item()
                sample += new_targets.size(0)

        return test_acc, sample  
    
    def add_pixel_pattern(self, x, trigger, pattern, pattern_mask, device, dataset, original_label=None):
        """
        添加像素级触发器到图像
        
        参数:
        - x: 输入图像
        - trigger: 触发器（可以是单个张量或触发器列表）
        - pattern: 触发器模式
        - pattern_mask: 模式掩码
        - device: 设备
        - dataset: 数据集名称
        - original_label: 原标签（用于从trigger_list中选择对应的触发器）
        """
        # 确保所有张量都在同一设备上
        x = x.to(device)
        pattern_mask = pattern_mask.to(device)
        
        # 处理trigger参数类型转换
        if isinstance(trigger, list):
            # 如果是触发器列表，根据原标签选择对应的触发器
            if original_label is not None:
                trigger_index = original_label % len(trigger)
                trigger = trigger[trigger_index]
            else:
                # 如果没有提供原标签，取第一个触发器
                trigger = trigger[0] if trigger else None
        elif not isinstance(trigger, torch.Tensor):
            # 如果不是张量，转换为张量
            trigger = torch.tensor(trigger, device=device)
        
        # 确保在正确的设备上
        trigger = trigger.to(device)
        
        # 原有的函数逻辑保持不变
        if dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            image = x * pattern_mask
            image = image + trigger
            image = torch.clamp(image, -1, 1)
        elif dataset.lower() in ["cifar10", "cifar100"]:
            image = x * pattern_mask
            image = image + trigger
            image = torch.clamp(image, -1, 1)
        elif dataset.lower() == "iot":
            image = x * pattern_mask
            image = image + trigger
        elif dataset.lower() in ["har", "uci_har"]:
            # HAR数据集：时间序列数据触发器应用
            image = x * pattern_mask
            image = image + trigger
            # HAR数据通常不需要clamp，保持原始数值范围
        else:
            # 默认处理其他数据集类型
            image = x * pattern_mask
            image = image + trigger
            image = torch.clamp(image, -1, 1)
        return image

    def estimate_grad(self, model, trainloader, device):
        """估计全局梯度 & 待定"""
        model.train()
        total_grad = [torch.zeros_like(p) for p in model.parameters()]
    
        # 使用全量训练数据计算平均梯度
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = nn.CrossEntropyLoss()(outputs, y)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss.backward()
            
            # 累加梯度
            with torch.no_grad():
                for i, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        total_grad[i] += param.grad
        
        # 计算平均梯度
        num_batches = len(trainloader)
        estimated_global_grad = [g / num_batches for g in total_grad]
        
        return estimated_global_grad

    def load_poison_data(self, trainloader, poison_ratio, poison_label, noise_trigger, pattern, batch_size, device, dataset):
        """
        加载混合训练数据：包含主任务数据和投毒数据
        poison_ratio: 每个batch中投毒的样本数量（如4表示batch_size=16时，4个样本投毒，12个样本正常）
        """
        # 检查是否为 Bad-PFL 攻击（noise_trigger 为 None 或 pattern 为 None）
        if noise_trigger is None or pattern is None:
            # Bad-PFL 攻击：直接返回原始数据，实际的中毒逻辑在训练时处理
            print("Bad-PFL: Using original data loader for poisoning")
            return trainloader
        
        # 传统攻击方法：使用触发器
        # 创建模式掩码
        if dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            patterntensor = torch.ones((1, 28, 28)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
        elif dataset.lower() in ["cifar10", "cifar100"]:
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[:, pos[0], pos[1]] = 0
        elif dataset.lower() == "iot":
            patterntensor = torch.ones((115)).float().to(device)
            for i in pattern:
                patterntensor[i] = 0
        elif dataset.lower() in ["har", "uci_har"]:
            # HAR数据集：9x1x128时间序列数据
            patterntensor = torch.ones((9, 1, 128)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                if len(pos) >= 3:  # [channel, height, time]
                    patterntensor[pos[0], pos[1], pos[2]] = 0
        else:
            # 默认处理其他数据集类型
            print(f"Warning: Unknown dataset {dataset}, using default pattern tensor")
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                if pos[0] < patterntensor.shape[1] and pos[1] < patterntensor.shape[2]:
                    patterntensor[:, pos[0], pos[1]] = 0

        poisoned_batches = []
        
        for x_batch, y_batch in trainloader:
            x_poison = x_batch.clone()
            y_poison = y_batch.clone()
            
            # 计算当前batch中需要投毒的样本数量
            batch_size_actual = len(x_batch)
            batch_poison_num = min(int(poison_ratio), batch_size_actual)  # 确保是整数
            
            # 随机选择要投毒的样本索引
            poison_indices = torch.randperm(batch_size_actual)[:batch_poison_num]
            
            for i in poison_indices:
                # 统一触发器选择策略：使用poison_label选择触发器，确保攻击一致性
                if isinstance(noise_trigger, list):
                    current_trigger = noise_trigger[poison_label % len(noise_trigger)]
                else:
                    current_trigger = noise_trigger
                
                x_poison[i] = self.add_pixel_pattern(x_batch[i], current_trigger, 
                                                    pattern, patterntensor, device, dataset, original_label=poison_label)
                y_poison[i] = poison_label
                
            poisoned_batches.append((x_poison, y_poison))
            
        poisoned_data = torch.utils.data.ConcatDataset([
            torch.utils.data.TensorDataset(x, y) for x, y in poisoned_batches
        ])
        
        return DataLoader(poisoned_data, batch_size=batch_size, drop_last=True, shuffle=False) 

    def load_poison_data3(self, trainloader, poison_ratio, poison_label, noise_trigger, pattern, batch_size, device, dataset):
        """加载分离的中毒数据 - 返回干净数据和中毒数据的DataLoader"""
        # 创建模式掩码
        if dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            patterntensor = torch.ones((1, 28, 28)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[0][pos[0]][pos[1]] = 0
        elif dataset.lower() in ["cifar10", "cifar100"]:
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                patterntensor[:, pos[0], pos[1]] = 0
        elif dataset.lower() == "iot":
            patterntensor = torch.ones((115)).float().to(device)
            for i in pattern:
                patterntensor[i] = 0
        elif dataset.lower() in ["har", "uci_har"]:
            # HAR数据集：9x1x128时间序列数据
            patterntensor = torch.ones((9, 1, 128)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                if len(pos) >= 3:  # [channel, height, time]
                    patterntensor[pos[0], pos[1], pos[2]] = 0
        else:
            # 默认处理其他数据集类型
            print(f"Warning: Unknown dataset {dataset}, using default pattern tensor")
            patterntensor = torch.ones((3, 32, 32)).float().to(device)
            for i in range(0, len(pattern)):
                pos = pattern[i]
                if pos[0] < patterntensor.shape[1] and pos[1] < patterntensor.shape[2]:
                    patterntensor[:, pos[0], pos[1]] = 0

        clean_batches = []
        poisoned_batches = []
        poison_count = 0
        target_poison_num = poison_ratio
        
        for x_batch, y_batch in trainloader:
            x_clean = x_batch.clone()
            y_clean = y_batch.clone()
            x_poison = x_batch.clone()
            y_poison = y_batch.clone()
            
            batch_poison_num = min(target_poison_num - poison_count, len(x_batch))
            
            # 分离干净数据和中毒数据
            for i in range(len(x_batch)):
                if i < batch_poison_num:
                    # 中毒数据
                    # 检查 noise_trigger 的类型，处理 Bad-PFL 和传统攻击的不同格式
                    if isinstance(noise_trigger, list):
                        # 传统攻击：noise_trigger 是列表
                        current_trigger = noise_trigger[poison_label]
                    else:
                        # Bad-PFL 攻击：noise_trigger 是张量
                        current_trigger = noise_trigger
                    
                    x_poison[i] = self.add_pixel_pattern(x_batch[i], current_trigger, 
                                                        pattern, patterntensor, device, dataset)
                    y_poison[i] = poison_label
                    poison_count += 1
                else:
                    # 干净数据
                    x_clean[i] = x_batch[i]
                    y_clean[i] = y_batch[i]
                
            if batch_poison_num < len(x_batch):
                clean_batches.append((x_clean[batch_poison_num:], y_clean[batch_poison_num:]))
            if batch_poison_num > 0:
                poisoned_batches.append((x_poison[:batch_poison_num], y_poison[:batch_poison_num]))
            
        # 创建DataLoader
        clean_loader = None
        poisoned_loader = None
        
        if clean_batches:
            clean_data = torch.utils.data.ConcatDataset([
                torch.utils.data.TensorDataset(x, y) for x, y in clean_batches
            ])
            clean_loader = DataLoader(clean_data, batch_size=batch_size, drop_last=True, shuffle=True)
            
        if poisoned_batches:
            poisoned_data = torch.utils.data.ConcatDataset([
                torch.utils.data.TensorDataset(x, y) for x, y in poisoned_batches
            ])
            poisoned_loader = DataLoader(poisoned_data, batch_size=batch_size, drop_last=True, shuffle=True)
            
        return clean_loader, poisoned_loader 