import copy
import random
import torch
import numpy as np
import time
import math
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms
from .base_attack import BaseAttack


# ==============================================================================
# 1. 辅助类 (这部分与您原始代码保持一致，无需改动)
# ==============================================================================

class AddTrigger:
    """基类，用于计算触发器叠加效果"""

    def __init__(self):
        pass

    def add_trigger(self, img):
        return (self.weight * img + self.res).type(torch.uint8)


class AddMNISTTrigger(AddTrigger):
    """为MNIST图像添加触发器"""

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """为CIFAR-10图像添加触发器"""

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:
            # 默认在右下角创建一个3x3的白色方块触发器
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        # 确保 pattern 是三通道的
        if self.pattern.shape[0] == 1:
            self.pattern = self.pattern.repeat(3, 1, 1)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # 确保 weight 是三通道的
        if self.weight.shape[0] == 1:
            self.weight = self.weight.repeat(3, 1, 1)

        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class AddHARTrigger(AddTrigger):
    """为HAR时间序列数据添加触发器"""

    def __init__(self, pattern, weight):
        super(AddHARTrigger, self).__init__()

        if pattern is None:
            # 默认触发器：使用更强的扰动设计以提高ASR
            self.pattern = torch.zeros((9, 1, 128), dtype=torch.float32)
            # 使用更强的扰动幅度和更长的窗口
            win_len, start, amp = 20, 70, 0.3
            t = torch.arange(win_len, dtype=torch.float32)
            import math
            patch = amp * torch.hann_window(win_len, periodic=True) * torch.sin(2 * math.pi * t / win_len)
            
            self.pattern[0, 0, start:start+win_len] = patch  # 通道0
            self.pattern[1, 0, start:start+win_len] = patch  # 通道1
            self.pattern[2, 0, start:start+win_len] = patch  # 通道2
            self.pattern[3, 0, start:start+win_len] = patch  # 通道3
        else:
            self.pattern = pattern

        if weight is None:
            self.weight = torch.zeros((9, 1, 128), dtype=torch.float32)
            # 使用更大的权重以提高触发效果
            win_len, start = 20, 70
            self.weight[0, 0, start:start+win_len] = 0.8
            self.weight[1, 0, start:start+win_len] = 0.8
            self.weight[2, 0, start:start+win_len] = 0.8
            self.weight[3, 0, start:start+win_len] = 0.8
        else:
            self.weight = weight

        self.res = self.weight * self.pattern

    def __call__(self, data):
        """为HAR数据添加触发器
        Args:
            data: torch.Tensor, shape (9, 1, 128)
        Returns:
            torch.Tensor: 添加触发器后的数据
        """
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        
        # 确保所有张量在同一设备上
        device = data.device
        weight = self.weight.to(device)
        res = self.res.to(device)
        
        # 应用触发器: triggered_data = (1 - weight) * data + weight * pattern
        triggered_data = (1.0 - weight) * data + res
        return triggered_data


# security/attack/BadNets.py (新代码)
class ModifyTarget:
    """修改标签的转换器"""
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        # 返回目标标签，并确保它是 Long 类型的 Tensor
        # 这对于分类任务的标签至关重要
        return torch.tensor(self.y_target).long()


# ==============================================================================
# 2. 核心修改部分 (新的、通用的投毒数据集实现)
# ==============================================================================

class PoisonedDatasetWrapper(Dataset):
    """
    一个通用的数据投毒包装器，它接收一个数据集（通常是元组列表），并应用投毒转换。
    """

    def __init__(self, benign_dataset, y_target, poisoned_rate, trigger_transform):
        self.benign_dataset = benign_dataset
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should be greater than or equal to zero.'

        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        self.trigger_transform = trigger_transform
        self.target_transform = ModifyTarget(y_target)

        # vvvvvvvvvvvvvv 新增：定义一个转换器，用于将 PIL Image 转为 Tensor vvvvvvvvvvvvvv
        self.to_tensor = transforms.ToTensor()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def __getitem__(self, index):
        img, target = self.benign_dataset[index]

        # 如果当前样本被选中进行投毒
        if index in self.poisoned_set:
            # 检查数据类型并应用触发器
            if isinstance(img, torch.Tensor):
                # 判断是时间序列还是图像
                if len(img.shape) == 3 and img.shape[1] == 1 and img.shape[2] > 50:
                    # 时间序列数据: (channels, 1, time_steps) - 如HAR (9, 1, 128)
                    img = self.trigger_transform(img)
                else:
                    # 图像数据Tensor -> PIL Image
                    if img.shape[0] == 3:  # RGB图像
                        img_pil = Image.fromarray((img.permute(1, 2, 0) * 255).byte().numpy())
                    elif img.shape[0] == 1:  # 灰度图像
                        img_pil = Image.fromarray((img.squeeze(0) * 255).byte().numpy(), mode='L')
                    else:
                        img_pil = Image.fromarray((img * 255).byte().numpy())
                    
                    # 应用触发器（返回PIL Image）
                    img_pil = self.trigger_transform(img_pil)
                    # 转换回Tensor
                    img = self.to_tensor(img_pil)
            else:
                # 已经是PIL Image，直接应用触发器
                img = self.trigger_transform(img)
                # 转换回Tensor
                img = self.to_tensor(img)
            
            target = self.target_transform(target)
        else:
            # 良性样本：确保返回Tensor
            if not isinstance(img, torch.Tensor):
                img = self.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.benign_dataset)


def CreatePoisonedDataset1(dataset_name, benign_dataset, y_target, poisoned_rate, pattern, weight, **kwargs):
    """
    新的工厂函数，根据数据集名称（字符串）来选择不同的触发器，并创建投毒数据集。
    """
    # 根据数据集名称选择合适的触发器转换器
    if 'MNIST' in dataset_name.upper() or 'FASHION' in dataset_name.upper():
        trigger_transform = AddMNISTTrigger(pattern, weight)
    elif 'CIFAR' in dataset_name.upper():
        trigger_transform = AddCIFAR10Trigger(pattern, weight)
    elif 'HAR' in dataset_name.upper():
        trigger_transform = AddHARTrigger(pattern, weight)
    else:
        # 如果未来支持更多数据集，在此处添加 elif 分支
        raise NotImplementedError(f"Trigger for dataset '{dataset_name}' is not implemented.")

    # 使用通用包装器创建投毒数据集
    return PoisonedDatasetWrapper(
        benign_dataset=benign_dataset,
        y_target=y_target,
        poisoned_rate=poisoned_rate,
        trigger_transform=trigger_transform
    )

# =======================================================================================
# 3. BadNets 攻击类 - 集成到联邦学习框架
# =======================================================================================


class BadNetsAttack(BaseAttack):
    """
    BadNets攻击实现
    
    核心思想：在训练数据中注入带有特定触发器的样本，并将其标签修改为目标标签
    这样训练出的模型在遇到带有相同触发器的输入时会输出目标标签
    """
    
    def __init__(self, args, device):
        super().__init__(args, device)
        
        # BadNets攻击参数
        self.target_label = getattr(args, 'badnets_target_label', 0)
        self.poisoned_rate = getattr(args, 'badnets_poison_rate', 0.5)
        self.trigger_size = getattr(args, 'badnets_trigger_size', 3)
        
        # 创建触发器模式和权重
        self.trigger_pattern, self.trigger_weight = self._create_trigger_pattern()
        
        print(f"BadNets Attack initialized:")
        print(f"  - Target label: {self.target_label}")
        print(f"  - Poisoned rate: {self.poisoned_rate}")
        print(f"  - Trigger size: {self.trigger_size}")
    
    def _create_trigger_pattern(self):
        """创建触发器模式和权重"""
        if self.dataset.lower() in ['mnist', 'fashionmnist']:
            # MNIST/FashionMNIST: 28x28 灰度图像
            pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            
            # 在右下角创建触发器
            pattern[0, -self.trigger_size:, -self.trigger_size:] = 255
            weight[0, -self.trigger_size:, -self.trigger_size:] = 1.0
            
        elif self.dataset.lower() in ['cifar10', 'cifar100']:
            # CIFAR: 32x32 彩色图像
            pattern = torch.zeros((3, 32, 32), dtype=torch.uint8)
            weight = torch.zeros((3, 32, 32), dtype=torch.float32)
            
            # 在右下角创建彩色触发器
            pattern[:, -self.trigger_size:, -self.trigger_size:] = 255
            weight[:, -self.trigger_size:, -self.trigger_size:] = 1.0
            
        elif self.dataset.lower() in ['har', 'uci_har']:
            # HAR: 9x1x128 时间序列数据
            pattern = torch.zeros((9, 1, 128), dtype=torch.float32)
            weight = torch.zeros((9, 1, 128), dtype=torch.float32)
            
            # 使用正弦波触发器
            win_len, start, amp = 20, 70, 0.3
            t = torch.arange(win_len, dtype=torch.float32)
            patch = amp * torch.hann_window(win_len, periodic=True) * torch.sin(2 * math.pi * t / win_len)
            
            # 在多个通道上添加触发器
            for c in [0, 1, 2, 3]:
                pattern[c, 0, start:start+win_len] = patch
                weight[c, 0, start:start+win_len] = 0.8
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        return pattern, weight
    
    def execute_attack_training(self, server, round_num, attack_start, external_pattern, **kwargs):
        """
        执行BadNets攻击训练
        
        Args:
            server: 联邦学习服务器
            round_num: 当前轮次
            attack_start: 攻击开始轮次
            external_pattern: 外部触发器模式
            **kwargs: 其他参数
            
        Returns:
            tuple: (trigger_pattern, external_pattern) 用于ASR评估
        """
        # 设置全局BadNets攻击实例，供客户端poisontest使用
        import sys
        from flcore.attacks import client_attack_utils
        client_attack_utils._global_badnets_attack = self
        
        client_data = []
        
        # 遍历所有选中的客户端
        for client in server.selected_clients:
            start_time = time.time()
            accuracy = 0.0
            poisonaccuracy = 0.0
            
            if client in server.malicious_clients:
                # 恶意客户端训练
                if round_num >= attack_start:
                    print(f"=== Malicious Client {client.id} - BadNets Attack (Round {round_num}) ===")
                    
                    # 加载投毒数据
                    trainloader = client.load_train_data()
                    poison_dataset = self._create_poison_dataset(
                        trainloader.dataset,
                        self.poisoned_rate,
                        self.target_label
                    )
                    
                    # 使用投毒数据训练
                    poison_loader = DataLoader(
                        poison_dataset,
                        batch_size=client.batch_size,
                        shuffle=True
                    )
                    
                    self._train_with_poison_data(client, poison_loader)
                    
                    # 评估
                    accuracy = client.evaluate()
                    poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=self.trigger_pattern,
                        poison_label=self.target_label,
                        pattern=external_pattern
                    )
                    poisonaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                    
                    print(f"Client {client.id} - ACC: {accuracy:.4f}, ASR: {poisonaccuracy:.4f}")
                else:
                    # 攻击开始前的正常训练
                    client.train(is_selected=True)
                    accuracy = client.evaluate()
                    
                    poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=self.trigger_pattern,
                        poison_label=self.target_label,
                        pattern=external_pattern
                    )
                    poisonaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poisonaccuracy,
                    'is_malicious': True,
                    'client_time': time.time() - start_time
                })
            else:
                # 正常客户端训练
                client.train(is_selected=True)
                accuracy = client.evaluate()
                
                poiosnacc_count, poiosnsumcount = client.poisontest(
                    trigger=self.trigger_pattern,
                    poison_label=self.target_label,
                    pattern=external_pattern
                )
                poisonaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poisonaccuracy,
                    'is_malicious': False,
                    'client_time': time.time() - start_time
                })
        
        # 记录攻击数据
        if client_data:
            server.base_attack.data_recorder.record_attack_data_from_collected(
                round_num=round_num,
                server=server,
                client_data=client_data,
                attack_method='badnets',
                original_trigger_list=self.trigger_pattern,
                external_pattern=external_pattern or []
            )
        
        return self.trigger_pattern, external_pattern or []
    
    def _create_poison_dataset(self, benign_dataset, poisoned_rate, target_label):
        """创建投毒数据集"""
        if self.dataset.lower() in ['mnist', 'fashionmnist']:
            trigger_transform = AddMNISTTrigger(self.trigger_pattern, self.trigger_weight)
        elif self.dataset.lower() in ['cifar10', 'cifar100']:
            trigger_transform = AddCIFAR10Trigger(self.trigger_pattern, self.trigger_weight)
        elif self.dataset.lower() in ['har', 'uci_har']:
            trigger_transform = AddHARTrigger(self.trigger_pattern, self.trigger_weight)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        return PoisonedDatasetWrapper(
            benign_dataset=benign_dataset,
            y_target=target_label,
            poisoned_rate=poisoned_rate,
            trigger_transform=trigger_transform
        )
    
    def _train_with_poison_data(self, client, poison_loader):
        """使用投毒数据训练客户端模型"""
        client.model.train()
        
        for epoch in range(client.local_epochs):
            for x, y in poison_loader:
                x = x.to(client.device)
                y = y.to(client.device)
                
                output = client.model(x)
                loss = client.loss(output, y)
                
                client.optimizer.zero_grad()
                loss.backward()
                client.optimizer.step()
    
    def _evaluate_asr_on_loader(self, model, test_loader, device):
        """在测试数据上评估ASR"""
        model.eval()
        total_samples = 0
        successful_attacks = 0
        
        # 将触发器模式和权重移动到正确的设备
        trigger_pattern_device = self.trigger_pattern.to(device)
        trigger_weight_device = self.trigger_weight.to(device)
        
        # 创建触发器转换器
        if self.dataset.lower() in ['mnist', 'fashionmnist']:
            trigger_transform = AddMNISTTrigger(trigger_pattern_device, trigger_weight_device)
        elif self.dataset.lower() in ['cifar10', 'cifar100']:
            trigger_transform = AddCIFAR10Trigger(trigger_pattern_device, trigger_weight_device)
        elif self.dataset.lower() in ['har', 'uci_har']:
            trigger_transform = AddHARTrigger(trigger_pattern_device, trigger_weight_device)
        else:
            return 0.0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                
                # 为每个样本添加触发器
                x_triggered = []
                for i in range(x.size(0)):
                    if self.dataset.lower() in ['har', 'uci_har']:
                        # HAR数据：直接处理张量
                        data_triggered = trigger_transform(x[i])
                        x_triggered.append(data_triggered)
                    else:
                        # 图像数据：转换为PIL图像处理
                        if x.shape[1] == 3:  # RGB
                            img_pil = transforms.ToPILImage()(x[i].cpu())
                        else:  # 灰度
                            img_pil = transforms.ToPILImage(mode='L')(x[i].cpu())
                        
                        # 添加触发器
                        img_triggered = trigger_transform(img_pil)
                        # 转换回张量
                        if not isinstance(img_triggered, torch.Tensor):
                            img_triggered = transforms.ToTensor()(img_triggered)
                        x_triggered.append(img_triggered.to(device))
                
                x_triggered = torch.stack(x_triggered).to(device)
                
                # 测试模型是否预测为目标标签
                output = model(x_triggered)
                pred = output.argmax(dim=1)
                successful_attacks += (pred == self.target_label).sum().item()
                total_samples += x.size(0)
        
        asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        return asr