import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .base_attack import BaseAttack
from .client_attack_utils import ClientAttackUtils
from .trigger_utils import TriggerUtils

class DBAAttack(BaseAttack):
    """
    Distributed Backdoor Attack (DBA) 实现
    
    核心思想：
    1. 将全局触发器分成四块，分配给4个恶意客户端
    2. 每个恶意客户端单独优化其中一块触发器
    3. 全局触发器保持不变，使用trigger_list[0]配置
    4. 训练函数与train_malicious一致
    """
    
    def __init__(self, args, device):
        super().__init__(args, device)
        self.trigger_utils = TriggerUtils()
        self.client_attack_utils = ClientAttackUtils(device=self.device, dataset=self.dataset)
        
        # DBA特定参数
        self.num_trigger_parts = 4  # 触发器分成4块
        self.trigger_parts = {}     # 存储每个恶意客户端的触发器部分
        self.optimized_trigger_parts = {}  # 存储优化后的触发器部分
        
    def divide_trigger_into_parts(self, global_trigger, malicious_clients):
        """
        将全局触发器分成4块，分配给4个恶意客户端
        
        Args:
            global_trigger: 全局触发器 (来自trigger_list[0])
            malicious_clients: 恶意客户端列表
            
        Returns:
            trigger_parts: 分配给每个客户端的触发器部分
        """
        if len(malicious_clients) < self.num_trigger_parts:
            print(f"Warning: 恶意客户端数量({len(malicious_clients)})少于触发器部分数量({self.num_trigger_parts})")
            # 如果恶意客户端少于4个，则重复分配
            while len(malicious_clients) < self.num_trigger_parts:
                malicious_clients.append(malicious_clients[-1])
        
        # 获取触发器形状
        trigger_shape = global_trigger.shape
        print(f"Global trigger shape: {trigger_shape}")
        
        # 根据数据集类型分割触发器
        if self.dataset.lower() in ["cifar10", "cifar100"]:
            # CIFAR: (3, 32, 32) -> 分成4个 (3, 8, 32)
            # 按高度分割，确保与main函数中的trigger_patten兼容
            part_height = trigger_shape[1] // self.num_trigger_parts
            trigger_parts = {}
            
            for i, client in enumerate(malicious_clients[:self.num_trigger_parts]):
                start_h = i * part_height
                end_h = (i + 1) * part_height if i < self.num_trigger_parts - 1 else trigger_shape[1]
                
                # 提取触发器部分
                trigger_part = global_trigger[:, start_h:end_h, :].clone()
                trigger_parts[client.id] = {
                    'trigger_part': trigger_part,
                    'global_trigger': global_trigger.clone(),
                    'part_coords': (start_h, end_h, 0, trigger_shape[2]),
                    'part_index': i
                }
                
        elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            # MNIST: (1, 28, 28) -> 分成4个 (1, 7, 28)
            part_height = trigger_shape[1] // self.num_trigger_parts
            trigger_parts = {}
            
            for i, client in enumerate(malicious_clients[:self.num_trigger_parts]):
                start_h = i * part_height
                end_h = (i + 1) * part_height if i < self.num_trigger_parts - 1 else trigger_shape[1]
                
                trigger_part = global_trigger[:, start_h:end_h, :].clone()
                trigger_parts[client.id] = {
                    'trigger_part': trigger_part,
                    'global_trigger': global_trigger.clone(),
                    'part_coords': (start_h, end_h, 0, trigger_shape[2]),
                    'part_index': i
                }
                
        elif self.dataset.lower() in ["har", "uci_har"]:
            # HAR: (9, 1, 128) -> 分成4个 (9, 1, 32)
            part_width = trigger_shape[2] // self.num_trigger_parts
            trigger_parts = {}
            
            for i, client in enumerate(malicious_clients[:self.num_trigger_parts]):
                start_w = i * part_width
                end_w = (i + 1) * part_width if i < self.num_trigger_parts - 1 else trigger_shape[2]
                
                trigger_part = global_trigger[:, :, start_w:end_w].clone()
                trigger_parts[client.id] = {
                    'trigger_part': trigger_part,
                    'global_trigger': global_trigger.clone(),
                    'part_coords': (0, trigger_shape[1], start_w, end_w),
                    'part_index': i
                }
                
        else:
            raise ValueError(f"Unsupported dataset for DBA: {self.dataset}")
        
        self.trigger_parts = trigger_parts
        print(f"DBA: 触发器已分成{self.num_trigger_parts}块，分配给{len(malicious_clients)}个恶意客户端")
        
        return trigger_parts
    
    def optimize_trigger_part(self, client, trigger_part_info, pattern, target_label, num_epochs=50):
        """
        优化单个客户端的触发器部分 - 修正版本
        
        核心改进：
        1. 直接优化触发器部分，而不是在全局触发器上下文中优化
        2. 使用触发器部分创建局部触发器进行优化
        3. 确保优化目标明确：让触发器部分激活目标标签
        
        Args:
            client: 恶意客户端
            trigger_part_info: 触发器部分信息
            pattern: 触发器模式
            target_label: 目标标签
            num_epochs: 优化轮数
            
        Returns:
            optimized_part: 优化后的触发器部分
        """
        device = self.device
        trigger_part = trigger_part_info['trigger_part'].clone().to(device)
        part_coords = trigger_part_info['part_coords']
        
        # 确保触发器部分可以计算梯度
        trigger_part = trigger_part.detach().clone().requires_grad_(True)
        
        # 创建优化器 - 使用更小的学习率进行精细优化
        optimizer = torch.optim.Adam([trigger_part], lr=0.005)
        
        # 获取客户端模型
        model = client.model.to(device)
        model.eval()
        
        # 准备训练数据
        trainloader = client.load_train_data()
        data_iter = iter(trainloader)
        
        def get_batch():
            nonlocal data_iter
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(trainloader)
                x, y = next(data_iter)
            return x.to(device), y.to(device)
        
        # 创建模式掩码 - 只针对触发器部分
        pattern_mask = self._create_partial_pattern_mask(pattern, part_coords, device)
        
        # 优化触发器部分
        best_loss = float('inf')
        best_trigger_part = trigger_part.clone()
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # 获取批次数据
            x, y = get_batch()
            batch_size = x.size(0)
            
            # 创建目标标签（全部设为target_label）
            y_target = torch.full((batch_size,), target_label, dtype=torch.long, device=device)
            
            # 直接使用触发器部分创建局部触发器
            # 将触发器部分扩展到完整尺寸，其他区域保持为0
            full_trigger = self._expand_trigger_part(trigger_part, part_coords, device)
            
            # 应用触发器到输入数据
            x_triggered = []
            for i in range(batch_size):
                triggered_img = self.client_attack_utils.add_pixel_pattern(
                    x[i], full_trigger, pattern, pattern_mask, 
                    device, self.dataset, original_label=int(y[i].item())
                )
                x_triggered.append(triggered_img)
            x_triggered = torch.stack(x_triggered, dim=0)
            
            # 前向传播
            output = model(x_triggered)
            loss = nn.functional.cross_entropy(output, y_target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 限制触发器值范围
            with torch.no_grad():
                trigger_part.clamp_(-1, 1)
            
            # 保存最佳触发器部分
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_trigger_part = trigger_part.clone()
            
            if epoch % 10 == 0:
                print(f"Client {client.id} - Epoch {epoch}, Loss: {loss.item():.4f}, Best Loss: {best_loss:.4f}")
        
        print(f"Client {client.id} - 优化完成，最佳损失: {best_loss:.4f}")
        return best_trigger_part.detach()
    
    def _create_partial_pattern_mask(self, pattern, part_coords, device):
        """为触发器部分创建模式掩码"""
        if self.dataset.lower() in ["cifar10", "cifar100"]:
            pattern_mask = torch.ones((3, 32, 32), device=device)
        elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            pattern_mask = torch.ones((1, 28, 28), device=device)
        elif self.dataset.lower() in ["har", "uci_har"]:
            pattern_mask = torch.ones((9, 1, 128), device=device)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        # 只对触发器部分区域应用模式
        start_h, end_h, start_w, end_w = part_coords
        for pos in pattern:
            if self.dataset.lower() in ["cifar10", "cifar100"]:
                if start_h <= pos[0] < end_h and start_w <= pos[1] < end_w:
                    pattern_mask[:, pos[0], pos[1]] = 0
            elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
                if start_h <= pos[0] < end_h and start_w <= pos[1] < end_w:
                    pattern_mask[0, pos[0], pos[1]] = 0
            elif self.dataset.lower() in ["har", "uci_har"]:
                if len(pos) >= 3 and start_w <= pos[2] < end_w:
                    pattern_mask[pos[0], pos[1], pos[2]] = 0
        
        return pattern_mask
    
    def _expand_trigger_part(self, trigger_part, part_coords, device):
        """将触发器部分扩展到完整尺寸"""
        start_h, end_h, start_w, end_w = part_coords
        
        if self.dataset.lower() in ["cifar10", "cifar100"]:
            full_trigger = torch.zeros((3, 32, 32), device=device)
            full_trigger[:, start_h:end_h, :] = trigger_part
        elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            full_trigger = torch.zeros((1, 28, 28), device=device)
            full_trigger[:, start_h:end_h, :] = trigger_part
        elif self.dataset.lower() in ["har", "uci_har"]:
            full_trigger = torch.zeros((9, 1, 128), device=device)
            full_trigger[:, :, start_w:end_w] = trigger_part
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        return full_trigger
    
    def _create_client_trigger_list(self, client, global_trigger_list, malicious_clients):
        """
        为每个客户端创建个性化的触发器列表
        
        核心思想：
        1. 如果客户端有自己的优化触发器部分，使用它创建个性化触发器
        2. 否则使用全局触发器
        
        Args:
            client: 当前客户端
            global_trigger_list: 全局触发器列表
            malicious_clients: 恶意客户端列表
            
        Returns:
            client_trigger_list: 客户端个性化的触发器列表
        """
        if client.id in self.optimized_trigger_parts:
            # 该客户端有自己的优化触发器部分
            optimized_part = self.optimized_trigger_parts[client.id]
            part_coords = self.trigger_parts[client.id]['part_coords']
            
            # 创建个性化触发器：将优化后的部分合并到全局触发器中
            personalized_trigger = global_trigger_list[0].clone()
            start_h, end_h, start_w, end_w = part_coords
            
            if self.dataset.lower() in ["cifar10", "cifar100"]:
                personalized_trigger[:, start_h:end_h, :] = optimized_part
            elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
                personalized_trigger[:, start_h:end_h, :] = optimized_part
            elif self.dataset.lower() in ["har", "uci_har"]:
                personalized_trigger[:, :, start_w:end_w] = optimized_part
            
            # 创建个性化触发器列表
            client_trigger_list = [personalized_trigger] + global_trigger_list[1:]
            print(f"DBA: 客户端 {client.id} 使用个性化触发器进行训练")
            
        else:
            # 该客户端没有优化触发器部分，使用全局触发器
            client_trigger_list = global_trigger_list
            print(f"DBA: 客户端 {client.id} 使用全局触发器进行训练")
        
        return client_trigger_list
    
    def merge_optimized_trigger_parts(self, malicious_clients):
        """
        合并所有优化后的触发器部分，形成最终的全局触发器
        
        Args:
            malicious_clients: 恶意客户端列表
            
        Returns:
            final_trigger: 合并后的全局触发器
        """
        if not self.optimized_trigger_parts:
            print("Warning: 没有优化后的触发器部分可合并")
            return None
        
        # 获取第一个客户端的全局触发器作为基础
        first_client = malicious_clients[0]
        if first_client.id not in self.trigger_parts:
            print("Error: 找不到第一个客户端的触发器信息")
            return None
        
        base_trigger = self.trigger_parts[first_client.id]['global_trigger'].clone()
        
        # 合并所有优化后的触发器部分
        for client in malicious_clients[:self.num_trigger_parts]:
            if client.id in self.optimized_trigger_parts:
                part_coords = self.trigger_parts[client.id]['part_coords']
                optimized_part = self.optimized_trigger_parts[client.id]
                
                start_h, end_h, start_w, end_w = part_coords
                
                if self.dataset.lower() in ["cifar10", "cifar100"]:
                    base_trigger[:, start_h:end_h, :] = optimized_part
                elif self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
                    base_trigger[:, start_h:end_h, :] = optimized_part
                elif self.dataset.lower() in ["har", "uci_har"]:
                    base_trigger[:, :, start_w:end_w] = optimized_part
        
        print("DBA: 触发器部分已合并为最终全局触发器")
        print(f"DBA: 最终触发器形状: {base_trigger.shape}, 设备: {base_trigger.device}")
        return base_trigger
    
    def execute_attack_training(self, server, round_num, attack_start, oneshot, clip_rate, 
                              original_trigger_list, external_pattern, optimized_trigger_list):
        """
        执行DBA攻击训练
        
        Args:
            server: 服务器实例
            round_num: 当前轮次
            attack_start: 攻击开始轮次
            oneshot: 是否一次性攻击
            clip_rate: 梯度裁剪率
            original_trigger_list: 原始触发器列表
            external_pattern: 外部模式
            optimized_trigger_list: 优化后的触发器列表
            
        Returns:
            updated_trigger_list: 更新后的触发器列表
        """
        local_optimized_trigger_list = copy.deepcopy(optimized_trigger_list)
        local_external_pattern = copy.deepcopy(external_pattern)
        
        # 处理恶意客户端的触发器优化（仅在攻击开始后）
        if round_num >= attack_start:
            # 获取恶意客户端
            malicious_clients = server.malicious_clients
            if len(malicious_clients) > 0:
                # 使用trigger_list[0]作为全局触发器
                global_trigger = original_trigger_list[0].clone()
                print(f"DBA: 使用trigger_list[0]，形状: {global_trigger.shape}, 设备: {global_trigger.device}")
                
                # 如果是第一轮攻击，初始化触发器分割
                if round_num == attack_start:
                    self.divide_trigger_into_parts(global_trigger, malicious_clients)
                
                # 每个恶意客户端优化其触发器部分
                for client in malicious_clients[:self.num_trigger_parts]:
                    if client.id in self.trigger_parts:
                        print(f"DBA: 客户端 {client.id} 开始优化触发器部分")
                        
                        optimized_part = self.optimize_trigger_part(
                            client=client,
                            trigger_part_info=self.trigger_parts[client.id],
                            pattern=external_pattern,
                            target_label=server.poisonlabel,
                            num_epochs=50  # 增加优化轮数以提高效果
                        )
                        
                        self.optimized_trigger_parts[client.id] = optimized_part
                
                # 合并优化后的触发器部分
                final_trigger = self.merge_optimized_trigger_parts(malicious_clients)
                
                if final_trigger is not None:
                    # 更新触发器列表
                    local_optimized_trigger_list[0] = final_trigger
                    print(f"DBA: 第{round_num}轮攻击完成，触发器已更新")
            else:
                print("Warning: 没有恶意客户端进行DBA攻击")
        
        # 初始化数据收集 - 每一轮都收集数据
        client_data = []
        
        # 训练所有客户端（包括恶意客户端和正常客户端）
        for client in server.alled_clients:
            if client in server.malicious_clients:
                # 恶意客户端训练
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("*********************************************")
                        print("DBA: 开始后门攻击训练!")
                        
                        # 为每个客户端创建个性化的触发器
                        client_trigger_list = self._create_client_trigger_list(
                            client, local_optimized_trigger_list, malicious_clients
                        )
                        
                        # 恶意客户端使用自己优化的触发器部分进行训练
                        client.train_malicious_dba(
                            is_selected=True,
                            poison_ratio=server.poisonratio,
                            poison_label=server.poisonlabel,
                            trigger=client_trigger_list,
                            pattern=external_pattern,
                            oneshot=oneshot,
                            clip_rate=clip_rate
                        )
                        
                        accuracy = client.evaluate()
                        
                        poiosnacc_count, poiosnsumcount = client.poisontest(
                            trigger=local_optimized_trigger_list,
                            poison_label=server.poisonlabel, 
                            pattern=external_pattern
                        )
                        poisonaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                        
                        print("*******************DBA攻击成功率、主任务准确度 & 恶意客户端******************")
                        print(f'Client {client.id} ASR :{poisonaccuracy:.4f}')
                        print(f'Client {client.id} ACC :{accuracy:.4f}')
                        
                else:
                    # 攻击开始前的正常训练
                    client.train(client.id in [c.id for c in server.selected_clients])
                    accuracy = client.evaluate()
                    
                    poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=local_optimized_trigger_list,
                        poison_label=server.poisonlabel, 
                        pattern=external_pattern
                    )
                    poisonaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                    
                    print("**************DBA攻击成功率、主任务准确度 & 恶意客户端（攻击前）********************")
                    print(f'Client {client.id} ASR :{poisonaccuracy:.4f}')
                    print(f'Client {client.id} ACC :{accuracy:.4f}')

                # 收集恶意客户端数据（无论是否在攻击轮次）
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poisonaccuracy,
                    'is_malicious': True,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })

            else:
                # 正常客户端训练
                client.train(client.id in [c.id for c in server.selected_clients])
                accuracy = client.evaluate()
                
                poiosnacc_count, poiosnsumcount = client.poisontest(
                    trigger=local_optimized_trigger_list,
                    poison_label=server.poisonlabel, 
                    pattern=external_pattern
                )
                poiosnaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0

                # 收集正常客户端数据
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poiosnaccuracy,
                    'is_malicious': False,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })

                print("====================DBA攻击成功率、主任务准确度 & 正常客户端==============================")
                print(f'Client {client.id} ASR :{poiosnaccuracy:.4f}')
                print(f'Client {client.id} ACC :{accuracy:.4f}')
        
        # 每一轮都记录数据 - 使用server的base_attack.data_recorder
        if client_data:  # 确保有数据才记录
            server.base_attack.data_recorder.record_attack_data_from_collected(
                round_num=round_num,
                server=server,
                client_data=client_data,
                attack_method='dba',
                original_trigger_list=local_optimized_trigger_list,
                external_pattern=external_pattern
            )
        
        return local_optimized_trigger_list