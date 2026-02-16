import torch
import torch.nn as nn
import copy
import random
import numpy as np
import time
import os
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable


class Autoencoder(nn.Module):
    def __init__(self, in_channels: int):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class BaseAttack:
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.poison_label = 1  # 默认中毒标签
        self.poison_ratio = args.poison_rate
        self.attack_method = args.attack
        random.seed(args.seed_ran)
        self.local_rng = random.Random(args.seed_ran)

        # 初始化统一的数据记录模块
        self.data_recorder = DataRecorder(args, device, self.attack_method)
        
    def select_malicious_clients(self, clients, num_malicious=2):
        """随机选择指定数量的恶意客户端"""
        malicious_clients = self.local_rng.sample(clients, num_malicious)
        
        print("=====================Malicious client=================================")
        print("Selected malicious client IDs:", [client.id for client in malicious_clients])
        print("=============================================================")
        return malicious_clients

    def select_malicious_clients_by_ids(self, clients, malicious_ids=None):
        """根据指定的客户端ID选择恶意客户端，默认选择客户端3、9、17、18"""
        if malicious_ids is None:
            malicious_ids = [3, 9, 17, 18]  # 默认恶意客户端ID
        
        malicious_clients = [client for client in clients if client.id in malicious_ids]
        print("=====================Malicious client=================================")
        print("Selected malicious client IDs:", [client.id for client in malicious_clients])
        print("=============================================================")
        return malicious_clients
    
    def create_pattern_mask(self, pattern, dataset):
        """take it literally"""
        if dataset.lower() in ["cifar10", "cifar100"]:
            pattern_mask = torch.ones((3, 32, 32), device=self.device)
            for pos in pattern:
                pattern_mask[:, pos[0], pos[1]] = 0
        elif dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            pattern_mask = torch.ones((1, 28, 28), device=self.device)
            for pos in pattern:
                pattern_mask[0, pos[0], pos[1]] = 0
        elif dataset.lower() == "iot":
            pattern_mask = torch.ones(115, device=self.device)
            pattern_mask[pattern] = 0
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return pattern_mask
    
    def add_trigger_to_image(self, image, noise_trigger, pattern, pattern_mask):
        """til"""
        image = copy.deepcopy(image).to(self.device)
        
        if self.dataset.lower() in ["mnist", "fmnist", "fashionmnist"]:
            image = image * pattern_mask
            image = image + noise_trigger
            image = torch.clamp(image, -1, 1)
        elif self.dataset.lower() in ["cifar10", "cifar100"]:
            image = image * pattern_mask
            image = image + noise_trigger
            image = torch.clamp(image, -1, 1)
        elif self.dataset.lower() == "iot":
            image = image * pattern_mask
            image = image + noise_trigger
            
        return image
    
    def evaluate_asr(self, clients, trigger, pattern, poison_label):
        """Evaluate the success rate of the attack(ASR)"""
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
        print("Global Attack Success Rate (ASR): {:.4f}".format(global_asr))
        return global_asr
    
    def match_l2_loss(self, grad_mal, grad_nor):
        """Calculate the L2 gradient matching loss."""
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(grad_nor)):
            gw_real_vec.append(grad_nor[ig].reshape((-1)))
            gw_syn_vec.append(grad_mal[ig].reshape((-1)))

        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)

        dis = torch.sqrt(torch.sum((gw_syn_vec - gw_real_vec) ** 2))
        return dis
    
    def save_experiment_data(self):
        """保存实验数据 - 所有攻击方法的统一接口"""
        self.data_recorder.save_data()
        
        # 打印实验摘要
        summary = self.data_recorder.get_experiment_summary()
        print(f"\n{self.attack_method.upper()}实验摘要:")
        print(f"算法: {summary.get('algorithm', 'N/A')}")
        print(f"攻击: {summary.get('attack', 'N/A')}")
        print(f"数据集: {summary.get('dataset', 'N/A')}")
        print(f"总轮数: {summary.get('total_rounds', 0)}")
        print(f"最终全局ACC: {summary.get('final_global_acc', 0.0):.4f}")
        print(f"最终全局ASR: {summary.get('final_global_asr', 0.0):.4f}")
        print(f"最终客户端ACC均值: {summary.get('final_client_acc_mean', 0.0):.4f}")
        print(f"最终客户端ASR均值: {summary.get('final_client_asr_mean', 0.0):.4f}")
        print(f"总训练时间: {summary.get('total_time', 0.0):.2f}秒") 

def get_shared_params(model):
    """
    获取共享参数名称
    
    Args:
        model: 模型
    
    Returns:
        set: 共享参数名称集合
    """
    # 直接返回所有参数名称，让select_backdoor_params中的is_personal_layer来过滤
    return {name for name, _ in model.named_parameters()}


def get_pattern_info(pattern):
    """
    根据pattern计算T1和T2的位置信息
    
    Args:
        pattern: 原始pattern位置列表
        
    Returns:
        dict: 包含T1和T2位置信息的字典
    """
    if not pattern:
        return None
        
    # 将pattern中的列表转换为元组，确保类型一致性
    pattern = [tuple(pos) if isinstance(pos, list) else pos for pos in pattern]
        
    # 计算pattern的边界
    rows = [pos[0] for pos in pattern]
    cols = [pos[1] for pos in pattern]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    # 计算中心位置
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    
    # T1尺寸为pattern的1/4
    pattern_height = max_row - min_row + 1
    pattern_width = max_col - min_col + 1
    t1_height = max(1, pattern_height // 2)  # 至少1个像素
    t1_width = max(1, pattern_width // 2)    # 至少1个像素
    
    # T1位置（中心区域）
    t1_start_row = center_row - t1_height // 2
    t1_end_row = center_row + t1_height // 2
    t1_start_col = center_col - t1_width // 2
    t1_end_col = center_col + t1_width // 2
    
    # T2位置（pattern中除了T1的部分）
    t1_positions = set()
    for i in range(t1_start_row, t1_end_row + 1):
        for j in range(t1_start_col, t1_end_col + 1):
            t1_positions.add((i, j))
    
    t2_positions = [pos for pos in pattern if pos not in t1_positions]
    
    return {
        't1_positions': list(t1_positions),
        't2_positions': t2_positions,
        't1_center': (center_row, center_col),
        't1_size': (t1_height, t1_width),
        'pattern_size': (pattern_height, pattern_width)
    }


class DataRecorder:
    """统一的数据记录模块，用于记录所有攻击方法的实验数据"""
    
    def __init__(self, args, device, attack_method):
        self.args = args
        self.device = device
        self.algorithm = getattr(args, 'algorithm', 'FedAvg')
        self.attack = attack_method
        self.dataset = getattr(args, 'dataset', 'MNIST')
        self.defense_mode = getattr(args, 'defense_mode', None)
        
        # 创建记录文件夹
        self.record_dir = self._create_record_directory()
        
        # 初始化数据存储
        self.round_data = []
        self.client_data = []
        
    def _create_record_directory(self):
        """创建数据记录文件夹"""
        # 获取当前工作目录
        current_dir = os.getcwd()
        
        # 创建LabRecord文件夹
        lab_record_dir = os.path.join(current_dir, "LabRecord")
        if not os.path.exists(lab_record_dir):
            os.makedirs(lab_record_dir)
            print(f"创建LabRecord目录: {lab_record_dir}")
        
        # 创建实验文件夹，包含防御方法信息
        # 格式: Algorithm_Attack_Defense_Dataset
        # 例如: FedAvg_badnets_krum_FashionMNIST 或 FedAvg_badnets_none_HAR
        defense_str = self.defense_mode if self.defense_mode else 'none'
        experiment_name = f"{self.algorithm}_{self.attack}_{defense_str}_{self.dataset}"
        experiment_dir = os.path.join(lab_record_dir, experiment_name)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            print(f"创建实验目录: {experiment_dir}")
        
        print(f"{self.attack.upper()}数据记录文件夹已创建: {experiment_dir}")
        return experiment_dir
    
    def record_attack_data_from_collected(self, round_num, server, client_data, attack_method, 
                                         original_trigger_list, external_pattern):
        """使用预收集的客户端数据记录攻击数据"""
        # 计算统计指标
        client_accs = [data['client_acc'] for data in client_data]
        client_asrs = [data['client_asr'] for data in client_data]
        client_times = [data['client_time'] for data in client_data]
        
        client_acc_mean = np.mean(client_accs) if client_accs else 0.0
        client_acc_std = np.std(client_accs) if client_accs else 0.0
        client_asr_mean = np.mean(client_asrs) if client_asrs else 0.0
        client_asr_std = np.std(client_asrs) if client_asrs else 0.0
        
        # 获取全局指标
        global_acc = server.rs_test_acc[-1] if server.rs_test_acc else 0.0
        global_asr = self._calculate_global_asr(server, original_trigger_list, external_pattern)
        round_time = server.Budget[-1] if server.Budget else 0.0
        
        # 记录轮次数据
        round_data = {
            'round': round_num,
            'global_acc': global_acc,
            'global_asr': global_asr,
            'round_time': round_time,
            'client_acc_mean': client_acc_mean,
            'client_acc_std': client_acc_std,
            'client_asr_mean': client_asr_mean,
            'client_asr_std': client_asr_std,
            'attack_method': attack_method,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.round_data.append(round_data)
        
        # 记录每个客户端的详细数据
        for data in client_data:
            client_detail = {
                'round': round_num,
                'client_id': data['client_id'],
                'client_acc': data['client_acc'],
                'client_asr': data['client_asr'],
                'client_time': data['client_time'],
                'is_malicious': data['is_malicious'],
                'attack_method': attack_method,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self.client_data.append(client_detail)
        
        print(f"{self.attack.upper()}第{round_num}轮数据已记录 - 客户端ACC均值: {client_acc_mean:.4f}, ASR均值: {client_asr_mean:.4f}")
    
    def _calculate_global_asr(self, server, trigger, pattern):
        """计算全局ASR"""
        try:
            return server.evaluate_asr(trigger=trigger, pattern=pattern)
        except:
            return 0.0
    
    def save_data(self):
        """保存数据到CSV文件"""
        try:
            # 保存轮次数据
            round_file = os.path.join(self.record_dir, "round_data.csv")
            if self.round_data:
                with open(round_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.round_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.round_data)
                print(f"轮次数据已保存到: {round_file}")
            else:
                print("警告: 没有轮次数据可保存")
            
            # 保存客户端详细数据
            client_file = os.path.join(self.record_dir, "client_data.csv")
            if self.client_data:
                with open(client_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.client_data[0].keys())
                    writer.writeheader()
                    writer.writerows(self.client_data)
                print(f"客户端数据已保存到: {client_file}")
            else:
                print("警告: 没有客户端数据可保存")
            
            print(f"{self.attack.upper()}实验数据已保存到: {self.record_dir}")
            
        except Exception as e:
            print(f"保存数据时发生错误: {e}")
            print(f"记录目录: {self.record_dir}")
            print(f"轮次数据数量: {len(self.round_data)}")
            print(f"客户端数据数量: {len(self.client_data)}")
    
    def get_experiment_summary(self):
        """获取实验摘要"""
        if not self.round_data:
            return {}
        
        final_round = self.round_data[-1]
        return {
            'algorithm': self.algorithm,
            'attack': self.attack,
            'dataset': self.dataset,
            'total_rounds': len(self.round_data),
            'final_global_acc': final_round['global_acc'],
            'final_global_asr': final_round['global_asr'],
            'final_client_acc_mean': final_round['client_acc_mean'],
            'final_client_asr_mean': final_round['client_asr_mean'],
            'total_time': sum(round_data['round_time'] for round_data in self.round_data)
        }