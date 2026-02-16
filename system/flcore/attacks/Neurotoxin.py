import torch
import torch.nn as nn
import copy
import numpy as np
import os
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .base_attack import BaseAttack


class NeurotoxinAttack(BaseAttack):
    """
    Neurotoxin攻击：
    在常规基线后门攻击的本地更新步骤中，
    计算上轮（或当前）观察到的"良性梯度" 
    → 选出 top-k 重坐标 
    → 将本次恶意梯度投影到其补集（bottom 部分）
    → 做 PGD/更新并上传，
    即只修改良性设备不常触及的参数，从而提高后门在后续多轮训练中的存活时间。
    """
    
    def __init__(self, args, device):
        super().__init__(args, device)
        self.poison_ratio = getattr(args, 'poison_rate', 0.2)
        self.grad_mask_ratio = getattr(args, 'grad_mask_ratio', 0.5)  # 梯度掩码比例
        self.aggregate_all_layer = getattr(args, 'aggregate_all_layer', 1)  # 是否聚合所有层
        
    '''=================Neurotoxin源代码======================='''
    def grad_mask_cv(self, model, dataset_clean, criterion, ratio=0.5):
        """
        Generate a gradient mask based on the given dataset
        计算良性梯度，选出top-k重坐标，生成梯度掩码
        """
        model.train()
        model.zero_grad()

        # 计算良性梯度
        for participant_id in range(len(dataset_clean)):
            train_data = dataset_clean[participant_id]
            
            for inputs, labels in train_data:
                if isinstance(inputs, list):
                    inputs = inputs[0]  # 处理多通道输入
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward(retain_graph=True)
                
                # 只处理一个批次就足够计算梯度掩码
                break
            break

        mask_grad_list = []
        if self.aggregate_all_layer == 1:
            grad_list = []
            grad_abs_sum_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad and parms.grad is not None:
                    grad_list.append(parms.grad.abs().view(-1))
                    grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
                    k_layer += 1

            grad_list = torch.cat(grad_list).to(self.device)
            # 修正：选择更新幅度最小的参数（良性梯度不常触及的参数）
            _, indices = torch.topk(grad_list, int(len(grad_list)*ratio))
            mask_flat_all_layer = torch.zeros(len(grad_list)).to(self.device)
            mask_flat_all_layer[indices] = 1.0

            count = 0
            percentage_mask_list = []
            k_layer = 0
            grad_abs_percentage_list = []
            for _, parms in model.named_parameters():
                if parms.requires_grad and parms.grad is not None:
                    gradients_length = len(parms.grad.abs().view(-1))
                    mask_flat = mask_flat_all_layer[count:count + gradients_length].to(self.device)
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(self.device))
                    count += gradients_length

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                    percentage_mask_list.append(percentage_mask1)

                    grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

                    k_layer += 1
        else:
            grad_abs_percentage_list = []
            grad_res = []
            l2_norm_list = []
            sum_grad_layer = 0.0
            for _, parms in model.named_parameters():
                if parms.requires_grad and parms.grad is not None:
                    grad_res.append(parms.grad.view(-1))
                    l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().to(self.device))/float(len(parms.grad.view(-1)))
                    l2_norm_list.append(l2_norm_l)
                    sum_grad_layer += l2_norm_l.item()

            grad_flat = torch.cat(grad_res)

            percentage_mask_list = []
            k_layer = 0
            for _, parms in model.named_parameters():
                if parms.requires_grad and parms.grad is not None:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    if ratio == 1.0:
                        _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                    else:
                        ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                        _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(self.device))

                    percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
                    percentage_mask_list.append(percentage_mask1)
                    k_layer += 1

        model.zero_grad()
        return mask_grad_list

    '''========================================================'''

    def execute_attack_training(self, server, round_num, attack_start, oneshot, clip_rate, 
                               original_trigger_list, external_pattern, optimized_trigger_list):
        """
        Neurotoxin攻击训练执行函数
        
        Args:
            server: 服务器实例
            round_num: 当前轮次
            attack_start: 攻击开始轮次
            oneshot: 是否一次性攻击
            clip_rate: 梯度裁剪率
            original_trigger_list: 原始触发器列表
            external_pattern: 外部模式
            optimized_trigger_list: 优化后的触发器列表（Neurotoxin使用固定触发器）
            
        Returns:
            optimized_trigger_list: 更新后的触发器列表（保持形式一致性）
        """
        # Neurotoxin使用固定触发器，不需要优化
        # 保持触发器列表的形式一致性，但实际使用原始触发器
        if round_num >= attack_start:
            print("Neurotoxin: 使用固定触发器进行攻击...")
            # 确保触发器不需要梯度
            for x in range(len(optimized_trigger_list)):  
                optimized_trigger_list[x].requires_grad = False
        
        # 初始化数据收集 - 每一轮都收集数据
        client_data = []
        
        # 恶意客户端训练
        for client in server.alled_clients:
            if client in server.malicious_clients:
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("*********************************************")
                        print("Neurotoxin: 执行后门攻击!")

                        # 生成梯度掩码
                        print("Neurotoxin: 生成梯度掩码...")
                        grad_mask = self.grad_mask_cv(
                            model=client.model,
                            dataset_clean=[client.load_train_data()],
                            criterion=client.loss,
                            ratio=self.grad_mask_ratio
                        )

                        # 使用原始触发器进行攻击
                        client.train_malicious_neurotoxin(
                            is_selected=True,  # 既然通过了上面的条件判断，这里直接设为True
                            poison_ratio=server.poisonratio, 
                            poison_label=server.poisonlabel,
                            trigger=original_trigger_list,  # 使用原始触发器
                            pattern=external_pattern, 
                            oneshot=oneshot,
                            clip_rate=clip_rate,
                            grad_mask=grad_mask
                        )
                        accuracy = client.evaluate()
                        
                        poiosnacc_count, poiosnsumcount = client.poisontest(trigger=original_trigger_list,
                                                                        poison_label=server.poisonlabel, 
                                                                        pattern=external_pattern)
                        poiosnaccuracy = poiosnacc_count / poiosnsumcount
                        
                        print("*******************Neurotoxin攻击成功率 & 准确度**********************")
                        print(f'Client {client.id} ASR :{poiosnaccuracy}') 
                        print(f'Client {client.id} ACC :{accuracy}')

                else:
                    # 攻击开始前的正常训练
                    client.train(client.id in [c.id for c in server.selected_clients])  
                    accuracy = client.evaluate()
                    poiosnacc_count, poiosnsumcount = client.poisontest(trigger=original_trigger_list,
                                                                    poison_label=server.poisonlabel, 
                                                                    pattern=external_pattern)
                    poiosnaccuracy = poiosnacc_count / poiosnsumcount
                    
                    print("***********************ASR & ACC----malicious client (pre-attack)**********************")
                    print(f'Client {client.id} ASR :{poiosnaccuracy}')
                    print(f'Client {client.id} ACC :{accuracy}')

                # 收集恶意客户端数据（无论是否在攻击轮次）
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poiosnaccuracy,
                    'is_malicious': True,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })

            else:  
                # 正常客户端训练
                client.train(client.id in [c.id for c in server.selected_clients])  
                accuracy = client.evaluate()
                poiosnacc_count, poiosnsumcount = client.poisontest(trigger=original_trigger_list,
                                                                poison_label=server.poisonlabel, 
                                                                pattern=external_pattern)
                poiosnaccuracy = poiosnacc_count / poiosnsumcount
                
                # 收集正常客户端数据
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poiosnaccuracy,
                    'is_malicious': False,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })
                
                print("***********************ASR & ACC----normal client**********************")
                print(f'Client {client.id} ASR :{poiosnaccuracy}')
                print(f'Client {client.id} ACC :{accuracy}')
        
        # 每一轮都记录数据 - 使用server的base_attack.data_recorder
        if client_data:  # 确保有数据才记录
            server.base_attack.data_recorder.record_attack_data_from_collected(
                round_num=round_num,
                server=server,
                client_data=client_data,
                attack_method='neurotoxin',
                original_trigger_list=original_trigger_list,
                external_pattern=external_pattern
            )
        
        return optimized_trigger_list