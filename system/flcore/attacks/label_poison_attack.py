import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .base_attack import BaseAttack
from .client_attack_utils import ClientAttackUtils
from .trigger_utils import TriggerUtils

class Label_Poison_Attack(BaseAttack):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.poison_ratio=getattr(args,'poison_rate',4)
        
        self.ce_loss=nn.CrossEntropyLoss()
        
        
    def poison_batch(self,labels, num_classes, poison_ratio):
        if poison_ratio is None:
            poison_ratio = self.poison_ratio

        labels = labels.to(self.device)

        mask = (torch.rand(labels.size(0), device=labels.device) <= poison_ratio)
        if mask.sum().item() == 0:
            return labels

        random_labels = torch.randint(
            low=0,
            high=num_classes,
            size=labels.size(),
            device=labels.device
        )

        random_labels = torch.where(random_labels == labels, 
                                    (random_labels + 1) % num_classes, 
                                    random_labels)

        poisoned_labels = torch.where(mask, random_labels, labels)

        return poisoned_labels

        
    
    def execute_attack_training(self, server, round_num, attack_start, oneshot, clip_rate, 
                              original_trigger_list, external_pattern, optimized_trigger_list):
        client_data=[]
        
        for client in server.alled_clients:
            if client in server.malicious_clients:
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("label_poison_attack: 恶意客户端参与训练...")
                    # print(client.id in [c.id for c in server.selected_clients])
                    client.train_malicious_label_poison(
                        is_selected=client.id in [c.id for c in server.selected_clients],
                        poison_ratio=server.poisonratio,
                        poison_label=server.poisonlabel,
                        trigger=None,
                        pattern=None,
                        oneshot=0,
                        clip_rate=clip_rate
                    )
                    # print(11111111111111)
                    accuracy=client.evaluate()

                    print(f'Client {client.id} ACC :{accuracy}')
                    
                else:
                    client.train(client.id in [c.id for c in server.selected_clients])
                    accuracy = client.evaluate()
                    
                    print(f'Client {client.id} ACC :{accuracy} (pre-attack)')

                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': 0,
                    'is_malicious': True,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })
                
            else:
                client.train(client.id in [c.id for c in server.selected_clients])
                accuracy = client.evaluate()
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': 0,
                    'is_malicious': False,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })
                

                print(f'Client {client.id} ACC :{accuracy}')
                
        