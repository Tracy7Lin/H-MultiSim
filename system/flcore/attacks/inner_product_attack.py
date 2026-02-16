import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .base_attack import BaseAttack
from .client_attack_utils import ClientAttackUtils
from .trigger_utils import TriggerUtils

class Inner_Product_Attack(BaseAttack):
    def __init__(self, args, device):
        super().__init__(args, device)
        
    def execute_attack_training(self, server, round_num, attack_start, oneshot, clip_rate, 
                              original_trigger_list, external_pattern, optimized_trigger_list):
        client_data=[]
        
        for client in server.alled_clients:
            if client in server.malicious_clients:
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("inner_product_attack: 恶意客户端参与训练...")
                        
                    client.train_malicious_inner_product(
                        is_selected=client.id in [c.id for c in server.selected_clients],
                        oneshot=0,
                        clip_rate=clip_rate
                    )
                    
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
                
    import numpy as np
    from typing import List

    def inner_product(self,server, clients, round_num, attack_start,
                    client_gradients,
                    negate_scale= 1.0):

        malicious_set = set(getattr(server, "malicious_clients", []))

        for i, client in enumerate(clients):
            if (client in malicious_set) and (round_num >= attack_start):
                grads = client_gradients[i]

                if not isinstance(grads, (list, tuple)):
                    try:
                        grads = list(grads)
                    except Exception:
                        print(f"[random_updates] Warning: client {i} gradients not list-like, skipping.")
                        continue

                new_grads = []
                for idx, (g,base_param) in enumerate(zip(grads,client.gradient)):
                    if not isinstance(g, np.ndarray):
                        print(f"[random_updates] Warning: client {i} gradient element {idx} is not ndarray (type={type(g)}); preserving original.")
                        new_grads.append(g)
                        continue

                    if g.size == 0:
                        new_grads.append(g.copy())
                        continue

                    if np.issubdtype(g.dtype, np.floating):
                        new_g = (-negate_scale * g).astype(g.dtype, copy=False)
                    else:
                        tmp = (-negate_scale * g.astype(np.float64))
                        try:
                            new_g = tmp.astype(g.dtype)
                        except Exception:
                            print(f"[random_updates] Warning: cannot cast negated grad back to dtype {g.dtype} for client {i}, element {idx}. Keeping float64.")
                            new_g = copy.deepcopy(tmp)

                    new_grads.append(new_g)
                    # new_g = torch.as_tensor(new_g, device=server.device, dtype=base_param.dtype)
                    client.gradient[idx]=copy.deepcopy(new_g)

                client_gradients[i] = new_grads
                
                # client.gradient=new_grads

        return client_gradients
