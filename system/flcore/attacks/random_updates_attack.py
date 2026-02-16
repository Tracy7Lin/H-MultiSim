import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .base_attack import BaseAttack
from .client_attack_utils import ClientAttackUtils
from .trigger_utils import TriggerUtils

class Random_Updates_Attack(BaseAttack):
    def __init__(self, args, device):
        super().__init__(args, device)
        
    def execute_attack_training(self, server, round_num, attack_start, oneshot, clip_rate, 
                              original_trigger_list, external_pattern, optimized_trigger_list):
        client_data=[]
        
        for client in server.alled_clients:
            if client in server.malicious_clients:
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("random_updates_attack: 恶意客户端参与训练...")
                        
                    client.train_malicious_random_uppdates(
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

    def random_updates(self, server,clients, round_num: int, attack_start: int, client_gradients,
                    mode= "add", eps= 1e-8, seed=42):
        if seed is not None:
            np.random.seed(seed)

        malicious_set = set(getattr(server, "malicious_clients", []))

        for i, client in enumerate(clients):

            if (client in malicious_set) and (round_num >= attack_start):
                print(f"{client.id}执行random_updates...")
                grads = client_gradients[i]

                if not isinstance(grads, (list, tuple)):
                    try:
                        grads = list(grads)
                    except Exception:
                        print(f"[random_updates] Warning: client {i} gradients not list-like, skipping.")
                        continue

                flattened = [g.reshape(-1) for g in grads if isinstance(g, np.ndarray) and g.size > 0]
                if len(flattened) == 0:
                    flat_all = np.array([], dtype=float)
                else:
                    flat_all = np.concatenate(flattened, axis=0)

                stdev = float(np.std(flat_all)) if flat_all.size > 0 else 0.0
                if stdev <= 0.0:
                    stdev = float(eps)

                new_grads = []
                for idx,g in enumerate(grads):
                    if not isinstance(g, np.ndarray):
                        print(f"[random_updates] Warning: client {i} has non-ndarray grad element of type {type(g)}; preserving it.")
                        new_grads.append(g)
                        continue

                    if g.size == 0:
                        new_grads.append(g.copy())
                        continue
                    # noise=-1*g
                    # print(stdev)
                    noise = np.random.uniform(np.minimum(-g, g), np.maximum(g, -g))*np.random.normal(loc=0.0, scale=stdev*10, size=g.shape).astype(g.dtype, copy=False)
                    # noise=np.random.uniform(np.minimum(-g,g),np.maximum(g,-g),size=g.shape)
                    
                    if mode == "replace":
                        new_grads.append(noise)
                        client.gradient[idx]=noise
                    elif mode == "add":
                        if np.issubdtype(g.dtype, np.integer):
                            tmp = g.astype(np.float32) + noise.astype(np.float32)
                            new_grads.append(tmp.astype(g.dtype))
                            client.gradient[idx]=tmp
                        else:
                            tmp=(g + noise).astype(g.dtype, copy=False)
                            new_grads.append(copy.deepcopy(tmp))
                            client.gradient[idx]=copy.deepcopy(tmp)
                    else:
                        raise ValueError(f"Unknown mode '{mode}'. Use 'replace' or 'add'.")
                    

                client_gradients[i] = new_grads
                # new_grads=torch.as_tensor(new_grads)
                # client.graient=new_grads
                

        return client_gradients

                        