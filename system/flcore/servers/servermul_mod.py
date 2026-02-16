import time
import copy
import torch
import numpy as np
import os
from flcore.clients.clientmul import clientMUL
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.attacks import BadPFLAttack, NeurotoxinAttack, DBAAttack, ModelReplacementAttack, BadNetsAttack, TriggerUtils, ClientAttackUtils, BaseAttack,Label_Poison_Attack,Random_Updates_Attack,Inner_Product_Attack,Model_Replace_Attack
from flcore.defense import RobustAggregation, BaseDefense
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from flcore.clients.clientbase import Client
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import random
import pytz

class FedMul_mod(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        self.args = args
        self.cluster_similarity_matrices = {}
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.set_slow_clients()
        self.set_clients(clientMUL)
        self.linkage_method = getattr(args, 'linkage_method', 'single')
        self.alpha = getattr(args, 'alpha', 0.01)
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        
        self.num_clusters = getattr(args, 'num_clusters', 3)
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.7)
        self.fusion_weight_gradient = getattr(args, 'fusion_weight_gradient', 0.5)
        self.fusion_weight_weight = getattr(args, 'fusion_weight_weight', 0.5)
        self.clustering_method = getattr(args, 'clustering_method', 'hierarchical')
        self.update_cluster_freq = getattr(args, 'update_cluster_freq', 5)
        
        self.t1=getattr(args,'time1',0)
        self.t2=getattr(args,'time2',25)
        self.t3=getattr(args,'time3',50)
        self.sigma=getattr(args,'sigma_Q',0.01)
        
        self.top_k=getattr(args,'top_k',0.4)
        self.gamma=getattr(args,'gamma',1)
        self.malicious_ratio=getattr(args,'malicious_ratio',0.2)
        self.top_k_neighbor=int(self.top_k*self.num_clients)
        self.top_k_score=np.zeros(self.num_clients)
        
        self.fuse_method = 'attention'
        
        for client in self.clients:
            client.fuse_method = self.fuse_method
            client.alpha = self.alpha
        
        self.temperature = getattr(args, 'temperature', 0.1)
        
        # 聚类相关
        self.client_clusters = {}  # {client_id: cluster_id}
        self.cluster_centers = {}  # {cluster_id: center_weights}
        self.similarity_matrix = None
        self.gradient_similarity_dir_matrix = None
        self.gradient_similarity_val_matrix = None
        self.weight_similarity_matrix = None
        
        # 个性化学习参数
        self.personalization_rounds = getattr(args, 'personalization_rounds', 5)
        self.cluster_aggregation_weight = getattr(args, 'cluster_aggregation_weight', 0.7)
        
        # 鲁棒性检测
        self.malicious_clients = []
        self.robustness_threshold = getattr(args, 'robustness_threshold', 0.3)
        
        self.base_attack = BaseAttack(args, self.device)
        self.attack_method = getattr(args, 'attack', None)
        self.poisonlabel = getattr(args, 'poison_label', 1)
        self.poisonratio = getattr(args, 'poison_rate', 4)
        self.malicious_clients = []
        self.client_features = {}
        self.sensitivity_matrix_cache = {}
        
        if self.attack_method and self.attack_method != 'none':
            self.malicious_clients = self.base_attack.select_malicious_clients(self.clients, num_malicious=int(self.num_clients*self.malicious_ratio))
            print(f"Selected {len(self.malicious_clients)} malicious clients for robustness testing")
            
            # initialize attack module
            if self.attack_method == 'badpfl':
                self.badpfl_attack = BadPFLAttack(args, self.device)
            elif self.attack_method == 'neurotoxin':
                self.neurotoxin_attack = NeurotoxinAttack(args, self.device)
            elif self.attack_method == 'dba':
                self.dba_attack = DBAAttack(args, self.device)
            elif self.attack_method == 'model_replacement':
                self.model_replacement_attack = ModelReplacementAttack(args, self.device)
            elif self.attack_method == 'badnets':
                self.badnets_attack = BadNetsAttack(args, self.device)
            elif self.attack_method == 'label_poison_attack':
                self.label_poison_attack=Label_Poison_Attack(args,self.device)
            elif self.attack_method == 'random_updates_attack':
                self.random_updates_attack=Random_Updates_Attack(args,self.device)
            elif self.attack_method == 'inner_product_attack':
                self.inner_product_attack=Inner_Product_Attack(args,self.device)
            elif self.attack_method == 'model_replace_attack':
                self.model_replace_attack = Model_Replace_Attack(args,self.device)
 
        if hasattr(args, 'defense') and args.defense and args.defense != 'none':
            self.defense = BaseDefense(args, self.device)
        
        self.Budget = []
        
        print(f"FedMul_mod initialized with {self.num_clusters} clusters")
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Using attention fusion method with temperature: {self.temperature}")
    
    def evaluate_asr(self, trigger, pattern):
        return self.base_attack.evaluate_asr(
            clients=self.clients,
            trigger=trigger,
            pattern=pattern,
            poison_label=self.poisonlabel
        )
    
    def avg_generalization_metrics(self):
        if not self.rs_test_acc:
            return 0.0
        return sum(self.rs_test_acc) / len(self.rs_test_acc)
    
    def all_clients(self):
        return self.clients

    def train(self):
        refre_client=Client(self.args, 
                            id=-1, 
                            train_samples=0, 
                            test_samples=0, 
                            train_slow=0, 
                            send_slow=0)
        T0 = getattr(self.args, 'initial_rounds', 10)  
        print(f"Phase 1: Clustering Stage with CorrelationFL for {T0} rounds...")
        
        num_all_clients = len(self.clients)
        all_client_indices = list(range(num_all_clients))
        self.global_similarity_matrix = np.eye(num_all_clients)
        
        for t in range(T0):
            self.send_models()
            print(f"\nClustering Round: {t+1}/{T0}")
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()

            refre_client.set_parameters(self.global_model)
            refre_model=self._get_client_params(refre_client)
            client_weights, client_gradients = self.run_one_round(
                self.clients, 
                self.global_similarity_matrix,
                all_client_indices,
                is_clustering_phase=True, 
                base_model=refre_model
            )
            
            P_g_dir = self.gradient_similarity_direction(client_gradients)
            P_w = self.compute_similarity_KLD(self.num_clients,[client.id for client in self.clients])
            P_g_val=self.gradient_similarity_value(client_gradients)
            self.global_similarity_matrix = self.similarity_fusion(t,[P_g_dir,P_g_val,P_w])
            if t==0:
                self.top_k_score=self.compute_top_k_score(self.global_similarity_matrix,self.top_k_neighbor)
            else:
                new_top_k_score=self.compute_top_k_score(self.global_similarity_matrix,self.top_k_neighbor)
                self.top_k_score=(self.top_k_score+new_top_k_score)/2
                print(new_top_k_score)
            self.selected_clients = self.clients
            
            print(self.top_k_score)
            self.receive_models_mulmod()
            if t==0:
                self.aggregate_parameters() 
            else:
                self.aggregate_parameters_by_grad()

            if t % self.eval_gap == 0:
                self.evaluate()

        self.global_similarity_matrix=self.mod_global_matrix()
        self.client_groups = self.hierarchical_clustering_from_similarity(self.global_similarity_matrix, all_client_indices)
        self.cluster_method='HC'
        self.topk_score=self.top_k_score.tolist()
        print(f"--- Clustering completed. Groups: {self.client_groups} ---")

        T1 = self.global_rounds - T0
        print(f"--- Phase 2: Intra-cluster Training for {T1} rounds ---")
        
        cluster_models = {}
        for group_id, group_client_indices in enumerate(self.client_groups):
            cluster_models[group_id] = copy.deepcopy(self.global_model)
            self.cluster_similarity_matrices[group_id] = np.eye(len(group_client_indices))

        self.top_k_score=np.zeros(self.num_clients)
        
        average_cluster_acc=[0]*self.num_clusters
        average_test_acc=0
        average_benign_acc=0
        average_clients_acc=[0]*self.num_clients
        
        for t in range(T1):
            print(f"\nIntra-cluster Training Round: {t+1}/{T1}")
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()
            for group_id, group_client_indices in enumerate(self.client_groups):
                
                cluster_clients = [self.clients[i] for i in group_client_indices]

                refre_client.set_parameters(cluster_models[group_id])
                refre_model=self._get_client_params(refre_client)
                client_weights, client_gradients = self.run_one_round(
                    cluster_clients,
                    self.cluster_similarity_matrices[group_id],
                    group_client_indices,
                    is_clustering_phase=False,
                    base_model=refre_model
                )
                
                P_g_dir = self.gradient_similarity_direction(client_gradients)
                P_w = self.compute_similarity_KLD(len(cluster_clients),group_client_indices)
                P_g_val=self.gradient_similarity_value(client_gradients)
                self.global_similarity_matrix = self.similarity_fusion(t+T0,[P_g_dir,P_g_val,P_w])
                new_top_k_cluster=self.compute_top_k_score(self.global_similarity_matrix,len(cluster_clients)-1)
                if t==0:
                   for i,id in enumerate(group_client_indices):
                       self.top_k_score[id]=new_top_k_cluster[i]
                else:
                    # new_top_k_score=self.cluster_top_k_fill(self.global_similarity_matrix,group_client_indices,new_top_k_cluster)
                    # idx = np.asarray(group_client_indices)
                    # self.top_k_score[idx]=(self.top_k_score[group_client_indices]+new_top_k_score[idx])/2
                    for i,id in enumerate(group_client_indices):
                       self.top_k_score[id]=(new_top_k_cluster[i]+self.top_k_score[id])/2
                self.selected_clients = cluster_clients 
                original_join_clients = self.join_clients
                original_current_num_join_clients = self.current_num_join_clients
                cluster_join_clients = max(1, int(self.join_ratio * len(cluster_clients)))
                self.join_clients = cluster_join_clients
                self.current_num_join_clients = cluster_join_clients

                self.receive_models_mulmod() 

                self.join_clients = original_join_clients
                self.current_num_join_clients = original_current_num_join_clients
                cluster_model=copy.deepcopy(cluster_models[group_id])
                cluster_models[group_id]=self.aggregate_cluster_models_by_grad(cluster_clients,client_gradients,cluster_model)
                
                for client in cluster_clients:
                    client.set_model(cluster_models[group_id])
                    

            if t % self.eval_gap == 0:
                for group_id, group_client_indices in enumerate(self.client_groups):
                    for client_idx in group_client_indices:
                        self.clients[client_idx].set_model(cluster_models[group_id])
                cluster_acc=self.evaluate_cluster()
                test_acc,benign_acc,clients_acc=self.evaluate()
                
                if t>=T1-5:
                    average_benign_acc=average_benign_acc+benign_acc
                    average_test_acc=average_test_acc+test_acc
                    average_cluster_acc=[x+y for x,y in zip(average_cluster_acc,cluster_acc)]
                    average_clients_acc=[x+y for x,y in zip(average_clients_acc,clients_acc)]
                    

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        
        # 训练完成后，统一处理所有保存任务
        print("\n" + "="*70)
        print("Training completed! Starting post-training tasks...")
        print("="*70)
        
        # 计算统一的保存路径（system目录下的 results/fedmul_demo）
        # __file__ = /system/flcore/servers/servermul.py
        # 需要上移3层到 /system/
        system_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(system_dir, "results", "fedmul_demo")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📁 All files will be saved to: {save_dir}")
        
        # 1. 保存训练结果（.h5文件）到统一目录
        print("\nStep 1/3: Saving training results (.h5)...")
        try:
            self._save_results_to_dir(save_dir)
            print(f"✓ Training results saved")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
        
        # 2. 导出模型和聚类可视化
        print("\nStep 2/3: Exporting models and clustering visualization...")
        try:
            self.export_model(save_dir=save_dir)
            print("✓ Models and visualization exported successfully")
        except Exception as e:
            print(f"✗ Model export failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. 进行本地推理测试（只使用10个客户端）
        print("\nStep 3/3: Running local inference with 10 clients (20 rounds)...")
        self.evaluate()
        for group_id, group_client_indices in enumerate(self.client_groups):
            for client_idx in group_client_indices:
                self.clients[client_idx].set_model(cluster_models[group_id])
        
        cluster_acc=self.evaluate_cluster()
        test_acc,benign_acc,clients_acc=self.evaluate()

        
        group_client_idx=[]
        for client_id in self.client_groups:
            group_client_idx.append(client_id)
            
        test_acc=average_test_acc/5
        benign_acc=average_benign_acc/5
        cluster_acc=[x/5 for x in average_cluster_acc]
        clients_acc=[x/5 for x in average_clients_acc]
            
        save_attack_results(self.gamma,self.malicious_ratio,self.top_k,'Benign',self.cluster_method,test_acc,cluster_acc,group_client_idx,[],benign_acc,self.topk_score,clients_acc,best_cluster_acc)
        
        # try:
        #     log_path = self.run_local_inference_after_training(
        #         num_rounds=20,
        #         num_clients_to_use=10,  # 只使用10个客户端
        #         save_dir=save_dir
        #     )
        #     print(f"✓ Inference completed! Logs saved to: {log_path}")
        # except Exception as e:
        #     print(f"✗ Local inference failed: {e}")
        #     import traceback
        #     traceback.print_exc()
        
        print("\n" + "="*70)
        print("Post-training tasks completed!")
        print(f"Check results in: {save_dir}")
        print("="*70)
    
    def train_with_attack(self, pattern, trigger):
        refre_client=Client(self.args, 
                            id=-1, 
                            train_samples=0, 
                            test_samples=0, 
                            train_slow=0, 
                            send_slow=0)
        original_trigger_list = trigger
        optimized_trigger_list = copy.deepcopy(original_trigger_list)
        external_pattern = pattern 
        attack_start = 3
        oneshot = 1         
        clip_rate = 0     
        self.attack_start=3

        T0 = getattr(self.args, 'initial_rounds', 10)
        print(f"Phase 1: Clustering Stage with CorrelationFL for {T0} rounds (with attack)...")
        
        num_all_clients = len(self.clients)
        all_client_indices = list(range(num_all_clients))
        self.global_similarity_matrix = np.eye(num_all_clients)
        
        for t in range(T0):
            print(f"\nClustering Round: {t+1}/{T0}")
            s_t = time.time()
            self.t=t
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()
            self.send_models()

            refre_client.set_parameters(self.global_model)
            refre_model=self._get_client_params(refre_client)
            client_weights, client_gradients = self.run_one_round_attack(
                self.clients, 
                self.global_similarity_matrix,
                all_client_indices,
                is_clustering_phase=True, 
                base_model=refre_model
            )
            if t>= attack_start:
                if hasattr(self, "random_updates_attack"):
                    print("gradients are random updated...")
                    client_gradients=self.random_updates_attack.random_updates(self,self.clients,t,attack_start,client_gradients)
                elif hasattr(self, "inner_product_attack"):
                    print("gradients are inner_product...")
                    client_gradients=self.inner_product_attack.inner_product(self,self.clients,t,attack_start,client_gradients)
                elif hasattr(self, 'model_replace_attack'):
                    print("gradients are model replace...")
                    client_gradients=self.model_replace_attack.model_replace(self,self.clients,t,attack_start,client_gradients)

            P_g_dir = self.gradient_similarity_direction(client_gradients)
            P_w = self.compute_similarity_KLD(self.num_clients,[client.id for client in self.clients])
            P_g_val=self.gradient_similarity_value(client_gradients)
            self.global_similarity_matrix = self.similarity_fusion(t,[P_g_dir,P_g_val,P_w])
            
            if t==0:
                self.top_k_score=self.compute_top_k_score(self.global_similarity_matrix,self.top_k_neighbor)
            else:
                new_top_k_score=self.compute_top_k_score(self.global_similarity_matrix,self.top_k_neighbor)
                self.top_k_score=(self.top_k_score+new_top_k_score)/2
                print(new_top_k_score)
                
            self.selected_clients = self.clients
            self.receive_models_mulmod()
            if t==0:
                self.aggregate_parameters() 
            else:
                self.aggregate_parameters_by_grad()

            if t % self.eval_gap == 0:
                print(f"\n-------------Round number: {t}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                
                # 评估ASR
                # if t >= attack_start and self.attack_method:
                #     self._evaluate_attack_success(t, attack_start, original_trigger_list, 
                #                                 external_pattern, optimized_trigger_list)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        self.global_similarity_matrix=self.mod_global_matrix()
        self.client_groups = self.hierarchical_clustering_from_similarity(self.global_similarity_matrix, all_client_indices)
        self.cluster_method='HC'
        self.topk_score=self.top_k_score.tolist()
        # self.client_groups=self.k_means_cluster(self.num_clusters,self.global_similarity_matrix,all_client_indices)
        # self.cluster_method='K-means'
        print(f"--- Clustering completed. Groups: {self.client_groups} ---")

        T1 = self.global_rounds - T0
        print(f"--- Phase 2: Intra-cluster Training for {T1} rounds (with attack) ---")
        
        cluster_models = {}
        for group_id, group_client_indices in enumerate(self.client_groups):
            cluster_models[group_id] = copy.deepcopy(self.global_model)
            self.cluster_similarity_matrices[group_id] = np.eye(len(group_client_indices))

        self.top_k_score=np.zeros(self.num_clients)
        
        average_cluster_acc=[0]*self.num_clusters
        average_test_acc=0
        average_benign_acc=0
        average_clients_acc=[0]*self.num_clients
        
        for t in range(T1):
            
            self.t=t
            print(f"\nIntra-cluster Training Round: {t+1}/{T1}")
            s_t = time.time()
            
            self.selected_clients = self.select_clients()
            self.alled_clients = self.all_clients()
            
            for group_id, group_client_indices in enumerate(self.client_groups):
                
                cluster_clients = [self.clients[i] for i in group_client_indices]

                refre_client.set_parameters(cluster_models[group_id])
                refre_model=self._get_client_params(refre_client)
                client_weights, client_gradients = self.run_one_round_attack(
                    cluster_clients,
                    self.cluster_similarity_matrices[group_id],
                    group_client_indices,
                    is_clustering_phase=False,
                    base_model=refre_model
                )
                
                if t>= attack_start:
                    if hasattr(self, "random_updates_attack"):
                        print("gradients are random updated...")
                        client_gradients=self.random_updates_attack.random_updates(self,cluster_clients,t,attack_start,client_gradients)
                    elif hasattr(self, "inner_product_attack"):
                        print("gradients are inner_product...")
                        client_gradients=self.inner_product_attack.inner_product(self,cluster_clients,t,attack_start,client_gradients)
                    elif hasattr(self, 'model_replace_attack'):
                        print("gradients are model replace...")
                        client_gradients=self.model_replace_attack.model_replace(self,cluster_clients,t,attack_start,client_gradients)
            
                P_g_dir = self.gradient_similarity_direction(client_gradients)
                P_w = self.compute_similarity_KLD(len(cluster_clients),group_client_indices)
                P_g_val=self.gradient_similarity_value(client_gradients)
                self.global_similarity_matrix = self.similarity_fusion(t+T0,[P_g_dir,P_g_val,P_w])

                new_top_k_cluster=self.compute_top_k_score(self.global_similarity_matrix,len(cluster_clients)-1)
                if t==0:
                   for i,id in enumerate(group_client_indices):
                       self.top_k_score[id]=new_top_k_cluster[i]
                else:
                    # new_top_k_score=self.cluster_top_k_fill(self.global_similarity_matrix,group_client_indices,new_top_k_cluster)
                    # idx = np.asarray(group_client_indices)
                    # self.top_k_score[idx]=(self.top_k_score[group_client_indices]+new_top_k_score[idx])/2
                    for i,id in enumerate(group_client_indices):
                       self.top_k_score[id]=(new_top_k_cluster[i]+self.top_k_score[id])/2
                print(self.top_k_score)

                self.selected_clients = cluster_clients
                original_join_clients = self.join_clients
                original_current_num_join_clients = self.current_num_join_clients
                cluster_join_clients = max(1, int(self.join_ratio * len(cluster_clients)))
                self.join_clients = cluster_join_clients
                self.current_num_join_clients = cluster_join_clients

                self.receive_models_mulmod()

                self.join_clients = original_join_clients
                self.current_num_join_clients = original_current_num_join_clients
                cluster_model=copy.deepcopy(cluster_models[group_id])
                cluster_models[group_id]=self.aggregate_cluster_models_by_grad(cluster_clients,client_gradients,cluster_model)

                for client in cluster_clients:
                    client.set_model(cluster_models[group_id])
                    
            if t % self.eval_gap == 0:
                for group_id, group_client_indices in enumerate(self.client_groups):
                    for client_idx in group_client_indices:
                        self.clients[client_idx].set_model(cluster_models[group_id])
                
                print(f"\n-------------Round number: {T0 + t}-------------")
                print("\nEvaluate global model")
                cluster_acc=self.evaluate_cluster()
                test_acc,benign_acc,clients_acc=self.evaluate()
                
                if t>=T1-5:
                    average_benign_acc=average_benign_acc+benign_acc
                    average_test_acc=average_test_acc+test_acc
                    average_cluster_acc=[x+y for x,y in zip(average_cluster_acc,cluster_acc)]
                    average_clients_acc=[x+y for x,y in zip(average_clients_acc,clients_acc)]
                    
                
                # 评估ASR
                # if t >= attack_start and self.attack_method:
                #     self._evaluate_attack_success(T0 + t, attack_start, original_trigger_list, 
                #                                 external_pattern, optimized_trigger_list)

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        
        system_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(system_dir, "results", "fedmul_mod_demo")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n📁 All files will be saved to: {save_dir}")
        
        # 1. 保存训练结果（.h5文件）到统一目录
        print("\nStep 1/3: Saving training results (.h5)...")
        try:
            self._save_results_to_dir(save_dir)
            print(f"✓ Training results saved")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
        
        # 2. 导出模型和聚类可视化
        print("\nStep 2/3: Exporting models and clustering visualization...")
        try:
            self.export_model(save_dir=save_dir)
            print("✓ Models and visualization exported successfully")
        except Exception as e:
            print(f"✗ Model export failed: {e}")
            import traceback
            traceback.print_exc()
        
        self.evaluate()
        for group_id, group_client_indices in enumerate(self.client_groups):
            for client_idx in group_client_indices:
                self.clients[client_idx].set_model(cluster_models[group_id])
        cluster_acc=self.evaluate_cluster()
        test_acc,benign_acc,clients_acc=self.evaluate()
                
        print("=====================Malicious client=================================")
        print("Selected malicious client IDs:", [client.id for client in self.malicious_clients])
        print("=============================================================")
        # try:
        #     log_path = self.run_local_inference_after_training(
        #         num_rounds=20,
        #         num_clients_to_use=10,  # 只使用20个客户端
        #         save_dir=save_dir
        #     )
        #     print(f"✓ Inference completed! Logs saved to: {log_path}")
        # except Exception as e:
        #     print(f"✗ Local inference failed: {e}")
        #     import traceback
        #     traceback.print_exc()

        print(f'+++++++++++++++++++++++++++++++++++++++++')
        gen_acc = self.avg_generalization_metrics()
        print(f'Generalization Acc: {gen_acc}')
        print(f'+++++++++++++++++++++++++++++++++++++++++')

        # 保存攻击实验数据 - 统一使用base_attack保存
        if self.attack_method and self.attack_method != 'none' and(self.attack_method != "label_poison_attack"):
            print(f"保存{self.attack_method.upper()}攻击实验数据...")
            self.base_attack.save_experiment_data()

        self.save_results()
        self.save_global_model()

        group_client_idx=[]
        for client_id in self.client_groups:
            group_client_idx.append(client_id)
            
        malicious_client=[client.id for client in self.malicious_clients]
        malicious_client.sort()
        
        test_acc=average_test_acc/5
        benign_acc=average_benign_acc/5
        cluster_acc=[x/5 for x in average_cluster_acc]
        clients_acc=[x/5 for x in average_clients_acc]
        save_attack_results(self.gamma,self.malicious_ratio,self.top_k,self.attack_method,self.cluster_method,test_acc,cluster_acc,group_client_idx,malicious_client,benign_acc,self.topk_score,clients_acc)
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
    
    def _execute_attack_training(self, round_num, attack_start, oneshot, clip_rate, 
                               original_trigger_list, external_pattern, optimized_trigger_list):
        """执行攻击训练的具体逻辑"""
        if round_num < attack_start:
            return optimized_trigger_list
            
        # 根据攻击类型执行对应的攻击训练
        if self.attack_method == 'badpfl':
            badpfl_trigger, badpfl_pattern = self.badpfl_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局Bad-PFL攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_badpfl_attack = self.badpfl_attack
            return badpfl_trigger, badpfl_pattern
            
        elif self.attack_method == 'neurotoxin':
            optimized_trigger_list = self.neurotoxin_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                oneshot=oneshot,
                clip_rate=clip_rate,
                original_trigger_list=original_trigger_list,
                external_pattern=external_pattern,
                optimized_trigger_list=optimized_trigger_list
            )
            
        elif self.attack_method == 'dba':
            optimized_trigger_list = self.dba_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                oneshot=oneshot,
                clip_rate=clip_rate,
                original_trigger_list=original_trigger_list,
                external_pattern=external_pattern,
                optimized_trigger_list=optimized_trigger_list
            )
            
        elif self.attack_method == 'model_replacement':
            optimized_trigger_list = self.model_replacement_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局模型替换攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_model_replacement_attack = self.model_replacement_attack
            
        elif self.attack_method == 'badnets':
            optimized_trigger_list = self.badnets_attack.execute_attack_training(
                server=self,
                round_num=round_num,
                attack_start=attack_start,
                external_pattern=external_pattern
            )
            # 设置全局BadNets攻击实例，供客户端poisontest使用
            import sys
            from flcore.attacks import client_attack_utils
            client_attack_utils._global_badnets_attack = self.badnets_attack
        
        elif self.attack_method == 'label_poison_attack':
            self.label_poison_attack.execute_attack_training(
                self,round_num,attack_start,oneshot,clip_rate,None,None,None
            )
            return None
        elif self.attack_method =='random_updates_attack':
            self.random_updates_attack.execute_attack_training(self,round_num,attack_start,oneshot,clip_rate,None,
                                                               None,None)
            return None
        
        elif self.attack_method =='inner_product_attack':
            self.inner_product_attack.execute_attack_training(self,round_num,attack_start,oneshot,clip_rate,None,
                                                               None,None)
            return None
        elif self.attack_method == 'model_replace_attack':
            self.model_replace_attack.execute_attack_training(self,round_num,attack_start,oneshot,clip_rate,None,
                                                               None,None)
            return None
        return optimized_trigger_list
    
    def _evaluate_attack_success(self, round_num, attack_start, original_trigger_list, 
                                external_pattern, optimized_trigger_list):
        """评估攻击成功率"""
        if round_num < attack_start:
            return
            
        if self.attack_method == 'badpfl':
            # BadPFL的ASR评估需要特殊处理
            global_asr = self.evaluate_asr(
                trigger=optimized_trigger_list,
                pattern=external_pattern
            )
        elif self.attack_method == 'model_replacement':
            # Evaluate ASR using all clients' test data
            total_samples = 0
            successful_attacks = 0
            
            for client in self.clients:
                test_loader = client.load_test_data(batch_size=32)
                client_successful, client_total = self.model_replacement_attack._evaluate_asr_on_loader(
                    self.global_model, test_loader, self.device
                )
                successful_attacks += client_successful
                total_samples += client_total
            
            global_asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        elif self.attack_method == 'badnets':
            # Evaluate BadNets ASR using all clients' test data
            total_samples = 0
            successful_attacks = 0
            
            for client in self.clients:
                test_loader = client.load_test_data(batch_size=32)
                client_asr = self.badnets_attack._evaluate_asr_on_loader(
                    self.global_model, test_loader, self.device
                )
                successful_attacks += client_asr * len(test_loader.dataset)
                total_samples += len(test_loader.dataset)
            
            global_asr = successful_attacks / total_samples if total_samples > 0 else 0.0
        else:
            global_asr = self.evaluate_asr(
                trigger=optimized_trigger_list, 
                pattern=external_pattern
            )
        
        print(f"Global ASR at round {round_num}: {global_asr:.4f}")
    
    def run_one_round(self, clients, similarity_matrix_for_training, client_indices, is_clustering_phase,base_model):
        
        client_weights = []
        client_gradients = []
        pre_params=copy.deepcopy(base_model)
        
        for epoch in range(self.local_epochs):
            neighbor_models_map = {}
            
            for client in clients:
                    # 为每个客户端准备其邻居模型列表
                    neighbor_models_map[client.id] = {
                        c.id: c.model.state_dict() for c in clients if c.id != client.id
                    }

            for client in clients:                
                client.set_neighbor_models(neighbor_models_map[client.id])
                client.train_with_similarity_constraint(similarity_matrix_for_training, client_indices)
                if epoch == self.local_epochs-1:
                    post_params = self._get_client_params(client)
                    gradient = [post - pre for post, pre in zip(post_params, pre_params)]
                    
                    client_weights.append(post_params)
                    client_gradients.append(gradient)
                    client.gradient=gradient
            
        return client_weights, client_gradients
    
    def run_one_round_attack(self, clients, similarity_matrix_for_training, client_indices, is_clustering_phase,base_model):
        
        client_weights = []
        client_gradients = []
        pre_params=copy.deepcopy(base_model)
                
        clip_rate = 0  
        
        for epoch in range(self.local_epochs):
            neighbor_models_map = {}
            
            for client in clients:
                    # 为每个客户端准备其邻居模型列表
                    neighbor_models_map[client.id] = {
                        c.id: c.model.state_dict() for c in clients if c.id != client.id
                    }
            
            for client in clients:
                
                client.set_neighbor_models(neighbor_models_map[client.id])
                if self.t >= self.attack_start and self.attack_method:
                    if client in self.malicious_clients and self.attack_method == 'label_poison_attack':
                        client.train_malicious_label_poison(
                        is_selected=client.id in [c.id for c in self.selected_clients],
                        poison_ratio=self.poisonratio,
                        poison_label=self.poisonlabel,
                        trigger=None,
                        pattern=None,
                        oneshot=0,
                        clip_rate=clip_rate
                    )
                    else:
                        client.train_with_similarity_constraint(similarity_matrix_for_training, client_indices)
                else:
                    client.train_with_similarity_constraint(similarity_matrix_for_training, client_indices)
                if epoch == self.local_epochs-1:
                    post_params = self._get_client_params(client)
                    gradient = [post - pre for post, pre in zip(post_params, pre_params)]
                    
                    client_weights.append(post_params)
                    client_gradients.append(gradient)
                    client.gradient=gradient
            
        return client_weights, client_gradients
    
    def aggregate_cluster_models_by_grad(self, cluster_clients,client_gradients,cluster_model):
        
        cluster_client_ids = [client.id for client in cluster_clients]

        uploaded_models_in_cluster = []
        training_samples_in_cluster = []
        uploaded_gradients_in_cluster= []
        uploaded_weight_in_cluster=[]
        for i, client_id in enumerate(self.uploaded_ids):
            if client_id in cluster_client_ids:
                print(client_id,self.uploaded_weights[i])
                uploaded_models_in_cluster.append(self.uploaded_models[i])
                training_samples_in_cluster.append(self.clients[client_id].train_samples)
                uploaded_gradients_in_cluster.append(self.uploaded_gradients[i])
                uploaded_weight_in_cluster.append(self.uploaded_weights[i])
        if not uploaded_models_in_cluster:
            if cluster_clients:
                return copy.deepcopy(cluster_clients[0].model)
            return None

        total_samples = sum(training_samples_in_cluster)
        base_model = copy.deepcopy(cluster_model)
            
        for client_gradient, weight in zip(uploaded_gradients_in_cluster, uploaded_weight_in_cluster):
            # weight = samples / total_samples
            for base_param, grad in zip(base_model.parameters(), client_gradient):
                grad = torch.as_tensor(grad, device=self.device, dtype=base_param.dtype)
                base_param.data += grad * weight
        
        return base_model
    
    # def aggregate_cluster_models(self, cluster_clients):
        
    #     cluster_client_ids = {client.id for client in cluster_clients}
        
    #     uploaded_models_in_cluster = []
    #     training_samples_in_cluster = []
    #     for i, client_id in enumerate(self.uploaded_ids):
    #         if client_id in cluster_client_ids:
    #             uploaded_models_in_cluster.append(self.uploaded_models[i])
    #             training_samples_in_cluster.append(self.clients[client_id].train_samples)

    #     if not uploaded_models_in_cluster:
    #         if cluster_clients:
    #             return copy.deepcopy(cluster_clients[0].model)
    #         return None

    #     total_samples = sum(training_samples_in_cluster)
    #     base_model = copy.deepcopy(uploaded_models_in_cluster[0])
        
    #     for param in base_model.parameters():
    #         param.data.zero_()
            
    #     for model, samples in zip(uploaded_models_in_cluster, ):
    #         # weight = samples / total_samples
    #         for base_param, client_param in zip(base_model.parameters(), model.parameters()):
    #             base_param.data += client_param.data.clone() * weight
        
    #     return base_model
    
    
    def gradient_similarity_value(self,client_gradients):
        ''' 
        计算梯度异常值相似性矩阵 P_g_val
        首先计算出客户端间梯度的曼哈顿距离，然后由于曼哈顿距离可能数量级较大，可能要
        缩放，比如除以梯度的维度
        然后可能应该要做exp(-x)投影到0-1
        然后再归一化
        '''
        num_clients = len(client_gradients)
        similarity_matrix = np.zeros((num_clients,num_clients))
        flattened_gradients = []
        for gradients in client_gradients:
            flat_grad = np.concatenate([g.flatten() if not isinstance(g,np.ndarray) else g.flatten() for g in gradients])
            flattened_gradients.append(flat_grad)
            
        for i in range(num_clients):
            for j in range(i,num_clients):
                if i == j:
                    similarity_matrix[i,j] = 1
                else:
                    g1, g2 = flattened_gradients[i], flattened_gradients[j]
                    diff = g1 - g2
                    diff = np.abs(diff)
                    manhattan_dis=np.sum(diff)
                    # 这个要看实际情况要不要缩放
                    manhattan_dis /=np.sqrt(len(flattened_gradients[i]))
                    sim_mhtdis = np.exp(-manhattan_dis)
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim_mhtdis
        
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums
        
        return normalized_matrix
    
    def gradient_manhattan_to_origin(self, client_gradients):
        '''
        计算每个客户端梯度到原点（0 向量）的曼哈顿距离
        返回 shape = (num_clients,) 的一维数组
        '''
        num_clients = len(client_gradients)
        distances = np.zeros(num_clients)

        for i, gradients in enumerate(client_gradients):
            flat_grad = np.concatenate([
                g.flatten() if not isinstance(g, np.ndarray) else g.flatten()
                for g in gradients
            ])
            manhattan_dis = np.sum(np.abs(flat_grad))

            manhattan_dis /= np.sqrt(len(flat_grad))

            distances[i] = manhattan_dis

        return distances

    
    def gradient_similarity_direction(self, client_gradients):
        """计算梯度方向相似性矩阵 P_g_dir (基于梯度)"""
        num_clients = len(client_gradients)
        similarity_matrix = np.zeros((num_clients, num_clients))
    
        flattened_gradients = []
        for gradients in client_gradients:
            flat_grad = np.concatenate([g.flatten() if not isinstance(g, np.ndarray) else g.flatten() for g in gradients])
            flattened_gradients.append(flat_grad)
        
        for i in range(num_clients):
            for j in range(i, num_clients): 
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    g1, g2 = flattened_gradients[i], flattened_gradients[j]
                    norm1, norm2 = np.linalg.norm(g1), np.linalg.norm(g2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(g1, g2) / (norm1 * norm2)
                        similarity_matrix[i, j] = max(0, sim) 
                        similarity_matrix[j, i] = similarity_matrix[i, j] 
                    else:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                        
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        # 防止除以零
        row_sums[row_sums == 0] = 1
        normalized_matrix = similarity_matrix / row_sums
        return normalized_matrix
    
    # def weight_similarity(self, client_weights):

    #     num_clients = len(client_weights)
    #     similarity_matrix = np.zeros((num_clients, num_clients))

    #     flattened_weights = []
    #     for weights in client_weights:
    #         flat_weight = np.concatenate([w.flatten() for w in weights])
    #         flattened_weights.append(flat_weight)

    #     for i in range(num_clients):
    #         for j in range(i, num_clients):
    #             if i == j:
    #                 similarity_matrix[i, j] = 1.0
    #             else:
    #                 w1, w2 = flattened_weights[i], flattened_weights[j]
    #                 norm1, norm2 = np.linalg.norm(w1), np.linalg.norm(w2)
    #                 if norm1 > 0 and norm2 > 0:
    #                     sim = np.dot(w1, w2) / (norm1 * norm2)
    #                     similarity_matrix[i, j] = max(0, sim)
    #                     similarity_matrix[j, i] = sim
    #                 else:
    #                     similarity_matrix[i, j] = 0.0
    #                     similarity_matrix[j, i] = 0.0
    #     # 增加行归一化
    #     row_sums = similarity_matrix.sum(axis=1, keepdims=True)
    #     row_sums[row_sums == 0] = 1
    #     normalized_matrix = similarity_matrix / row_sums
    #     return normalized_matrix
    
    
     
    

    # def _logsumexp(self,x):
    #     x = np.asarray(x, dtype=np.float64)
    #     m = np.max(x)
    #     return m + np.log(np.sum(np.exp(x - m)))

    # def _log_softmax(self,weights):
    #     w = np.asarray(weights, dtype=np.float64).ravel()
    #     lse = self._logsumexp(w)
    #     return w - lse

    # def _kl_from_log(self,logp, logq):
    #     p = np.exp(logp)
    #     min_logq = -1e6  
    #     logq_clipped = np.maximum(logq, min_logq)
    #     return float(np.sum(p * (logp - logq_clipped)))

    # def _js_divergence_from_log(self,logp, logq):
    #     p = np.exp(logp)
    #     q = np.exp(logq)
    #     m = 0.5 * (p + q)
    #     m = np.clip(m, 1e-300, None)
    #     logm = np.log(m)
    #     kl_pm = np.sum(p * (logp - logm))
    #     kl_qm = np.sum(q * (logq - logm))
    #     return float(0.5 * (kl_pm + kl_qm))

    # def _cosine_similarity_flat(self,a, b):
    #     a = np.asarray(a, dtype=np.float64).ravel()
    #     b = np.asarray(b, dtype=np.float64).ravel()
    #     na = np.linalg.norm(a)
    #     nb = np.linalg.norm(b)
    #     if na == 0 or nb == 0:
    #         return 0.0
    #     return float(np.dot(a, b) / (na * nb))

    # def output_test(self,client):
    #     test_data=test_data = read_client_data(client.dataset, self.num_clients+1, is_train=False, few_shot=self.few_shot)
    #     test_loader= DataLoader(test_data, self.batch_size, drop_last=False, shuffle=True)
    #     outputs=[]
    #     for i, (x, y) in enumerate(test_loader):
    #         if type(x) == type([]):
    #             x[0] = x[0].to(self.device)
    #         else:
    #             x = x.to(self.device)
    #         y = y.to(self.device)
            
    #         # 标准损失计算
    #         output = client.model(x)
    #         outputs.append(output.detach().cpu().numpy())
    #     return np.concatenate(outputs, axis=0)


    # def weight_similarity_mod(self, client_weights, client_ids, method='js', fallback='cosine'):

    #     num_clients = len(client_weights)
    #     # print(len(client_weights))
    #     sim_mat = np.zeros((num_clients, num_clients), dtype=np.float64)
    #     need_log = method in ('kl', 'symkl', 'js','kl_test','js_test')

    #     if method == 'kl_test':
    #         output_list=[]
    #         for id in client_ids:
    #             client=self.clients[id]
    #             output=self.output_test(client)
    #             output_list.append(output)
    #         for i in range(num_clients):
    #             for j in range(num_clients):
    #                 dist=0
    #                 if i==j:
    #                     sim=1.0
    #                 else:
    #                     output_p=output_list[i]
    #                     output_q=output_list[j]
    #                     for k in range(len(output_p)):
    #                         logp=np.log(output_p[k]+1e-10)
    #                         logq=np.log(output_q[k]+1e-10)
    #                         dist+=output_p[k]*(logp-logq)
    #                     kl=dist/output_p.shape[0]
    #                     sim=float(np.exp(-kl))
    #                     sim=max(0.0,sim)
    #                 sim_mat[i,j]=sim
    #     elif method == 'js_test':
    #         output_list = []
    #         for id in client_ids:
    #             client = self.clients[id]
    #             output= self.output_test(client)
    #             output_list.append(output)

    #         for i in range(num_clients):
    #             for j in range(num_clients):
    #                 if i == j:
    #                     sim_mat[i, j] = 1.0
    #                 else:
    #                     p = output_list[i]  
    #                     q = output_list[j]
    #                     M = 0.5 * (p + q)
    #                     kl_pm = np.sum(p * (np.log(p + 1e-10) - np.log(M + 1e-10)), axis=1)
    #                     kl_qm = np.sum(q * (np.log(q + 1e-10) - np.log(M + 1e-10)), axis=1)
    #                     js = 0.5 * np.mean(kl_pm + kl_qm)  

    #                     sim = float(np.exp(-js))
    #                     sim_mat[i, j] = max(0.0, sim)
    #     else:
    #         flattened_weights = []
    #         for weights in client_weights:
    #             arrs = [np.asarray(w, dtype=np.float64).ravel() for w in weights]
    #             if arrs:
    #                 flat = np.concatenate(arrs)
    #             else:
    #                 flat = np.asarray([], dtype=np.float64)
    #             flattened_weights.append(flat)
    #             # print(len(flattened_weights))

    #         log_probs = [None] * num_clients
    #         if need_log:
    #             for i, w in enumerate(flattened_weights):
    #                 if w.size == 0:
    #                     log_probs[i] = None
    #                 else:
    #                     log_probs[i] = self._log_softmax(w)
    #         for i in range(num_clients):
    #             for j in range(i, num_clients):
    #                 if i == j:
    #                     sim = 1.0
    #                 else:
    #                     w1 = flattened_weights[i]
    #                     w2 = flattened_weights[j]
    #                     if w1.size == 0 or w2.size == 0 or w1.size != w2.size:
    #                         sim = 0.0
    #                     else:
    #                         try:
    #                             if method == 'cosine':
    #                                 sim = self._cosine_similarity_flat(w1, w2)
    #                             elif method == 'kl':
    #                                 logp = log_probs[i]
    #                                 logq = log_probs[j]
    #                                 if logp is None or logq is None:
    #                                     raise ValueError("invalid log probs")
    #                                 kl = self._kl_from_log(logp, logq)
    #                                 sim = float(np.exp(-kl))
    #                             elif method == 'symkl':
    #                                 logp = log_probs[i]; logq = log_probs[j]
    #                                 if logp is None or logq is None:
    #                                     raise ValueError("invalid log probs")
    #                                 kl1 = self._kl_from_log(logp, logq)
    #                                 kl2 = self._kl_from_log(logq, logp)
    #                                 sim = float(np.exp(-0.5 * (kl1 + kl2)))
    #                             elif method == 'js':
    #                                 logp = log_probs[i]; logq = log_probs[j]
    #                                 if logp is None or logq is None:
    #                                     raise ValueError("invalid log probs")
    #                                 js = self._js_divergence_from_log(logp, logq)
    #                                 sim = 1.0 - np.sqrt(js / np.log(2.0))
    #                                 sim = float(np.clip(sim, 0.0, 1.0))
    #                             else:
    #                                 raise ValueError(f"Unknown method: {method}")
    #                         except Exception as e:
    #                             print("weight_similarity: fallback due to:", repr(e))
    #                             if fallback == 'cosine':
    #                                 sim = self._cosine_similarity_flat(w1, w2)
    #                             else:
    #                                 sim = 0.0
    #                     sim = max(0.0, sim)

    #                 sim_mat[i, j] = sim
    #                 sim_mat[j, i] = sim 

    #     row_sums = sim_mat.sum(axis=1, keepdims=True)
    #     row_sums[row_sums == 0] = 1.0
    #     normalized = sim_mat / row_sums
    #     return normalized

    
    # 这里修改参数为一个列表，用于直接调用融合方法
    def similarity_fusion(self, t, P_list):
        """相似性融合 P^c ← SimilarityFusion(P_g^c, P_w^c) (伪代码第15行)"""
        # 使用注意力融合方法
        return self.fuse_attention(t, P_list)
    
    
    def k_means_cluster(self, k, similarity_matrix, indices):
        """基于相似性矩阵的 K-means 聚类（先将相似性转换为距离，再用 MDS 嵌入到欧氏空间，最后做 KMeans）"""
        # 局部导入以避免模块未安装导致类导入失败
        try:
            from sklearn.manifold import MDS
            from sklearn.cluster import KMeans
            import numpy as np
        except Exception as e:
            print(f"Required sklearn modules missing: {e}. Falling back to random grouping.")
            # 回退到随机分组
            n = len(indices)
            if n == 0:
                return []
            if k <= 0:
                k = 1
            if k > n:
                k = n
            group_size = n // k
            groups = []
            for i in range(k):
                start = i * group_size
                if i == k - 1:
                    groups.append(indices[start:])
                else:
                    groups.append(indices[start:start + group_size])
            return [g for g in groups if g]

        # 基本边界情况
        n = len(indices)
        if n <= 1:
            return [indices]
        if k <= 0:
            k = 1
        if k > n:
            k = n

        try:
            # 将相似性转换为距离
            distance_matrix = 1.0 - similarity_matrix

            # 保证对称、对角为0
            distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
            np.fill_diagonal(distance_matrix, 0.0)

            # 处理数值问题：确保所有距离非负
            min_val = distance_matrix.min()
            if min_val < 0:
                distance_matrix = distance_matrix - min_val

            # 选择嵌入维度：不超过 n-1，且至少为 1，通常选择 min(k, n-1)
            n_components = max(1, min(k, n - 1))

            # 使用 MDS 将距离矩阵嵌入到欧氏空间
            # 注意：dissimilarity='precomputed' 表示传入的是距离矩阵
            mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=0, n_init=4, max_iter=300)
            embedding = mds.fit_transform(distance_matrix)

            # 如果嵌入失败或返回维度不匹配，退回到简单的替代（例如使用第一列作为特征）
            if embedding.shape[0] != n:
                raise RuntimeError("MDS embedding returned unexpected shape")

            # 在嵌入空间上运行 KMeans
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
            labels = kmeans.fit_predict(embedding)

            # 将标签映射回原始 indices
            groups = {}
            for i, lbl in enumerate(labels):
                groups.setdefault(int(lbl), []).append(indices[i])

            # 返回按组组成的列表（丢弃空组）
            return [g for g in groups.values() if g]

        except Exception as e:
            print(f"K-means clustering failed: {e}. Falling back to random grouping.")
            # 降级到简单均匀切分（保留顺序）
            group_size = n // k
            groups = []
            for i in range(k):
                start_idx = i * group_size
                if i == k - 1:
                    groups.append(indices[start_idx:])
                else:
                    groups.append(indices[start_idx:start_idx + group_size])
            return [g for g in groups if g]
    
    
    def hierarchical_clustering_from_similarity(self, similarity_matrix, indices):
        """基于相似性矩阵的层次聚类 C ← HierarchicalClustering(P, h)"""
        from scipy.cluster.hierarchy import linkage, fcluster
        
        if len(indices) <= 1:
            return [indices]
        
        try:
            # 将相似性转换为距离
            distance_matrix = 1.0 - similarity_matrix
            
            # 确保距离矩阵是对称的且对角线为0
            distance_matrix = (distance_matrix + distance_matrix.T) / 2
            np.fill_diagonal(distance_matrix, 0)
            
            # 转换为压缩距离矩阵
            from scipy.spatial.distance import squareform
            condensed_dist = squareform(distance_matrix)
            
            # 层次聚类
            linkage_matrix = linkage(condensed_dist, method='single')
            cluster_labels = fcluster(linkage_matrix, t=self.num_clusters, criterion='maxclust')
            
            # 分组
            groups = {}
            for i, label in enumerate(cluster_labels):
                if label not in groups:
                    groups[label] = []
                groups[label].append(indices[i])
            
            return list(groups.values())
            
        except Exception as e:
            print(f"Hierarchical clustering failed: {e}, using random grouping")
            # 降级到随机分组
            group_size = len(indices) // self.num_clusters
            groups = []
            for i in range(self.num_clusters):
                start_idx = i * group_size
                if i == self.num_clusters - 1:
                    groups.append(indices[start_idx:])
                else:
                    groups.append(indices[start_idx:start_idx + group_size])
            return [g for g in groups if g]

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        # 输出最终聚类结果和统计信息
        self.print_final_clustering_stats()

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientMUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            
    def stable_softmax_numpy(self,logits, eps= 1e-12):

        x = logits.astype(np.float64)
        if x.ndim == 1:
            x = x - np.max(x)
            ex = np.exp(x)
            s = np.sum(ex)
            return (ex / (s + eps)).astype(np.float64)
        else:
            x = x - np.max(x, axis=1, keepdims=True)
            ex = np.exp(x)
            s = np.sum(ex, axis=1, keepdims=True)
            return (ex / (s + eps)).astype(np.float64)

    def kl_divergence_between_samples(self,p_batch, q_batch, eps= 1e-12):

        p = np.asarray(p_batch, dtype=np.float64)
        q = np.asarray(q_batch, dtype=np.float64)
        assert p.shape == q.shape, "p and q must have same shape"
        p = np.clip(p, eps, None)
        q = np.clip(q, eps, None)
        kl_per_sample = np.sum(p * (np.log(p) - np.log(q)), axis=1)
        kl_avg = np.mean(kl_per_sample)  
        return float(kl_avg)

    def output_test_kld(self, client):
        if self.dataset=='HAR':
            mod=30
        elif self.dataset=='PAMAP2':
            mod=9
        elif self.dataset=='DC' or self.dataset =='HARBox':
            mod=5
        test_data = read_client_data(client.dataset, (self.num_clients + 1) % mod, is_train=False, few_shot=self.few_shot)
        test_loader = DataLoader(test_data, self.batch_size, drop_last=False, shuffle=True)
        outputs = []
        for i, (x, y) in enumerate(test_loader):
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
                out = client.model(x)
            else:
                x = x.to(self.device)
                out = client.model(x)

            out_np = out.detach().cpu().numpy()
            if out_np.ndim == 1:
                out_np = np.expand_dims(out_np, axis=0)

            probs = self.stable_softmax_numpy(out_np)
            outputs.append(probs)

        if len(outputs) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.concatenate(outputs, axis=0)  

    def compute_similarity_KLD(self, num_clients, client_ids):
        sim_mat = np.zeros((num_clients, num_clients), dtype=np.float64)
        output_list = []
        for id in client_ids:
            client = self.clients[id]
            output = self.output_test_kld(client)  
            if output.size == 0:
                print(f"Warning: client {id} has empty test outputs. Using small uniform distribution.")
                output = np.ones((1, 1), dtype=np.float64)
                output = output / np.sum(output)  
            output_list.append(output)

        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    sim = 1.0
                else:
                    p = output_list[i]  
                    q = output_list[j]  
                    if p.shape[1] != q.shape[1]:
                        minC = min(p.shape[1], q.shape[1])
                        p = p[:, :minC]
                        q = q[:, :minC]
                    p_mean = np.mean(p, axis=0)  
                    q_mean = np.mean(q, axis=0)  
                    kl = self.kl_divergence_between_samples(p_mean[np.newaxis, :], q_mean[np.newaxis, :])
                    sim = float(np.exp(-kl))
                    if not np.isfinite(sim):
                        sim = 0.0
                    sim = max(0.0, sim)
                sim_mat[i, j] = sim

        row_sums = sim_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = sim_mat / row_sums
        return normalized

    def compute_multisim_similarity(self):
        """计算MultiSim多度量相似性矩阵"""
        print("Computing MultiSim similarity matrices...")
        
        num_clients = len(self.selected_clients)
        self.gradient_similarity_dir_matrix = np.zeros((num_clients, num_clients))
        self.gradient_similarity_val_matrix=np.zeros((num_clients,num_clients))
        self.weight_similarity_matrix = np.zeros((num_clients, num_clients))
        
        # 收集所有客户端的梯度和权重签名
        client_gradients = []
        client_weights = []
        
        for client in self.selected_clients:
            gradient_sig = client.get_gradient_signature()
            weight_sig = client.get_weight_signature()
            client_gradients.append(gradient_sig)
            client_weights.append(weight_sig)
        
        # 计算梯度相似性矩阵
        for i in range(num_clients):
            for j in range(num_clients):
                if i == j:
                    self.gradient_similarity_dir_matrix[i][j] = 1.0
                    self.weight_similarity_matrix[i][j] = 1.0
                    
                else:
                    # 梯度方向相似性（余弦相似度）
                    grad_sim = self.selected_clients[i].compute_gradient__similarity(client_gradients[j])
                    self.gradient_similarity_dir_matrix[i][j] = grad_sim
                    
                    # 权重相似性（KL散度转换）
                    weight_sim = self.selected_clients[i].compute_weight_similarity(client_weights[j])
                    self.weight_similarity_matrix[i][j] = weight_sim
                    
                    
        
        # 使用注意力融合方法
        self.similarity_matrix = self.fuse_attention(round,[
            
            self.gradient_similarity_dir_matrix,
            self.gradient_similarity_val_matrix,
            self.weight_similarity_matrix
        ])
        
        print(f"Gradient similarity dirction matrix shape: {self.gradient_similarity_dir_matrix.shape}")
        print(f"Gradient similarity value matrix shape: {self.gradient_similarity_val_matrix.shape}")
        print(f"Weight similarity matrix shape: {self.weight_similarity_matrix.shape}")
        print(f"Fused similarity matrix shape: {self.similarity_matrix.shape}")
        
        # 输出相似性统计信息
        avg_grad_sim = np.mean(self.gradient_similarity_dir_matrix[np.triu_indices_from(self.gradient_similarity_dir_matrix, k=1)])
        avg_grad_sim = np.mean(self.gradient_similarity_val_matrix[np.triu_indices_from(self.gradient_similarity_val_matrix, k=1)])
        avg_weight_sim = np.mean(self.weight_similarity_matrix[np.triu_indices_from(self.weight_similarity_matrix, k=1)])
        avg_fused_sim = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
        
        print(f"Average gradient similarity: {avg_grad_sim:.4f}")
        print(f"Average weight similarity: {avg_weight_sim:.4f}")
        print(f"Average fused similarity: {avg_fused_sim:.4f}")

    def update_clusters(self):
        """基于相似性矩阵更新聚类"""
        if self.similarity_matrix is None:
            print("Similarity matrix not computed yet, skipping clustering")
            return
        
        print(f"Updating clusters using {self.clustering_method} method...")
        
        num_clients = len(self.selected_clients)
        
        if num_clients < self.num_clusters:
            print(f"Warning: Number of clients ({num_clients}) < number of clusters ({self.num_clusters})")
            # 每个客户端分配到不同的聚类
            for i, client in enumerate(self.selected_clients):
                self.client_clusters[client.id] = i
                client.set_cluster_id(i)
            return
        
        try:
            if self.clustering_method == 'hierarchical':
                # 使用层次聚类
                # 将相似性矩阵转换为距离矩阵
                distance_matrix = 1.0 - self.similarity_matrix
                
                # 确保距离矩阵是对称的且对角线为0
                distance_matrix = (distance_matrix + distance_matrix.T) / 2
                np.fill_diagonal(distance_matrix, 0)
                
                # 将距离矩阵转换为压缩格式用于层次聚类
                condensed_dist = pdist(distance_matrix, metric='precomputed')
                
                # 执行层次聚类
                linkage_matrix = linkage(condensed_dist, method='ward')
                cluster_labels = fcluster(linkage_matrix, t=self.num_clusters, criterion='maxclust')
                
            elif self.clustering_method == 'agglomerative':
                # 使用sklearn的聚合聚类
                # 将相似性转换为距离
                affinity_matrix = self.similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=self.num_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
                cluster_labels = clustering.fit_predict(affinity_matrix)
                cluster_labels += 1  # 使聚类标签从1开始
            
            else:
                raise ValueError(f"Unknown clustering method: {self.clustering_method}")
            
            # 更新客户端聚类分配
            for i, client in enumerate(self.selected_clients):
                cluster_id = int(cluster_labels[i])
                self.client_clusters[client.id] = cluster_id
                client.set_cluster_id(cluster_id)
            
            # 输出聚类结果
            self.print_clustering_info()
            
            # 检测异常客户端（可能的恶意客户端）
            self.detect_malicious_clients()
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            # 降级到随机分配
            for i, client in enumerate(self.selected_clients):
                cluster_id = (i % self.num_clusters) + 1
                self.client_clusters[client.id] = cluster_id
                client.set_cluster_id(cluster_id)

    def update_cluster_centers(self):
        """更新聚类中心"""
        print("Updating cluster centers...")
        
        # 按聚类分组客户端
        clusters = {}
        for client in self.selected_clients:
            cluster_id = client.get_cluster_id()
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client)
        
        # 计算每个聚类的中心
        self.cluster_centers = {}
        for cluster_id, clients_in_cluster in clusters.items():
            if not clients_in_cluster:
                continue
            
            # 获取聚类中所有客户端的模型参数
            cluster_params = []
            for client in clients_in_cluster:
                params = []
                for param in client.model.parameters():
                    params.append(param.data.flatten().cpu())
                cluster_params.append(torch.cat(params).numpy())
            
            # 计算平均值作为聚类中心
            if cluster_params:
                cluster_center = np.mean(cluster_params, axis=0)
                self.cluster_centers[cluster_id] = cluster_center
                
                # 将聚类中心发送给聚类内的客户端
                for client in clients_in_cluster:
                    client.set_cluster_weight(cluster_center)
        
        print(f"Updated {len(self.cluster_centers)} cluster centers")

    def multisim_aggregate(self):
        """MultiSim聚合：基于聚类的加权聚合 - 简化版用于baseline"""
        assert (len(self.uploaded_models) > 0)
        
        # 如果客户端聚类信息不存在，使用传统FedAvg
        if not hasattr(self, 'client_clusters') or not self.client_clusters:
            print("No clustering info, using FedAvg aggregation")
            self.aggregate_parameters()
            return
        
        # 按聚类分组上传的模型
        cluster_models = {}
        cluster_weights = {}
        
        for i, client_id in enumerate(self.uploaded_ids):
            cluster_id = self.client_clusters.get(client_id, 0)
            
            if cluster_id not in cluster_models:
                cluster_models[cluster_id] = []
                cluster_weights[cluster_id] = []
            
            cluster_models[cluster_id].append(self.uploaded_models[i])
            # 使用训练样本数作为权重
            cluster_weights[cluster_id].append(self.clients[client_id].train_samples)
        
        # 聚类内简单加权平均聚合
        cluster_aggregated = {}
        for cluster_id, models in cluster_models.items():
            if not models:
                continue
            
            weights = cluster_weights[cluster_id]
            total_weight = sum(weights)
            
            # 简单加权平均（不使用ADMM）
            aggregated_model = copy.deepcopy(models[0])
            for param in aggregated_model.parameters():
                param.data.zero_()
            
            for model, weight in zip(models, weights):
                weight_ratio = weight / total_weight
                for agg_param, model_param in zip(aggregated_model.parameters(), model.parameters()):
                    agg_param.data += model_param.data * weight_ratio
            
            cluster_aggregated[cluster_id] = (aggregated_model, total_weight)
        
        # 全局聚合：聚类间聚合
        if cluster_aggregated:
            # 计算全局聚合权重
            total_samples = sum(weight for _, weight in cluster_aggregated.values())
            
            # 初始化全局模型
            self.global_model = copy.deepcopy(list(cluster_aggregated.values())[0][0])
            for param in self.global_model.parameters():
                param.data.zero_()
            
            # 按聚类权重聚合
            for cluster_id, (model, cluster_weight) in cluster_aggregated.items():
                weight_ratio = cluster_weight / total_samples
                
                for global_param, cluster_param in zip(self.global_model.parameters(), model.parameters()):
                    global_param.data += cluster_param.data.clone() * weight_ratio
        
        else:
            # 如果没有聚类信息，使用传统FedAvg聚合
            self.aggregate_parameters()
        
        print(f"MultiSim aggregation completed with {len(cluster_aggregated)} clusters")

    def detect_malicious_clients(self):
        """基于相似性检测恶意客户端"""
        if self.similarity_matrix is None:
            return
        
        detected_malicious = []
        
        for i, client in enumerate(self.selected_clients):
            # 计算客户端与其他客户端的平均相似性
            similarities = self.similarity_matrix[i, :]
            avg_similarity = np.mean(similarities[similarities != 1.0])  # 排除自己
            
            # 如果平均相似性低于阈值，标记为可疑
            if avg_similarity < self.robustness_threshold:
                detected_malicious.append(client.id)
                print(f"Client {client.id} detected as potentially malicious (avg_sim: {avg_similarity:.4f})")
        
        # 更新恶意客户端列表
        self.malicious_clients.extend(detected_malicious)
        self.malicious_clients = list(set(self.malicious_clients))  # 去重
        
        if detected_malicious:
            print(f"Total detected malicious clients: {len(self.malicious_clients)}")
    
    
    
    

    def evaluate_personalized_models(self):
        """评估个性化模型性能"""
        print("\nEvaluating personalized models...")
        
        cluster_accuracies = {}
        total_personalized_acc = 0
        total_samples = 0
        
        for client in self.clients:
            if hasattr(client, 'evaluate_personalized'):
                acc, samples = client.evaluate_personalized()
                cluster_id = client.get_cluster_id()
                
                if cluster_id not in cluster_accuracies:
                    cluster_accuracies[cluster_id] = []
                
                cluster_accuracies[cluster_id].append(acc)
                total_personalized_acc += acc * samples
                total_samples += samples
        
        # 输出聚类级别的性能
        for cluster_id, accuracies in cluster_accuracies.items():
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"Cluster {cluster_id}: Avg Acc = {avg_acc:.2f}% ± {std_acc:.2f}%")
        
        # 输出全局个性化性能
        if total_samples > 0:
            global_personalized_acc = total_personalized_acc / total_samples
            print(f"Global Personalized Accuracy: {global_personalized_acc:.2f}%")

    def print_clustering_info(self):
        """输出聚类信息"""
        print("\nCurrent clustering assignment:")
        clusters = {}
        for client_id, cluster_id in self.client_clusters.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(client_id)
        
        for cluster_id, client_ids in clusters.items():
            print(f"Cluster {cluster_id}: Clients {client_ids} ({len(client_ids)} clients)")

    def print_final_clustering_stats(self):
        """输出最终聚类统计信息"""
        print("\n" + "="*50)
        print("FINAL MULTISIM CLUSTERING STATISTICS")
        print("="*50)
        
        # 聚类分布
        cluster_sizes = {}
        for cluster_id in self.client_clusters.values():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        print(f"Number of clusters formed: {len(cluster_sizes)}")
        for cluster_id, size in cluster_sizes.items():
            print(f"Cluster {cluster_id}: {size} clients")
        
        # 相似性统计
        if self.similarity_matrix is not None:
            avg_similarity = np.mean(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            max_similarity = np.max(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            min_similarity = np.min(self.similarity_matrix[np.triu_indices_from(self.similarity_matrix, k=1)])
            
            print(f"\nSimilarity Statistics:")
            print(f"Average similarity: {avg_similarity:.4f}")
            print(f"Maximum similarity: {max_similarity:.4f}")
            print(f"Minimum similarity: {min_similarity:.4f}")
        
        # 恶意客户端检测结果
        if self.malicious_clients:
            print(f"\nDetected malicious clients: {self.malicious_clients}")
        else:
            print(f"\nNo malicious clients detected")
        
        print("="*50)

    def _get_client_params(self, client):
        """获取客户端模型参数（列表格式，与原始论文一致）"""
        params = []
        for param in client.model.parameters():
            params.append(param.data.cpu().numpy().copy())
        return params
    
    def _get_model_params(self,model):
        params=[]
        for param in model.parameters():
            params.append(param.data().cpu().numpy().copy())
        return params
    
    
    def _save_results_to_dir(self, save_dir):
        """将训练结果保存到指定目录（覆盖父类方法）"""
        import h5py
        
        algo = self.dataset + "_" + self.algorithm
        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(save_dir, f"{algo}.h5")
            print(f"  Saving to: {file_path}")
            
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            print(f"  ✓ Saved accuracy history: {len(self.rs_test_acc)} rounds")



    def get_multisim_config(self):
        """获取MultiSim配置信息"""
        return {
            'num_clusters': self.num_clusters,
            'similarity_threshold': self.similarity_threshold,
            'fusion_weight_gradient': self.fusion_weight_gradient,
            'fusion_weight_weight': self.fusion_weight_weight,
            'clustering_method': self.clustering_method,
            'update_cluster_freq': self.update_cluster_freq,
            'personalization_rounds': self.personalization_rounds,
            'cluster_aggregation_weight': self.cluster_aggregation_weight,
            'robustness_threshold': self.robustness_threshold,
            'fuse_method': self.fuse_method
        }
    
    def Q_weight(self, now_t, t):
        weighted=np.exp(-((now_t-t)**2)*(self.sigma**2/2))
        return weighted
    
    def fuse_attention(self, t, similarity_matrices):
            """注意力机制融合方法 - 修正版，更贴近论文
               这里修改了Q，用高斯分布来去设置三个指标的权重
            """
            print("Using attention fusion method (corrected)...")
            
            if not similarity_matrices:
                return np.array([])
            matrices = [np.array(m) for m in similarity_matrices]
            
            # 将矩阵堆叠: [num_matrices, num_clients, num_clients]
            stacked_matrices = np.stack(matrices)
            num_matrices, num_clients, _ = stacked_matrices.shape
            
            K = V = stacked_matrices
            
            w_P_g_dir=self.Q_weight(t,self.t1)
            w_P_g_val=self.Q_weight(t, self.t2)
            w_P_w=self.Q_weight(t,self.t3)
            w_sum=w_P_g_dir+w_P_g_val+w_P_w
            w_P_g_dir/=w_sum
            w_P_g_val/=w_sum
            w_P_w/=w_sum
            Q=w_P_g_dir*matrices[0]+w_P_g_val*matrices[1]+w_P_w*matrices[2]
            
            # Q = np.mean(stacked_matrices, axis=0, keepdims=True) # Shape: [1, num_clients, num_clients]
            
            Q_flat = Q.reshape(1, -1)
            K_flat = K.reshape(num_matrices, -1)
            
            d_k = K_flat.shape[1]
            
            scores = np.dot(Q_flat, K_flat.T) 

            from scipy.special import softmax
            attention_weights = softmax(scores.flatten() / self.temperature) # Shape: [num_matrices],temperature include d_k
            
            print(f"Attention weights (corrected): {attention_weights}")

            reshaped_weights = attention_weights.reshape(num_matrices, 1, 1)
            fused_matrix = np.sum(reshaped_weights * V, axis=0)
            
            return fused_matrix
    
    def export_model(self, save_dir=None):
        """
        导出训练好的全局模型和簇模型
        
        Args:
            save_dir: 保存目录路径（如果为None，使用system目录下的results）
        
        Returns:
            dict: 包含导出文件路径的字典
        """
        if save_dir is None:
            # __file__ = /system/flcore/servers/servermul.py
            # 需要上移3层到 /system/
            system_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            save_dir = os.path.join(system_dir, "results", "fedmul_demo")
        
        os.makedirs(save_dir, exist_ok=True)
        
        export_info = {
            'global_model': None,
            'cluster_models': {},
            'client_clusters': self.client_clusters,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        
        print(f"  📁 Export directory: {save_dir}")
        
        # 导出全局模型
        global_model_path = os.path.join(save_dir, f"global_model_{export_info['timestamp']}.pt")
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'model_config': {
                'dataset': self.dataset,
                'num_classes': self.num_classes,
                'algorithm': 'FedMul'
            }
        }, global_model_path)
        export_info['global_model'] = global_model_path
        print(f"  ✓ Global model: global_model_{export_info['timestamp']}.pt")
        
        # 导出各簇模型（如果有）
        if hasattr(self, 'client_groups') and hasattr(self, 'cluster_similarity_matrices'):
            for group_id in range(len(self.client_groups)):
                cluster_model_path = os.path.join(save_dir, f"cluster_{group_id}_model_{export_info['timestamp']}.pt")
                # 获取簇内代表性客户端的模型（使用第一个客户端）
                representative_client_idx = self.client_groups[group_id][0]
                representative_client = self.clients[representative_client_idx]
                
                torch.save({
                    'model_state_dict': representative_client.model.state_dict(),
                    'cluster_id': group_id,
                    'client_indices': self.client_groups[group_id],
                    'similarity_matrix': self.cluster_similarity_matrices.get(group_id, None)
                }, cluster_model_path)
                export_info['cluster_models'][group_id] = cluster_model_path
                print(f"  ✓ Cluster {group_id} model: cluster_{group_id}_model_{export_info['timestamp']}.pt")
        
        # 保存元数据
        metadata_path = os.path.join(save_dir, f"export_metadata_{export_info['timestamp']}.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'timestamp': export_info['timestamp'],
                'dataset': self.dataset,
                'num_clients': self.num_clients,
                'num_clusters': len(self.client_groups) if hasattr(self, 'client_groups') else 0,
                'client_groups': self.client_groups if hasattr(self, 'client_groups') else [],
                'test_accuracy': self.rs_test_acc[-1] if self.rs_test_acc else 0.0
            }, f, indent=2)
        print(f"  ✓ Metadata: export_metadata_{export_info['timestamp']}.json")
        
        # Generate static clustering visualization
        self._generate_clustering_visualization(save_dir, export_info['timestamp'])
        
        return export_info
    
    def _generate_clustering_visualization(self, save_dir, timestamp):
        """Generate static clustering result visualization after training"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            from matplotlib.patches import Circle
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist, squareform
            
            if not hasattr(self, 'client_groups') or not hasattr(self, 'cluster_similarity_matrices'):
                print("Warning: No clustering information available, skipping visualization")
                return
            
            # Simulate client features based on cluster assignments
            np.random.seed(42)
            num_clients = self.num_clients
            client_features = []
            cluster_labels = []
            
            for group_id, client_indices in enumerate(self.client_groups):
                for client_idx in client_indices:
                    # Generate features centered around cluster
                    base_feature = np.array([group_id * 3.0, group_id * 2.5])
                    noise = np.random.randn(2) * 0.5
                    client_features.append(base_feature + noise)
                    cluster_labels.append(group_id)
            
            client_features = np.array(client_features)
            cluster_labels = np.array(cluster_labels)
            
            # Calculate similarity matrix
            similarity_matrix = 1.0 / (1.0 + squareform(pdist(client_features, 'euclidean')))
            linkage_matrix = linkage(client_features, method='ward')
            
            # Create figure
            fig = plt.figure(figsize=(20, 8), facecolor='#0a0a0a')
            gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3, width_ratios=[2, 1, 1])
            
            ax_clients = fig.add_subplot(gs[0], facecolor='#1a1a1a')
            ax_similarity = fig.add_subplot(gs[1], facecolor='#1a1a1a')
            ax_dendrogram = fig.add_subplot(gs[2], facecolor='#1a1a1a')
            
            # Define cluster colors
            cluster_colors_map = {
                0: '#ff6b6b',  # Red
                1: '#4ecdc4',  # Cyan
                2: '#45b7d1',  # Blue
                3: '#96ceb4',  # Green
                4: '#ffeaa7',  # Yellow
            }
            
            # 1. Client Feature Distribution with Clusters
            ax_clients.set_facecolor('#0a0a0a')
            ax_clients.set_xlim(-2, max(10, len(self.client_groups) * 3 + 2))
            ax_clients.set_ylim(-2, max(10, len(self.client_groups) * 2.5 + 2))
            
            # Draw grid
            for i in range(-2, int(ax_clients.get_xlim()[1]) + 1):
                ax_clients.axhline(y=i, color='#2a2a2a', linewidth=0.5, alpha=0.3)
                ax_clients.axvline(x=i, color='#2a2a2a', linewidth=0.5, alpha=0.3)
            
            # Draw cluster boundaries
            for cluster_id in range(len(self.client_groups)):
                cluster_indices = [i for i, l in enumerate(cluster_labels) if l == cluster_id]
                cluster_points = client_features[cluster_indices]
                color = cluster_colors_map.get(cluster_id, '#4ecdc4')
                
                if len(cluster_points) > 0:
                    center = cluster_points.mean(axis=0)
                    max_dist = np.max([np.linalg.norm(p - center) for p in cluster_points])
                    
                    boundary = Circle(center, max_dist + 0.6, 
                                    facecolor=color, alpha=0.15, zorder=0,
                                    edgecolor=color, linewidth=3, linestyle='--')
                    ax_clients.add_patch(boundary)
            
            # Draw clients
            for i, (x, y) in enumerate(client_features):
                color = cluster_colors_map.get(cluster_labels[i], '#4ecdc4')
                
                # Glow effect
                glow = Circle((x, y), 0.4, color=color, alpha=0.3, zorder=1)
                ax_clients.add_patch(glow)
                
                # Client circle
                client_circle = Circle((x, y), 0.25, facecolor=color, 
                                      edgecolor='white', linewidth=2, alpha=0.9, zorder=3)
                ax_clients.add_patch(client_circle)
                
                ax_clients.text(x, y, f'C{i}', ha='center', va='center',
                               fontsize=10, color='white', fontweight='bold', zorder=4)
                
                # Cluster label
                ax_clients.text(x, y-0.7, f'Cluster {cluster_labels[i]}', 
                               ha='center', va='top', fontsize=8,
                               color=color, fontweight='bold', 
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='#1a1a1a', 
                                       edgecolor=color, alpha=0.9, linewidth=1.5))
            
            ax_clients.set_xlabel('Feature Dimension 1', fontsize=13, color='white', fontweight='bold')
            ax_clients.set_ylabel('Feature Dimension 2', fontsize=13, color='white', fontweight='bold')
            ax_clients.set_title('Client Clustering Result (MultiSim)', fontsize=15, 
                                fontweight='bold', color='#00ff88', pad=15)
            ax_clients.tick_params(colors='white', labelsize=10)
            for spine in ax_clients.spines.values():
                spine.set_edgecolor('#4a4a4a')
                spine.set_linewidth(2)
            
            # 2. Improved Similarity Matrix (block-sorted by cluster)
            # Sort clients by cluster for better visualization
            sorted_indices = np.argsort(cluster_labels)
            sorted_matrix = similarity_matrix[sorted_indices, :][:, sorted_indices]
            sorted_labels = cluster_labels[sorted_indices]
            
            im = ax_similarity.imshow(sorted_matrix, cmap='RdYlGn', aspect='auto',
                                     interpolation='nearest', vmin=0, vmax=1)
            ax_similarity.set_title('Similarity Matrix\n(Sorted by Cluster)', fontsize=13, 
                                   fontweight='bold', color='#00ddff', pad=10)
            ax_similarity.set_xlabel('Client ID (Sorted)', fontsize=11, color='white')
            ax_similarity.set_ylabel('Client ID (Sorted)', fontsize=11, color='white')
            
            # Add cluster boundaries on matrix
            cluster_boundaries = []
            for i in range(1, len(sorted_labels)):
                if sorted_labels[i] != sorted_labels[i-1]:
                    cluster_boundaries.append(i - 0.5)
            
            for boundary in cluster_boundaries:
                ax_similarity.axhline(y=boundary, color='#ff6b35', linewidth=2, alpha=0.8)
                ax_similarity.axvline(x=boundary, color='#ff6b35', linewidth=2, alpha=0.8)
            
            ax_similarity.tick_params(colors='white', labelsize=9)
            for spine in ax_similarity.spines.values():
                spine.set_edgecolor('#4a4a4a')
                spine.set_linewidth(2)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax_similarity, fraction=0.046, pad=0.04)
            cbar.set_label('Similarity Score', color='white', fontsize=10, fontweight='bold')
            cbar.ax.tick_params(colors='white', labelsize=9)
            
            # 3. Dendrogram
            dend = dendrogram(linkage_matrix, ax=ax_dendrogram, 
                             color_threshold=linkage_matrix[-len(self.client_groups)+1, 2],
                             above_threshold_color='#4a4a4a',
                             no_labels=True)
            
            ax_dendrogram.set_title('Hierarchical Clustering\nDendrogram', 
                                   fontsize=13, fontweight='bold', 
                                   color='#ffd700', pad=10)
            ax_dendrogram.set_xlabel('Client Index', fontsize=11, color='white')
            ax_dendrogram.set_ylabel('Distance', fontsize=11, color='white')
            ax_dendrogram.tick_params(colors='white', labelsize=9)
            ax_dendrogram.set_facecolor('#1a1a1a')
            for spine in ax_dendrogram.spines.values():
                spine.set_edgecolor('#4a4a4a')
                spine.set_linewidth(2)
            
            # Add cluster count annotation
            bbox_props = dict(boxstyle='round,pad=0.6', facecolor='#ff6b35', 
                             edgecolor='white', linewidth=2, alpha=0.95)
            ax_dendrogram.text(0.5, 0.97, f'Clusters: {len(self.client_groups)}',
                              transform=ax_dendrogram.transAxes,
                              ha='center', va='top', fontsize=12,
                              bbox=bbox_props, color='white', fontweight='bold')
            
            # Overall title
            fig.suptitle(f'FedMul Training Results - Clustering Visualization', 
                        fontsize=18, fontweight='bold', color='#00ff88', y=0.98)
            
            # Save figure
            output_path = os.path.join(save_dir, f"clustering_result_{timestamp}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
            plt.close()
            
            print(f"  ✓ Clustering visualization: clustering_result_{timestamp}.png")
            
        except Exception as e:
            print(f"Warning: Failed to generate clustering visualization: {e}")
            import traceback
            traceback.print_exc()
    
    def run_local_inference_after_training(self, num_rounds=20, num_clients_to_use=10, save_dir="results/fedmul_demo"):
        """
        训练完成后，使用训练好的客户端模型进行本地推理测试
        每轮随机采样测试集的子集，模拟真实在线服务场景
        
        Args:
            num_rounds: 推理轮数
            num_clients_to_use: 使用多少个客户端参与推理（默认10个）
            save_dir: 保存目录
        """
        import json
        import random
        from collections import defaultdict
        
        # 随机选择指定数量的客户端参与推理
        selected_clients_for_inference = random.sample(self.clients, min(num_clients_to_use, len(self.clients)))
        
        print(f"\n{'='*70}")
        print(f"🚀 INFERENCE FUNCTION CALLED!")
        print(f"Starting Local Inference with Trained Client Models")
        print(f"Simulating online service with random sampling (80-100% of test data per round)")
        print(f"{'='*70}")
        print(f"📍 Save directory: {save_dir}")
        print(f"📊 Number of rounds: {num_rounds}")
        print(f"👥 Number of clients (selected for inference): {len(selected_clients_for_inference)}/{len(self.clients)}")
        print(f"🔧 Device: {self.device}")
        print(f"{'='*70}\n")
        
        inference_logs = []
        service_metrics = defaultdict(list)
        
        # 设置随机种子，但每轮使用不同的种子
        base_seed = 42
        
        for round_idx in range(num_rounds):
            print(f"\n--- Inference Round {round_idx + 1}/{num_rounds} ---")
            round_start_time = time.time()
            round_results = []
            
            # 用于收集全局预测分布
            all_predictions = []
            
            # 每轮使用不同的随机种子
            round_seed = base_seed + round_idx
            random.seed(round_seed)
            np.random.seed(round_seed)
            
            # 只对选中的客户端进行推理
            for client in selected_clients_for_inference:
                # 每个客户端在自己的测试数据上进行推理
                client.model.eval()
                
                test_acc = 0.0
                test_num = 0
                y_prob = []
                y_true = []
                
                # 加载完整测试数据
                testloader = client.load_test_data()
                
                # 随机采样80%-100%的测试数据（模拟在线服务中不同批次的请求）
                sample_ratio = random.uniform(0.8, 1.0)
                all_batches = list(testloader)
                num_batches_to_sample = max(1, int(len(all_batches) * sample_ratio))
                sampled_batches = random.sample(all_batches, num_batches_to_sample)
                
                with torch.no_grad():
                    for x, y in sampled_batches:
                        if type(x) == type([]):
                            x[0] = x[0].to(self.device)
                        else:
                            x = x.to(self.device)
                        y = y.to(self.device)
                        
                        output = client.model(x)
                        predictions = torch.argmax(output, dim=1)
                        
                        test_acc += (torch.sum(predictions == y)).item()
                        test_num += y.shape[0]
                        
                        # 收集预测结果用于分布统计
                        all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                        
                        y_prob.append(torch.softmax(output, dim=1).detach().cpu().numpy())
                        y_true.append(y.detach().cpu().numpy())
                
                accuracy = test_acc / test_num if test_num > 0 else 0.0
                
                # 获取客户端的簇ID
                cluster_id = None
                for group_id, group_clients in enumerate(self.client_groups):
                    if client.id in group_clients:
                        cluster_id = group_id
                        break
                
                # 计算平均置信度
                all_probs = np.concatenate(y_prob, axis=0) if y_prob else np.array([])
                avg_confidence = np.mean(np.max(all_probs, axis=1)) if len(all_probs) > 0 else 0.0
                
                # 记录客户端推理结果
                client_result = {
                    'client_id': client.id,
                    'cluster_id': cluster_id,
                    'round': round_idx,
                    'timestamp': time.time(),
                    'accuracy': float(accuracy),
                    'test_samples': test_num,
                    'avg_confidence': float(avg_confidence),
                    'latency_ms': (time.time() - round_start_time) * 1000 / len(self.clients),
                    'is_online': True
                }
                round_results.append(client_result)
            
            # 计算全局预测分布
            prediction_distribution = {}
            if all_predictions:
                unique_predictions, counts = np.unique(all_predictions, return_counts=True)
                for pred, count in zip(unique_predictions, counts):
                    prediction_distribution[int(pred)] = int(count)
                print(f"  ✓ Collected {len(all_predictions)} predictions, {len(prediction_distribution)} unique classes")
                print(f"  📊 Prediction distribution: {prediction_distribution}")
            else:
                print(f"  ⚠️  WARNING: No predictions collected in this round!")
            
            # 聚合本轮统计信息
            round_summary = {
                'round': round_idx,
                'timestamp': time.time(),
                'num_active_clients': len(round_results),
                'total_predictions': sum(r['test_samples'] for r in round_results),
                'avg_accuracy': float(np.mean([r['accuracy'] for r in round_results])),
                'avg_confidence': float(np.mean([r['avg_confidence'] for r in round_results])),
                'avg_latency_ms': float(np.mean([r['latency_ms'] for r in round_results])),
                'client_results': round_results,
                'global_prediction_distribution': prediction_distribution
            }
            inference_logs.append(round_summary)
            
            # 记录服务指标
            service_metrics['round'].append(round_idx)
            service_metrics['avg_accuracy'].append(round_summary['avg_accuracy'])
            service_metrics['avg_latency'].append(round_summary['avg_latency_ms'])
            service_metrics['total_requests'].append(round_summary['total_predictions'])
            
            round_time = time.time() - round_start_time
            print(f"Round completed in {round_time:.2f}s")
            print(f"Avg Accuracy: {round_summary['avg_accuracy']:.4f}")
            print(f"Avg Confidence: {round_summary['avg_confidence']:.4f}")
        
        # 保存推理日志
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, 'inference_logs.json')
        
        export_data = {
            'inference_logs': inference_logs,
            'service_metrics': dict(service_metrics),
            'client_status': [
                {
                    'client_id': c.id,
                    'cluster_id': next((gid for gid, g in enumerate(self.client_groups) if c.id in g), None),
                    'is_online': True,
                    'total_inferences': num_rounds,
                    'avg_latency_ms': np.mean([log['avg_latency_ms'] for log in inference_logs])
                } for c in selected_clients_for_inference  # 只保存参与推理的客户端状态
            ]
        }
        
        with open(log_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n{'='*70}")
        print("Local Inference Completed!")
        print(f"{'='*70}")
        print(f"✓ Inference logs saved to: {log_path}")
        print(f"  - Total rounds: {num_rounds}")
        print(f"  - Clients tested: {len(selected_clients_for_inference)}")
        print(f"  - Overall avg accuracy: {np.mean([log['avg_accuracy'] for log in inference_logs]):.4f}")
        print(f"{'='*70}\n")
        
        return log_path
    

    def evaluate_cluster(self, acc=None, loss=None):
        ids,num_samples,tot_correct,tot_auc,_,_,_=self.test_metrics()
        ids_train,num_samples_train,losses = self.train_metrics()
        cluster_acc=[]
        for group_id,group_client_indices in enumerate(self.client_groups):
            group_test_acc=0
            group_test_auc=0
            group_train_loss=0
            group_accs=[]
            group_aucs=[]
            group_benign_test_acc=0
            for idx in group_client_indices:
                group_test_acc+=tot_correct[idx]/num_samples[idx]
                group_test_auc=tot_auc[idx]/num_samples[idx]
                group_train_loss=losses[idx]/num_samples_train[idx]
                group_accs.append(tot_correct[idx]/num_samples[idx])
                group_aucs.append(tot_auc[idx]/num_samples[idx])
            group_test_acc/=len(group_client_indices)
            group_test_auc/=len(group_client_indices)
            group_train_loss/=len(group_client_indices)
        
            cluster_acc.append(group_test_acc)
            print(f"Group {group_id} evluate result:")
            print(group_client_indices)
            print("Averaged Train Loss: {:.4f}".format(group_train_loss))
            print("Averaged Test Accuracy: {:.4f}".format(group_test_acc))
            print("Averaged Test AUC: {:.4f}".format(group_test_auc))
            # self.print_(test_acc, train_acc, train_loss)
            print("Std Test Accuracy: {:.4f}".format(np.std(group_accs)))
            print("Std Test AUC: {:.4f}".format(np.std(group_aucs)))
            
        return cluster_acc
            
    def compute_top_k_score(self,A, k):

        A = np.asarray(A)
        n, m = A.shape
        assert n == m, "此实现假设方阵"

        if k <= 0:
            return np.zeros(n, dtype=float)

        k = min(k, n - 1)

        C = A.copy().astype(float)

        diag = np.diag(C).copy()

        zero_mask = (diag == 0)
        if np.any(zero_mask):
            safe_diag = diag.copy()
            safe_diag[zero_mask] = 1.0
        else:
            safe_diag = diag

        C = C / safe_diag[:, None]

        B = C.copy()
        idx = np.arange(n)
        B[idx, idx] = -np.inf
        topk = np.partition(B, -k, axis=1)[:, -k:]
        return np.sum(topk, axis=1)/k
    
    def cluster_top_k_fill(self,b, idx, a):
        b = np.asarray(b)
        idx = np.asarray(idx)
        a = np.asarray(a)
        b[idx] = a
        return b
    
    
    def receive_models_mulmod(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_gradients= []
        tot_samples = 0
        for i,client in enumerate(active_clients):
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples*self.top_k_score[client.id]**self.gamma
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples*self.top_k_score[client.id]**self.gamma)
                print(client.id,self.top_k_score[client.id],self.uploaded_weights[-1])
                self.uploaded_models.append(client.model)
                self.uploaded_gradients.append(client.gradient)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        if len(self.uploaded_weights)==1:
            self.uploaded_weights[0]=1
            
    def mod_global_matrix(self):
        global_matrix = np.asarray(self.global_similarity_matrix, dtype=float)
        score = np.asarray(self.top_k_score, dtype=float)

        n = len(score)
        assert global_matrix.shape == (n, n), "score长度应与global_matrix维度一致"

        weight = 1.0 - np.abs(score[:, None] - score[None, :])

        B = global_matrix * weight**2

        row_sums = B.sum(axis=1, keepdims=True)
        zero_rows = (row_sums == 0).flatten()
        row_sums[zero_rows, 0] = 1.0

        B_normalized = B / row_sums
        return B_normalized

    # def mod_global_matrix(self):
    #     global_matrix = np.asarray(self.global_similarity_matrix, dtype=float)
    #     score = np.asarray(self.top_k_score, dtype=float)

    #     n = len(score)
    #     assert global_matrix.shape == (n, n), "score长度应与global_matrix维度一致"

    #     # =========================
    #     # Step 1: 第一次对角线标准化
    #     # =========================
    #     diag1 = np.diag(global_matrix).copy()
    #     diag1[diag1 == 0] = 1.0   # 防止除 0

    #     G_norm1 = global_matrix / diag1[:, None]

    #     # =========================
    #     # Step 2: score 差异权重
    #     # =========================
    #     weight = 1.0 - np.abs(score[:, None] - score[None, :])
    #     weight = np.clip(weight, 0.0, 1.0)

    #     B = G_norm1 * weight

    #     # =========================
    #     # Step 3: 第二次对角线标准化
    #     # =========================
    #     diag2 = np.diag(B).copy()
    #     diag2[diag2 == 0] = 1.0   # 防止除 0

    #     B_norm2 = B / diag2[:, None]

    #     return B_norm2
    
    
# attack_result_saver_pretty.py
import os
import json
import threading
from datetime import datetime
import re
from typing import List, Any

_lock = threading.Lock()

def _now_str():
    tz = pytz.timezone("Asia/Shanghai")
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

def _safe_filename_part(s: Any) -> str:
    return re.sub(r'[^0-9A-Za-z._-]', '_', str(s))

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _format_json_block(data: dict) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2)

    # 单行压缩字段
    for key in ["malicious_client", "group_client_idx"]:
        pattern = rf'"{key}": \[\n\s*(.*?)\n\s*\]'
        matches = re.findall(pattern, text, re.S)

        for m in matches:
            compact = "[" + " ".join(line.strip().rstrip(",") + "," for line in m.splitlines())
            compact = compact.rstrip(",") + "]"
            text = re.sub(rf'"{key}": \[\n\s*{re.escape(m)}\n\s*\]',
                          f'"{key}": {compact}', text)

    # ✅ clients_acc：每5个一行
    keys = ["clients_acc","best_clients_acc","top_k_score"]
    for key in keys:
        if key in data and isinstance(data[key], list):
            lst = data[key]
            grouped = [lst[i:i+5] for i in range(0, len(lst), 5)]

            formatted = '[\n'
            for g in grouped:
                formatted += "  " + ", ".join(map(str, g)) + ",\n"
            formatted = formatted.rstrip(",\n") + "\n]"

            text = re.sub(
                rf'"{key}": \[\n.*?\n\s*\]',
                f'"{key}": {formatted}',
                text,
                flags=re.S
            )

    return text

                
def save_attack_results(
    gamma:int,
    malicious_ratio:float,
    topk:float,
    attack_method: str,
    cluster_method: str,
    test_acc: Any,
    cluster_acc: List[Any],
    group_client_idx: List[Any],
    malicious_client: Any,
    benign_acc: Any,
    top_k_score:List[Any],
    clients_acc:List[Any],
    out_dir: str = "result",
    filename: str = None,
) -> str:
    """
    按高可读格式保存攻击实验结果。
    """

    if not isinstance(cluster_acc, (list, tuple)):
        cluster_acc = [cluster_acc]
    if not isinstance(group_client_idx, (list, tuple)):
        group_client_idx = [group_client_idx]

    if len(cluster_acc) != len(group_client_idx):
        raise ValueError("cluster_acc 与 group_client_idx 长度必须一致")

    if filename:
        fname = filename
    else:
        fname = f"{_safe_filename_part(gamma)}_{_safe_filename_part(malicious_ratio)}_{_safe_filename_part(topk)}_{_safe_filename_part(attack_method)}_{_safe_filename_part(cluster_method)}.log"

    filepath = os.path.join(out_dir, fname)
    _ensure_dir(filepath)

    meta = {
        "record_type": "META",
        "time": _now_str(),
        "attack_method": attack_method,
        "cluster_method": cluster_method,
        "test_acc": test_acc,
        "benign_acc": benign_acc,
        "malicious_client": malicious_client,
        "top_k_score":top_k_score,
        "clients_acc":clients_acc
    }

    client_blocks = []
    for g_idx, c_acc in zip(group_client_idx, cluster_acc):
        client_blocks.append({
            "record_type": "CLIENT",
            "group_client_idx": g_idx,
            "cluster_acc": c_acc
        })

    with _lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(_format_json_block(meta) + "\n\n")
            for blk in client_blocks:
                f.write(_format_json_block(blk) + "\n\n")
            f.flush()

    return filepath
