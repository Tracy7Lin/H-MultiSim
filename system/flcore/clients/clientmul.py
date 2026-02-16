import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.attacks import BaseAttack, ClientAttackUtils, TriggerUtils, BadPFLAttack, NeurotoxinAttack, DBAAttack,Label_Poison_Attack,Random_Updates_Attack,Inner_Product_Attack,Model_Replace_Attack
from sklearn.metrics.pairwise import cosine_similarity


class clientMUL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.cluster_id = -1  
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.7)
        self.gradient_history = []  
        self.weight_history = []   
        self.max_history_len = getattr(args, 'max_history_len', 5)
        self.neighbor_models = {}
        self.fuse_method = 'attention'

        self.personalization_weight = getattr(args, 'personalization_weight', 0.5)
        self.cluster_weight = None 
        
        self.attack_utils = ClientAttackUtils(self.device, self.dataset)
        self.trigger_utils = TriggerUtils()
        self.base_attack = BaseAttack(args, self.device)
        
        if hasattr(args, 'attack') and args.attack:
            if args.attack.lower() == 'badpfl':
                self.attack_method = BadPFLAttack(args, self.device)
                self.model.badpfl_attack = self.attack_method
            elif args.attack.lower() == 'neurotoxin':
                self.attack_method = NeurotoxinAttack(args, self.device)
            elif args.attack.lower() == 'dba':
                self.attack_method = DBAAttack(args, self.device)
            elif args.attack.lower() == 'model_replacement':
                self.attack_method = args.attack.lower() 
            elif args.attack.lower() == 'badnets':
                self.attack_method = args.attack.lower()  
            elif args.attack.lower() == 'label_poison_attack':
                self.attack_method = Label_Poison_Attack(args,self.device)
            elif args.attack.lower() == 'random_updates_attack':
                self.attack_method = Random_Updates_Attack(args,self.device)
            elif args.attack.lower() == 'inner_product_attack':
                self.attack_method = Inner_Product_Attack(args,self.device)
            elif args.attack.lower() == 'model_replace_attack':
                self.attack_method = Model_Replace_Attack(args,self.device)
            else:
                self.attack_method = None
        else:
            self.attack_method = None
        
        print(f"Client {self.id} initialized with MultiSim framework")
    
    def set_neighbor_models(self, neighbor_models_state_dicts):
        self.neighbor_models = {}
        for client_id, state_dict in neighbor_models_state_dicts.items():
            if client_id != self.id: 
                model = copy.deepcopy(self.model)
                model.load_state_dict(state_dict)
                model.to(self.device)
                self.neighbor_models[client_id] = model

    def train(self, is_selected, F=None, omega=None, u=None, cluster_idx=None):
        """标准训练方法（用于初始训练阶段）"""
        if not is_selected:
            return
            
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()
        
        # 记录训练前的模型参数
        pre_train_params = self._get_model_params()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 标准损失计算
                output = self.model(x)
                loss = self.loss(output, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # 记录训练后的梯度和权重变化
        post_train_params = self._get_model_params()
        gradient = self._compute_gradient_difference(pre_train_params, post_train_params)
        self._update_gradient_history(gradient)
        self._update_weight_history(post_train_params)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_with_similarity_constraint(self, similarity_matrix, client_indices_in_cluster):
        trainloader = self.load_train_data()
        self.model.train()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # for epoch in range(max_local_epochs):
        for batch_idx, (x, y) in enumerate(trainloader):
            if isinstance(x, list):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)

            output = self.model(x)
            loss = self.loss(output, y)

            my_cluster_idx = client_indices_in_cluster.index(self.id)
            
            if similarity_matrix is not None and self.neighbor_models:
                similarity_loss = self._compute_similarity_constraint_loss(
                    similarity_matrix,
                    my_cluster_idx,
                    client_indices_in_cluster
                )
                loss += self.alpha * similarity_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        
    
    def _compute_similarity_constraint_loss(self, similarity_matrix, my_cluster_idx, client_indices_in_cluster):
        my_params = list(self.model.parameters())
        total_constraint_loss = torch.tensor(0.0, device=self.device)
        
        num_neighbors = 0
        for neighbor_matrix_idx, neighbor_global_id in enumerate(client_indices_in_cluster):
            if neighbor_global_id == self.id:
                continue
                
            neighbor_model = self.neighbor_models.get(neighbor_global_id)
            if neighbor_model:
                num_neighbors += 1
                similarity_pik = similarity_matrix[my_cluster_idx, neighbor_matrix_idx]
                neighbor_params = [p.detach() for p in neighbor_model.parameters() if p.requires_grad]
                
                l2_distance_sq = torch.tensor(0.0, device=self.device)
                for my_p, neighbor_p in zip(my_params, neighbor_params):
                    l2_distance_sq += torch.norm(my_p - neighbor_p, p=2) ** 2
                
                total_constraint_loss += similarity_pik * l2_distance_sq

        return total_constraint_loss / num_neighbors if num_neighbors > 0 else total_constraint_loss

    def get_multisim_stats(self):
        return {
            'client_id': self.id,
            'cluster_id': self.cluster_id,
            'gradient_history_len': len(self.gradient_history),
            'weight_history_len': len(self.weight_history),
            'personalization_weight': self.personalization_weight,
            'fuse_method': self.fuse_method
        }

    def _get_model_params(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.flatten())
        return torch.cat(params).cpu().numpy()

    def _compute_gradient_difference(self, pre_params, post_params):
        return post_params - pre_params

    def _update_gradient_history(self, gradient):
        """更新梯度历史"""
        self.gradient_history.append(gradient)
        if len(self.gradient_history) > self.max_history_len:
            self.gradient_history.pop(0)

    def _update_weight_history(self, weights):
        """更新权重历史"""
        self.weight_history.append(weights)
        if len(self.weight_history) > self.max_history_len:
            self.weight_history.pop(0)

    def _compute_personalization_loss(self):
        """计算个性化损失（与聚类中心的距离）"""
        if self.cluster_weight is None:
            return torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        current_params = []
        cluster_params = []
        
        param_idx = 0
        for param in self.model.parameters():
            param_flat = param.flatten()
            current_params.append(param_flat)
            
            # 从cluster_weight中提取对应参数
            param_size = param_flat.size(0)
            cluster_param = self.cluster_weight[param_idx:param_idx + param_size]
            cluster_params.append(torch.tensor(cluster_param, device=self.device, dtype=torch.float32))
            param_idx += param_size
        
        # 计算MSE损失
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for curr_param, cluster_param in zip(current_params, cluster_params):
            loss += torch.nn.functional.mse_loss(curr_param, cluster_param)
        
        return loss

    def get_gradient_signature(self):
        """获取客户端的梯度签名"""
        if not self.gradient_history:
            return np.array([])
        
        # 使用最近的梯度作为签名
        return self.gradient_history[-1]

    def get_weight_signature(self):
        """获取客户端的权重签名"""
        if not self.weight_history:
            return np.array([])
        
        # 使用最近的权重作为签名
        return self.weight_history[-1]

    def compute_gradient_similarity(self, other_gradient):
        """计算与其他客户端的梯度相似性"""
        if not self.gradient_history or len(other_gradient) == 0:
            return 0.0
        
        my_gradient = self.gradient_history[-1]
        if len(my_gradient) != len(other_gradient):
            return 0.0
        
        # 归一化处理
        my_norm = np.linalg.norm(my_gradient)
        other_norm = np.linalg.norm(other_gradient)
        
        if my_norm == 0 or other_norm == 0:
            return 0.0
        
        my_gradient_norm = my_gradient / my_norm
        other_gradient_norm = other_gradient / other_norm
        
        # 余弦相似度
        similarity = np.dot(my_gradient_norm, other_gradient_norm)
        return max(0.0, similarity)  # 确保非负

    def compute_weight_similarity(self, other_weights):
        """计算与其他客户端的权重相似性（基于KL散度）"""
        if not self.weight_history or len(other_weights) == 0:
            return 0.0
        
        my_weights = self.weight_history[-1]
        if len(my_weights) != len(other_weights):
            return 0.0
        
        # 使用简化的KL散度计算
        try:
            # 将权重转换为概率分布
            my_prob = self._weights_to_prob(my_weights)
            other_prob = self._weights_to_prob(other_weights)
            
            # 计算KL散度
            kl_div = self._compute_kl_divergence(my_prob, other_prob)
            
            # 转换为相似性（越小的KL散度表示越相似）
            similarity = np.exp(-kl_div)
            return similarity
        except:
            # 如果KL散度计算失败，使用余弦相似度
            return self._compute_cosine_similarity(my_weights, other_weights)

    def _weights_to_prob(self, weights):
        """将权重转换为概率分布"""
        # 使用softmax转换
        weights_shifted = weights - np.max(weights)  # 数值稳定性
        exp_weights = np.exp(weights_shifted)
        return exp_weights / np.sum(exp_weights)

    def _compute_kl_divergence(self, p, q):
        """计算KL散度"""
        # 添加小的epsilon避免log(0)
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        return np.sum(p * np.log(p / q))

    def _compute_cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def set_cluster_weight(self, cluster_weight):
        """设置聚类权重"""
        self.cluster_weight = cluster_weight

    def set_cluster_id(self, cluster_id):
        """设置聚类ID"""
        self.cluster_id = cluster_id
        print(f"Client {self.id} assigned to cluster {cluster_id}")

    def get_cluster_id(self):
        """获取聚类ID"""
        return self.cluster_id

    def is_malicious_detected(self, similarity_scores):
        """基于相似性分数检测是否为恶意客户端"""
        if not similarity_scores:
            return False
        
        avg_similarity = np.mean(similarity_scores)
        return avg_similarity < self.similarity_threshold

    def evaluate_personalized(self):
        """评估个性化模型性能"""
        testloader = self.load_test_data()
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy, total

    def evaluate(self):
        """基本评估方法（与clientAVG保持一致）"""
        testloader = self.load_test_data()
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x)
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        accuracy = 100. * correct / total
        return accuracy
    
    def train_malicious(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()} 
        if is_selected:
            trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                              noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
            self.model.train()   
            
            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            for step in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)  
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    if trigger is None or pattern is None:
                        if hasattr(self, 'attack_method') and self.attack_method and hasattr(self.attack_method, 'poison_batch'):
                            try:
                                poison_ratio_float = poison_ratio / len(x)
                                poisoned_x, poisoned_y = self.attack_method.poison_batch(
                                    data=x, 
                                    labels=y, 
                                    client_model=self.model, 
                                    poison_ratio=poison_ratio_float
                                )
                                x, y = poisoned_x, poisoned_y
                            except Exception as e:
                                print(f"Bad-PFL poison_batch error: {e}")
                    
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if oneshot == 1 and clip_rate > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)

            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
    
    def train_malicious_model_replace(self,is_selected,oneshot,clip_rate):
        last_local_model={name:param.clone() for name, param in self.model.named_parameters()}
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()
            
            start_time=time.time()
            max_local_epochs=self.local_epochs
            
            if self.train_slow:
                max_local_epochs=np.random.randint(1,max_local_epochs//2)
                
            for step in range(max_local_epochs):
                for i,(x,y) in enumerate(trainloader):
                    if type(x)==type([]):
                        x[0]=x[0].to(self.device)
                    else:
                        x=x.to(self.device)
                    y=y.to(self.device)
                    
                    output=self.model(x)
                    loss=self.loss(output,y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            if oneshot == 1 and clip_rate > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)
            
            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
    
    def train_malicious_inner_product(self,is_selected,oneshot,clip_rate):
        last_local_model={name:param.clone() for name, param in self.model.named_parameters()}
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()
            
            start_time=time.time()
            max_local_epochs=self.local_epochs
            
            if self.train_slow:
                max_local_epochs=np.random.randint(1,max_local_epochs//2)
                
            for step in range(max_local_epochs):
                for i,(x,y) in enumerate(trainloader):
                    if type(x)==type([]):
                        x[0]=x[0].to(self.device)
                    else:
                        x=x.to(self.device)
                    y=y.to(self.device)
                    
                    output=self.model(x)
                    loss=self.loss(output,y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            if oneshot == 1 and clip_rate > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)

            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

    def train_malicious_random_uppdates(self,is_selected,oneshot,clip_rate):
        last_local_model={name:param.clone() for name, param in self.model.named_parameters()}
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()
            
            start_time=time.time()
            max_local_epochs=self.local_epochs
            
            if self.train_slow:
                max_local_epochs=np.random.randint(1,max_local_epochs//2)
                
            for step in range(max_local_epochs):
                for i,(x,y) in enumerate(trainloader):
                    if type(x)==type([]):
                        x[0]=x[0].to(self.device)
                    else:
                        x=x.to(self.device)
                    y=y.to(self.device)
                    
                    output=self.model(x)
                    loss=self.loss(output,y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            if oneshot == 1 and clip_rate > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)
            
            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
                    
    
    def train_malicious_label_poison(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        last_local_model={name:param.clone() for name, param in self.model.named_parameters()}
        if is_selected:
            trainloader = self.load_train_data()
            self.model.train()
            
            start_time=time.time()
            max_local_epochs=self.local_epochs
            
            if self.train_slow:
                max_local_epochs=np.random.randint(1,max_local_epochs//2)
                
            for step in range(max_local_epochs):
                for i,(x,y) in enumerate(trainloader):
                    if type(x)==type([]):
                        x[0]=x[0].to(self.device)
                    else:
                        x=x.to(self.device)
                    poison_ratio_float = poison_ratio / len(x)
                    y=y.to(self.device)
                    y=self.attack_method.poison_batch( 
                        labels=y,
                        num_classes=self.num_classes, 
                        poison_ratio=poison_ratio_float
                        
                    )
                    
                    output=self.model(x)
                    loss=self.loss(output,y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            if oneshot == 1 and clip_rate > 0:
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        original_param = last_local_model.get(name, param.data.clone())
                        scaled_update = (param.data - original_param) * clip_rate
                        param.data.copy_(original_param + scaled_update)

            last_local_model = {name: param.data.clone() for name, param in self.model.named_parameters()}

            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
                    
                    
    
    def train_malicious_neurotoxin(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate, grad_mask):
        """Neurotoxin恶意训练方法（与clientAVG保持一致）"""
        if not is_selected:
            return
        
        # 记录原始参数（用于梯度裁剪）
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()} 
        
        # 加载混合训练数据：包含主任务数据和投毒数据
        trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                          noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
        
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                output = self.model(x)
                loss = self.loss(output, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 应用梯度掩码 - Neurotoxin核心：只对更新幅度较小的参数进行投毒
                with torch.no_grad():
                    param_idx = 0
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and param_idx < len(grad_mask):
                            # 使用梯度掩码来控制梯度更新
                            mask = grad_mask[param_idx]
                            if mask.shape == param.grad.shape:
                                param.grad *= mask
                            param_idx += 1
                
                # 参数更新
                self.optimizer.step()

        # 梯度裁剪：限制参数更新幅度
        if oneshot == 1 and clip_rate > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    original_param = last_local_model.get(name, param.data.clone())
                    scaled_update = (param.data - original_param) * clip_rate
                    param.data.copy_(original_param + scaled_update)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_malicious_dba(self, is_selected, poison_ratio, poison_label, trigger, pattern, oneshot, clip_rate):
        """DBA恶意训练方法（与clientAVG保持一致）"""
        if not is_selected:
            return
        
        # 记录原始参数（用于梯度裁剪）
        last_local_model = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # 加载混合训练数据：包含主任务数据和投毒数据
        trainloader = self.load_poison_data(poison_ratio=poison_ratio, poison_label=poison_label,
                                          noise_trigger=trigger, pattern=pattern, batch_size=self.batch_size)
        
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # 前向传播
                output = self.model(x)
                loss = self.loss(output, y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 参数更新
                self.optimizer.step()

        # 梯度裁剪：限制参数更新幅度
        if oneshot == 1 and clip_rate > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    original_param = last_local_model.get(name, param.data.clone())
                    scaled_update = (param.data - original_param) * clip_rate
                    param.data.copy_(original_param + scaled_update)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def evaluate_attack_success_rate(self, trigger, pattern, poison_label):
        """评估攻击成功率 (ASR)"""
        if hasattr(self.base_attack, 'evaluate_asr'):
            return self.base_attack.evaluate_asr([self], trigger, pattern, poison_label)
        else:
            # 如果没有evaluate_asr方法，使用默认的poisontest方法
            asr_correct, asr_samples = self.poisontest(
                poison_label=poison_label,
                trigger=trigger,
                pattern=pattern
            )
            return asr_correct / asr_samples if asr_samples > 0 else 0.0

    def get_attack_utils(self):
        """获取攻击工具实例"""
        return {
            'base_attack': self.base_attack,
            'attack_utils': self.attack_utils,
            'trigger_utils': self.trigger_utils,
            'attack_method': getattr(self, 'attack_method', None)
        }

    def load_poison_data(self, poison_ratio, poison_label, noise_trigger, pattern, batch_size):
        """加载混合训练数据：包含主任务数据和投毒数据"""
        trainloader = self.load_train_data()
        return self.attack_utils.load_poison_data(
            trainloader=trainloader,
            poison_ratio=poison_ratio,
            poison_label=poison_label,
            noise_trigger=noise_trigger,
            pattern=pattern,
            batch_size=batch_size,
            device=self.device,
            dataset=self.dataset
        )

    def poisontest(self, trigger=None, poison_label=None, pattern=None):
        """测试攻击成功率 (ASR)"""
        testloader = self.load_test_data()
        self.model.eval()
        
        # 检查攻击类型并设置相应的触发器标识
        if hasattr(self, 'attack_method') and self.attack_method == 'model_replacement':
            # 对于模型替换攻击，使用特殊的触发器标识
            trigger = "MODEL_REPLACEMENT_TRIGGER"
            pattern = "MODEL_REPLACEMENT_PATTERN"
        elif hasattr(self, 'attack_method') and self.attack_method == 'badnets':
            # 对于BadNets攻击，使用特殊的触发器标识
            trigger = "BADNETS_TRIGGER"
            pattern = "BADNETS_PATTERN"
        
        # 使用ClientAttackUtils的poisontest方法
        return self.attack_utils.poisontest(
            model=self.model,
            testloader=testloader,
            poison_label=poison_label,
            trigger=trigger,
            pattern=pattern,
            device=self.device,
            dataset=self.dataset
        )
    
    def get_flattened_params(self):
        params = []
        for param in self.model.parameters():
            params.append(param.data.flatten().cpu().numpy())
        return np.concatenate(params)
    
    def get_model_update_diff(self, global_model):
        """计算模型更新差值（用于聚类）"""
        current_params = self.get_flattened_params()
        
        global_params = []
        for param in global_model.parameters():
            global_params.append(param.data.flatten().cpu().numpy())
        global_params_flat = np.concatenate(global_params)
        
        return current_params - global_params_flat
    
    def set_model(self, model):
        self.model = model 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)