import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_attack import BaseAttack, Autoencoder, get_shared_params, get_pattern_info


class HARAutoencoder(nn.Module):
    """HAR数据集专用的1D卷积Autoencoder"""
    def __init__(self, in_channels=9, seq_length=128):
        super(HARAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 输入: (batch_size, 9, 1, 128)
            nn.Conv2d(in_channels, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


class BadPFLAttack(BaseAttack):
    """
    Bad-PFL method：
    - PGD -> adversarial samples
    - autoencoder generates trigger noise
    """

    def __init__(self, args, device):
        super().__init__(args, device)
        self.poison_ratio = getattr(args, 'poison_rate', 0.2)

        # 根据数据集类型选择不同的Autoencoder
        if self.dataset.lower() in ['har', 'uci_har']:
            # HAR数据集使用1D卷积Autoencoder
            self.trigger_gen = HARAutoencoder(in_channels=9, seq_length=128).to(self.device)
        else:
            # 图像数据集使用原有的2D卷积Autoencoder
            in_channels = 3 if self.dataset.lower() in ["cifar10", "cifar100"] else 1
            self.trigger_gen = Autoencoder(in_channels=in_channels).to(self.device)
        
        self.gen_optimizer = torch.optim.Adam(self.trigger_gen.parameters(), lr=1e-2)
        self.ce_loss = nn.CrossEntropyLoss()

    def _pgd_attack(self, model, images, labels, epsilon=4./255., alpha=4./255., num_iter=1):
        model.eval()
        adv_images = images.clone().detach() + torch.zeros_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1)

        for _ in range(num_iter):
            adv_images.requires_grad = True
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            with torch.no_grad():
                adv_images = adv_images + alpha * torch.sign(adv_images.grad)
                eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
                adv_images = torch.clamp(images + eta, min=0, max=1)
            adv_images = adv_images.detach()
        return adv_images

    def optimize_trigger(self, malicious_clients, steps: int = 30):
        """
        generate trigger 
        """
        self.trigger_gen.train()
        for client in malicious_clients:
            trainloader = client.load_train_data()
            local_model = client.model.to(self.device)
            local_model.eval()

            data_iter = iter(trainloader)
            for _ in range(steps):
                try:
                    clean_data, clean_label = next(data_iter)
                except StopIteration:
                    data_iter = iter(trainloader)
                    clean_data, clean_label = next(data_iter)

                clean_data = clean_data.to(self.device)
                clean_label = clean_label.to(self.device)

                self.gen_optimizer.zero_grad()
                adv_imgs = self._pgd_attack(local_model, clean_data, clean_label)
                gen_trigger = self.trigger_gen(clean_data)
                
                # 确保 gen_trigger 与 adv_imgs 尺寸匹配
                if gen_trigger.shape != adv_imgs.shape:
                    gen_trigger = torch.nn.functional.interpolate(
                        gen_trigger, 
                        size=adv_imgs.shape[2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                pred = local_model(adv_imgs + gen_trigger)
                target = torch.full_like(clean_label, fill_value=int(self.poison_label))
                loss = self.ce_loss(pred, target)
                loss.backward()
                self.gen_optimizer.step()

        self.trigger_gen.eval()

    #@torch.no_grad()
    def poison_batch(self, data, labels, client_model, poison_ratio: float = None):
        """
        PGD adv sample + natural noise
        return (poisoned_data, poisoned_labels)
        """
        if poison_ratio is None:
            poison_ratio = self.poison_ratio
        target_label = int(self.poison_label)

        data = data.to(self.device)
        labels = labels.to(self.device)
        client_model = client_model.to(self.device)
        client_model.eval()

        mask = (torch.rand(labels.size(0), device=labels.device) <= poison_ratio)
        if mask.sum().item() == 0:
            return data, labels

        adv_imgs = self._pgd_attack(client_model, data, labels)
        gen_trigger = self.trigger_gen(data)
        
        # 确保 gen_trigger 与 adv_imgs 尺寸匹配
        if gen_trigger.shape != adv_imgs.shape:
            gen_trigger = torch.nn.functional.interpolate(
                gen_trigger, 
                size=adv_imgs.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        mask_4d = mask.view(-1, 1, 1, 1).float()
        poisoned_data = mask_4d * (adv_imgs + gen_trigger) + (1.0 - mask_4d) * data
        poisoned_labels = mask.long() * target_label + (~mask).long() * labels
        return poisoned_data, poisoned_labels

    def poison_batch_for_eval(self, data, labels, client_model):
        """
        专门用于评估的投毒方法，不进行梯度计算
        避免在torch.no_grad()上下文中调用需要梯度的PGD攻击
        """
        target_label = int(self.poison_label)

        data = data.to(self.device)
        labels = labels.to(self.device)
        client_model = client_model.to(self.device)
        client_model.eval()

        # 确保trigger_gen处于评估模式
        self.trigger_gen.eval()
        
        # 使用torch.no_grad()避免梯度计算
        with torch.no_grad():
            # 对所有样本进行投毒（评估模式）
            mask = torch.ones(labels.size(0), device=labels.device, dtype=torch.bool)
            
            # 生成对抗样本（不使用PGD，直接使用原始数据）
            # 在评估模式下，我们不需要生成真正的对抗样本，使用原始数据即可
            adv_imgs = data.clone()
            
            # 生成触发器
            gen_trigger = self.trigger_gen(data)
            
            # 确保 gen_trigger 与 adv_imgs 尺寸匹配
            if gen_trigger.shape != adv_imgs.shape:
                gen_trigger = torch.nn.functional.interpolate(
                    gen_trigger, 
                    size=adv_imgs.shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )

            mask_4d = mask.view(-1, 1, 1, 1).float()
            poisoned_data = mask_4d * (adv_imgs + gen_trigger) + (1.0 - mask_4d) * data
            poisoned_labels = mask.long() * target_label + (~mask).long() * labels
            
        return poisoned_data, poisoned_labels

    def execute_attack_training(self, server, round_num, attack_start, external_pattern):
        """
        Bad-PFL攻击训练执行函数
        
        Args:
            server: 服务器实例
            round_num: 当前轮次
            attack_start: 攻击开始轮次
            external_pattern: 外部模式
            
        Returns:
            badpfl_trigger: Bad-PFL触发器（特殊标识符）
            badpfl_pattern: Bad-PFL模式（特殊标识符）
        """
        badpfl_trigger = None
        badpfl_pattern = None
        
        # 初始化触发器
        if round_num == attack_start:
            print("Bad-PFL: 初始化触发器...")
            if self.dataset.lower() in ['har', 'uci_har']:
                # HAR数据集：9x1x128
                badpfl_trigger = torch.randn(9, 1, 128).to(server.device)
            else:
                # 图像数据集
                badpfl_trigger = torch.randn(3, 32, 32).to(server.device) if server.dataset.lower() in ["cifar10", "cifar100"] else torch.randn(1, 28, 28).to(server.device)
            badpfl_pattern = [(0, 0)]
        
        # 优化触发器生成器
        if round_num >= attack_start:
            print("Bad-PFL: 优化触发器生成器...")
            if hasattr(server, 'badpfl_attack'):
                server.badpfl_attack.optimize_trigger(
                    malicious_clients=server.malicious_clients,
                    steps=30
                )
                print("Bad-PFL: 触发器生成器优化完成")
        
        # 初始化数据收集 - 每一轮都收集数据
        client_data = []
        
        # 恶意客户端训练
        for client in server.alled_clients:
            if client in server.malicious_clients:
                if round_num >= attack_start:
                    if client.id in [c.id for c in server.selected_clients]:
                        print("Bad-PFL: 恶意客户端参与训练...")
                    
                    # 使用现有的 train_malicious 方法，但传入 Bad-PFL 触发器
                    client.train_malicious(
                        is_selected=client.id in [c.id for c in server.selected_clients],
                        poison_ratio=server.poisonratio,
                        poison_label=server.poisonlabel,
                        trigger=badpfl_trigger,
                        pattern=badpfl_pattern,
                        oneshot=0,
                        clip_rate=0.2
                    )
                    
                    # 评估攻击成功率
                    accuracy = client.evaluate()
                    # Bad-PFL攻击使用特殊的poisontest方式，传递None表示使用Bad-PFL方法
                    poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=None,  # Bad-PFL不使用传统触发器
                        poison_label=server.poisonlabel,
                        pattern=None   # Bad-PFL不使用传统模式
                    )
                    poiosnaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                    
                    print("***************Bad-PFL攻击成功率、主任务准确度 & 恶意客户端******************")
                    print(f'Client {client.id} ASR :{poiosnaccuracy}')
                    print(f'Client {client.id} ACC :{accuracy}')

                else:
                    # 攻击开始前的正常训练
                    client.train(client.id in [c.id for c in server.selected_clients])
                    accuracy = client.evaluate()
                    # Bad-PFL攻击使用特殊的poisontest方式
                    poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=None,
                        poison_label=server.poisonlabel,
                        pattern=None
                    )
                    poiosnaccuracy = poiosnacc_count / poiosnsumcount if poiosnsumcount > 0 else 0.0
                    
                    print("***************Bad-PFL攻击成功率、主任务准确度 & 恶意客户端（攻击前）******************")
                    print(f'Client {client.id} ASR :{poiosnaccuracy}')
                    print(f'Client {client.id} ACC :{accuracy} (pre-attack)')

                # 收集恶意客户端数据（无论是否在攻击轮次）
                client_data.append({
                    'client_id': client.id,
                    'client_acc': accuracy,
                    'client_asr': poiosnaccuracy,
                    'is_malicious': True,
                    'client_time': getattr(client, 'train_time_cost', {}).get('total_cost', 0.0)
                })

            else:  # 正常客户端
                client.train(client.id in [c.id for c in server.selected_clients])
                accuracy = client.evaluate()
                # 正常客户端也使用Bad-PFL的poisontest方式
                poiosnacc_count, poiosnsumcount = client.poisontest(
                        trigger=None,  # Bad-PFL不使用传统触发器
                        poison_label=server.poisonlabel,
                        pattern=None   # Bad-PFL不使用传统模式
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
                
                print("====================Bad-PFL攻击成功率、主任务准确度 & 正常客户端==============================")
                print(f'Client {client.id} ASR :{poiosnaccuracy}')
                print(f'Client {client.id} ACC :{accuracy}')
        
        # 每一轮都记录数据 - 使用server的base_attack.data_recorder
        if client_data:  # 确保有数据才记录
            server.base_attack.data_recorder.record_attack_data_from_collected(
                round_num=round_num,
                server=server,
                client_data=client_data,
                attack_method='badpfl',
                original_trigger_list=[badpfl_trigger] if badpfl_trigger is not None else [],
                external_pattern=badpfl_pattern
            )
        
        # 返回特殊的Bad-PFL标识符，用于全局ASR评估
        return "BADPFL_TRIGGER", "BADPFL_PATTERN"