import os
import argparse
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms.functional as TF
import math
from open_clip import  get_tokenizer, create_model_and_transforms, create_model_from_pretrained
import pandas as pd
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from pca_m import AdaptivePCASpace
import logging

def seed_everything(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class OpenCLIPWrapper(torch.nn.Module):
    """
    Wrapper to unify OpenCLIP model interface with transformers.CLIPModel
    """
    def __init__(self, model, preprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess
        self.model.train()

    def parameters(self, recurse=True):
        # no trainable parameters
        return []

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def get_image_features(self, pixel_values=None, images=None):
        # Accept either preprocessed tensor or raw PILs
        # We expect: either pixel_values Tensor [B,3,H,W], or images: List[PIL]
        if pixel_values is not None:
            x = pixel_values
        else:
            # process raw PILs
            x = torch.stack([self.preprocess(img) for img in images], dim=0)
        x = x.to(next(self.model.parameters(), torch.tensor(0)).device)
        
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats

    def __call__(self, *args, **kwargs):
        return self.get_image_features(*args, **kwargs)

class DropPathBlock_trans(nn.Module):
    """
    把一个 CLIPEncoderLayer 包裹起来，
    丢弃它时返回一个和它同样“解包”形式的输出。
    """
    def __init__(self, block: nn.Module, drop_prob: float):
        super().__init__()
        self.block = block
        self.drop_prob = drop_prob

    def forward(self, hidden_states, *args, **kwargs):
        # 先拿到一次真实的输出，来探测它到底是 tuple 还是 dataclass
        real_out = self.block(hidden_states, *args, **kwargs)
        if not self.training or torch.rand(()) >= self.drop_prob:
            # 正常执行
            return real_out

        # 下面是“丢弃”分支：把 hidden_states 原样放回去，
        # 并且填充同样长度的 None，这样后面 layer_outputs[0] 拿到 hidden_states，layer_outputs[1] 拿到 None...
        if isinstance(real_out, tuple):
            L = len(real_out)
            return (hidden_states,) + (None,) * (L - 1)
        else:
            # transformers 的 BaseModelOutput 系列都是 dataclass，有 .last_hidden_state/.hidden_states/.attentions
            # 我们用一个简单的 hack：构造一个和它同类型的新对象，所有字段都设 hidden_states
            fields = {k: hidden_states for k in real_out.to_dict().keys()}
            return real_out.__class__(**fields)

class DropPathBlock(nn.Module):
    def __init__(self, block: nn.Module, drop_prob: float):
        super().__init__()
        self.block = block
        self.drop_prob = drop_prob

    def forward(self, x, *args, **kwargs):
        # 随机丢弃整个 block
        if torch.rand(()) < self.drop_prob:
            return x
        else:
            return self.block(x, *args, **kwargs)

class RegularizedTransformersCLIPWrapper(nn.Module):
    """
    将 HuggingFace CLIPModel 的视觉主干注入 DropPath + PatchDrop
    并对外暴露 get_image_features 与 HF CLIPModel 接口一致。
    """
    def __init__(
        self,
        hf_clip: CLIPModel,
        hf_proc: CLIPProcessor,
        drop_path_rate: float = 0.2,
        patch_drop_rate: float = 0.2,
    ):
        super().__init__()
        self.clip = hf_clip.train()
        self.proc = hf_proc
        self.drop_path_rate = drop_path_rate
        self.patch_drop_rate = patch_drop_rate

        # --- 在视觉 Transformer 上插入 DropPathBlock ---
        # HF CLIPModel 的视觉部分保存在 `.vision_model.encoder.layers`
        vision = self.clip.vision_model
        encoder_layers = vision.encoder.layers  # nn.ModuleList
        self.L = len(encoder_layers)
        for idx, layer in enumerate(encoder_layers):
            # i 从 1..L，对应论文中 i-th block 丢弃概率 = (i/L)*p_max
            p_i = (idx + 1) / self.L * self.drop_path_rate
            encoder_layers[idx] = DropPathBlock_trans(layer, p_i)

    def get_image_features(self, pixel_values=None, images=None):
        # 1) 预处理：支持已预处理 tensor 或 PIL 列表
        if pixel_values is None:
            pixel_values = self.proc(images=images, return_tensors="pt").pixel_values
        x = pixel_values.to(next(self.clip.parameters()).device)

        B, C, H, W = x.shape
        # --- PatchDrop ---
        # 取 patch 大小（HF CLIP 默认 patch_size 存在于 vision.embeddings.patch_size）
        Ps = self.clip.config.vision_config
        P = Ps.patch_size if isinstance(Ps.patch_size, int) else Ps.patch_size[0]
        nH, nW = H // P, W // P
        # 需要丢弃的 patch 数
        num_drop = int(self.patch_drop_rate * nH * nW + 0.5)
        for b in range(B):
            idxs = random.sample(range(nH * nW), num_drop)
            for idx in idxs:
                i, j = divmod(idx, nW)
                x[b, :, i*P:(i+1)*P, j*P:(j+1)*P] = 0.0

        # 2) 前向：调用 transformers.CLIPModel 自带的 get_image_features
        feats = self.clip.get_image_features(pixel_values=x)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats

    def __call__(self, *args, **kwargs):
        return self.get_image_features(*args, **kwargs)

class RegularizedOpenCLIPWrapper(nn.Module):
    def __init__(self,
                 model,
                 preprocess,
                 drop_path_rate: float = 0.2,
                 patch_drop_rate: float = 0.2):
        super().__init__()
        self.model = model.train()
        self.preprocess = preprocess
        self.drop_path_rate = drop_path_rate
        self.patch_drop_rate = patch_drop_rate
        # 1) open_clip 官方模型
        if hasattr(self.model.visual, "transformer"):
            vt = self.model.visual.transformer
            blocks = vt.resblocks
            patch_size = self.model.visual.patch_size
            patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]

        # 2) TimmModel 封装的 ViT（CustomTextCLIP）
        elif hasattr(self.model.visual, "trunk") and hasattr(self.model.visual.trunk, "blocks"):
            vt = self.model.visual.trunk
            blocks = vt.blocks
            patch_size = vt.patch_embed.proj.kernel_size[0]  # 通常是 (P,P)
            patch_size = patch_size if isinstance(patch_size, int) else patch_size[0]
        else:
            raise RuntimeError("Cannot find vision transformer blocks in model.visual")
        
        # 拿到所有的 blocks
        self.L = len(blocks)

        # 用 DropPathBlock 包一遍
        for idx, blk in enumerate(blocks):
            p_i = (idx + 1) / self.L * self.drop_path_rate
            blocks[idx] = DropPathBlock(blk, p_i)

        # 存下来，后面 patch_drop 用
        self._blocks = blocks
        self._patch_size = patch_size


    def get_image_features(self, pixel_values=None, images=None):
        # —— preprocess …… 同你原来写的 —— 
        if pixel_values is not None:
            x = pixel_values
        else:
            x = torch.stack([self.preprocess(img) for img in images], dim=0)
        x = x.to(next(self.model.parameters()).device)

        B, C, H, W = x.shape

        # —— PatchDrop —— 
        P = self._patch_size if isinstance(self._patch_size, int) else self._patch_size[0]
        nH, nW = H // P, W // P
        num_drop = int(self.patch_drop_rate * nH * nW + 0.5)
        for b in range(B):
            drop_idxs = random.sample(range(nH * nW), num_drop)
            for di in drop_idxs:
                i, j = divmod(di, nW)
                x[b, :, i*P:(i+1)*P, j*P:(j+1)*P] = 0.0

        # —— 正常前向 和 L2 归一化 —— 
        
        feats = self.model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats

    def __call__(self, *args, **kwargs):
        return self.get_image_features(*args, **kwargs)
# ——— 纯 PyTorch、可微分的 JPEG（分开 Y/C 量化表） ———
class DiffJPEG(nn.Module):
    def __init__(self, quality: float = 75.0):
        super().__init__()
        # JPEG 标准亮度／色度量化表（Q50, QC50），归一到 [0,1]
        QY50 = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]
        ], dtype=torch.float32) / 255.0
        QC50 = torch.tensor([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99]
        ], dtype=torch.float32) / 255.0

        # 根据 quality 缩放
        if quality < 50:
            scale = 5000.0 / quality
        else:
            scale = 200.0 - 2.0 * quality
        QY = torch.clamp((QY50 * scale + 50.0) / 100.0, min=1e-2)
        QC = torch.clamp((QC50 * scale + 50.0) / 100.0, min=1e-2)

        # 构造 8×8 DCT
        N = 8
        coords = torch.arange(N, dtype=torch.float32)
        i, j = torch.meshgrid(coords, coords, indexing='ij')
        D = torch.where(
            i == 0,
            1.0 / math.sqrt(N),
            math.sqrt(2.0 / N) * torch.cos(((2*j + 1) * i * math.pi) / (2 * N))
        )

        # 注册成 buffer，随 .to(device) 一起移动
        self.register_buffer('QY', QY)      # [8,8]
        self.register_buffer('QC', QC)      # [8,8]
        self.register_buffer('D', D)        # [8,8]
        self.register_buffer('DT', D.t())   # [8,8]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W],H,W 必须是 8 的倍数(外部保证 pad)
        B, C, H, W = x.shape
        nH, nW = H // 8, W // 8

        # 拆成块 [B,3,nH,nW,8,8]
        x_blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)
        # 平成 [M,8,8]
        M = B * C * nH * nW
        flat = x_blocks.contiguous().view(M, 8, 8)

        # DCT
        D, DT = self.D, self.DT
        normed = flat - 0.5
        dct = D @ normed @ DT   # [M,8,8]

        # reshape 回 [B,3,nH,nW,8,8]
        dct = dct.view(B, C, nH, nW, 8, 8)

        # 按通道量化:Y→QY,Cb/Cr→QC
        QY, QC = self.QY, self.QC
        q = torch.empty_like(dct)
        q[:,0] = torch.round(dct[:,0] / QY) * QY
        q[:,1] = torch.round(dct[:,1] / QC) * QC
        q[:,2] = torch.round(dct[:,2] / QC) * QC

        # IDCT
        q_flat = q.view(M, 8, 8)
        rec = DT @ q_flat @ D
        rec = rec + 0.5

        # 重组回 [B,3,H,W]
        rec = rec.view(B, C, nH, nW, 8, 8)
        rec = rec.permute(0,1,2,4,3,5).contiguous().view(B, C, H, W)
        return rec

class PaperAugmentation_new:
    def __init__(
        self,
        input_size: tuple[int, int],
        epsilon: float = 8/255,
        jpeg_quality: float = 75.0,
        p: float = 0.2,
        device: torch.device = torch.device('cuda')
    ):
        self.input_size = input_size  # 模型目标输入尺寸，如 (384, 384)
        self.epsilon = epsilon
        self.p = p
        self.device = device
        self.jpeg = DiffJPEG(quality=jpeg_quality).to(device)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,3,H,W], 值域 [0, 1]
        _, C, H, W = x.shape
        D_H, D_W = self.input_size

        # Step 1: Add Gaussian noise
        if random.random() < 0.5:
            noise = torch.randn_like(x) * (self.epsilon / 4)
            x = torch.clamp(x + noise, 0, 1)

        # Step 2: Random crop
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.0)
            crop_h, crop_w = int(H * scale), int(W * scale)
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            x = x[:, :, top:top+crop_h, left:left+crop_w]
            H, W = crop_h, crop_w  # 更新尺寸

        # Step 3: Random pad to (D, D)
        if random.random() < 0.5:
            pad_h = max(D_H - H, 0)
            pad_w = max(D_W - W, 0)
            pad_top = random.randint(0, pad_h)
            pad_left = random.randint(0, pad_w)
            pad_bottom = pad_h - pad_top
            pad_right = pad_w - pad_left
            pad = (pad_left, pad_right, pad_top, pad_bottom)  # (left, right, top, bottom)
            x = TF.pad(x, pad, fill=0)
            H += pad_h
            W += pad_w

        # Step 4: Pad to 8x multiple for JPEG
        H2, W2 = x.shape[2], x.shape[3]
        extra_h = (8 - H2 % 8) % 8
        extra_w = (8 - W2 % 8) % 8
        pad = (extra_w // 2, extra_h // 2, extra_w - extra_w // 2, extra_h - extra_h // 2)
        x = TF.pad(x, pad, fill=0)

        # Step 5: Differentiable JPEG
        if random.random() < self.p:
            x = self.jpeg(x)
        
        # Step 6: Remove the JPEG padding
        l, t, r, b = pad
        x = x[..., t:t+H2, l:l+W2]

        return x  # [1,3,H',W']，仍在 [0,1] 空间

class PairedContrastiveDataset_299(Dataset):
    def __init__(
        self,
        attack_dir,
        csv_file,
        base_dir,
        num_pos_samples=50,
        num_neg_samples=50,
        image_extension='.png',
        transform=None
    ):
        """
        :param attack_dir: 目录，包含待攻击的图像（.png)
        :param csv_file: CSV 路径，包含 ImageId, TrueLabel, TargetClass 列
        :param base_dir: 包含 000-999 子目录的根目录，每个子目录下为该类别图像
        :param num_pos_samples: 每个样本采样的正例数量
        :param num_neg_samples: 每个样本采样的负例数量
        :param image_extension: 攻击图像后缀，默认 `.png`
        :param transform: 可选的 torchvision.transforms
        """
        self.attack_dir = attack_dir
        self.base_dir = base_dir
        self.num_pos = num_pos_samples
        self.num_neg = num_neg_samples
        self.image_extension = image_extension
        self.transform = transform

        # 1. 从 CSV 加载映射
        df = pd.read_csv(csv_file, dtype={'ImageId': str})
        required_cols = ['ImageId', 'TrueLabel', 'TargetClass']
        assert all(c in df.columns for c in required_cols), \
            f"CSV 文件必须包含列: {required_cols}"

        # 2. 构建样本列表
        self.samples = []
        for _, row in df.iterrows():
            image_id = row['ImageId']
            src_label = int(row['TrueLabel'])
            tgt_label = int(row['TargetClass'])
            img_path = os.path.join(self.attack_dir, image_id + self.image_extension)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"未找到攻击图像: {img_path}")
            self.samples.append({
                'image_path': img_path,
                'src_label': src_label,
                'tgt_label': tgt_label
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]

        # 加载攻击图像
        src_img = Image.open(entry['image_path']).convert('RGB')
        if self.transform:
            src_img = self.transform(src_img)

        # 从目标类别目录中随机采样正例
        tgt_dir = os.path.join(self.base_dir, f"{entry['tgt_label']:03d}")
        pos_pool = [os.path.join(tgt_dir, f) for f in os.listdir(tgt_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        if len(pos_pool) < self.num_pos:
            raise ValueError(f"目标类别 {entry['tgt_label']} 图像不足 {self.num_pos} 张")
        pos_paths = random.sample(pos_pool, self.num_pos)
        pos_images = []
        for p in pos_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            pos_images.append(img)

        # 从自身类别目录中随机采样负例
        neg_dir = os.path.join(self.base_dir, f"{entry['src_label']:03d}")
        neg_pool = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        if len(neg_pool) < self.num_neg:
            raise ValueError(f"自身类别 {entry['src_label']} 图像不足 {self.num_neg} 张")
        neg_paths = random.sample(neg_pool, self.num_neg)
        neg_images = []
        for n in neg_paths:
            img = Image.open(n).convert('RGB')
            if self.transform:
                img = self.transform(img)
            neg_images.append(img)

        return {
            'src_image': src_img,
            'pos_images': pos_images,
            'neg_images': neg_images,
            'src_class': entry['src_label'],
            'tgt_class': entry['tgt_label'],
            'image_id': os.path.basename(entry['image_path'])
        }



    def forward_feats(self, x_adv_norm: torch.Tensor,
                      pos_feats: torch.Tensor,
                      neg_feats: torch.Tensor) -> torch.Tensor:
        """
        :param x_adv_norm: [B,3,H,W]  归一化后的对抗图像
        :param pos_feats:   [B,N_pos,D]  已缓存并 L2 归一化的正例特征
        :param neg_feats:   [B,N_neg,D]  已缓存并 L2 归一化的负例特征
        """
        B, N_pos, D = pos_feats.shape
        N_neg = neg_feats.shape[1]

        # 1) 编码对抗样本、并 L2 归一化
        adv_feats = self.clip_model.get_image_features(pixel_values=x_adv_norm)  # [B,D]
        adv_feats = F.normalize(adv_feats, dim=-1)

        # 2) 计算余弦相似度（点积即可）
        sim_pos = torch.einsum('bd,bpd->bp', adv_feats, pos_feats) / self.temp
        sim_neg = torch.einsum('bd,bnd->bn', adv_feats, neg_feats) / self.temp

        # 3) 合并、log-softmax
        sim = torch.cat([sim_pos, sim_neg], dim=1)                        # [B, N_pos+N_neg]
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        # 4) 正例 Top-K、负例均值
        topk_vals, _ = torch.topk(logp[:, :N_pos], k=self.top_k, dim=1)
        loss_pos = - topk_vals.mean()    # ≥ 0
        loss_neg =   logp[:, N_pos:].mean()  # ≤ 0

        total_loss = loss_pos + loss_neg
        print(f'Loss Pos:{loss_pos.item():.4f}, Loss Neg:{loss_neg.item():.4f}, Total Loss:{total_loss.item():.4f}')
        logging.info(f'Loss Pos:{loss_pos.item():.4f}, Loss Neg:{loss_neg.item():.4f}, Total Loss:{total_loss.item():.4f}')
        return total_loss

class VisualContrastiveLoss_enhance_PCA(nn.Module):
    def __init__(self, clip_model, pca_space, temp=0.1, top_k=10, top_m=10, margin=0.2, lambda_margin=2.0):
        super().__init__()
        self.clip_model = clip_model.train()
        for param in self.clip_model.parameters():
            param.requires_grad_(False)

        self.pca_space = pca_space  # PCA空间
        self.temp = temp
        self.top_k = top_k
        self.top_m = top_m
        self.margin = margin
        self.lambda_margin = lambda_margin

    def forward_feats(self, x_adv_norm, pos_feats, neg_feats):
        adv_feats = self.clip_model.get_image_features(pixel_values=x_adv_norm)
        adv_feats = F.normalize(adv_feats, dim=-1)

        # PCA投影
        adv_feats_proj = self.pca_space.project(adv_feats)
        pos_feats_proj = self.pca_space.project(pos_feats)
        neg_feats_proj = self.pca_space.project(neg_feats)

        sim_pos = torch.einsum('bd,bpd->bp', adv_feats_proj, pos_feats_proj) / self.temp
        sim_neg = torch.einsum('bd,bnd->bn', adv_feats_proj, neg_feats_proj) / self.temp

        sim_all = torch.cat([sim_pos, sim_neg], dim=1)
        log_probs = sim_all - torch.logsumexp(sim_all, dim=1, keepdim=True)

        pos_log = log_probs[:, :pos_feats.shape[1]]
        topk_vals, _ = torch.topk(pos_log, self.top_k, dim=1)
        topk_weights = F.softmax(topk_vals.detach(), dim=1)
        loss_pos_topk = -(topk_weights * topk_vals).sum(dim=1)
        loss_pos_max = -pos_log.max(dim=1).values
        loss_pos = (0.8 * loss_pos_topk + 0.2 * loss_pos_max).mean()

        neg_log = log_probs[:, pos_feats.shape[1]:]
        topm_vals, _ = torch.topk(neg_log, min(self.top_m, neg_log.shape[1]), dim=1)
        loss_neg = topm_vals.mean()

        max_pos = sim_pos.max(dim=1).values
        max_neg = sim_neg.max(dim=1).values
        margin_loss = F.relu(max_neg - max_pos + self.margin).mean()

        total_loss = loss_pos + loss_neg + self.lambda_margin * margin_loss

        return total_loss


def attack_pipeline():
    model_dict_transformers = {
    'metaclip-h14-fullcc2.5b':'/var/lib/docker/huggingface_model/metaclip-h14-fullcc2.5b',  
    'DFN5B-CLIP-ViT-H-14-378':'apple/DFN5B-CLIP-ViT-H-14-378',   
    'CLIP-ViT-bigG-14':'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k' 
    }
    model_dict_open_clip = {
    'ViT-H-14-CLIPA-336':{'weights_path':'UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B/open_clip_model.safetensors',},
    'ViT-SO400M-14-SigLIP':{'weights_path':'timm/ViT-SO400M-14-SigLIP/open_clip_model.safetensors',},
    'ViT-SO400M-14-SigLIP-384':{'weights_path':'timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors',},
    'ViT-H-14':{'weights_path':'apple/DFN5B-CLIP-ViT-H-14/open_clip_pytorch_model.bin',},
    'convnext_xxlarge':{'weights_path':'laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg/open_clip_pytorch_model.bin',}
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/seu_nvme/home/230239304/dataset/ImageNet_val_1000', help='ImageNet数据集路径')
    parser.add_argument('--clip_models', nargs='+', default=['CLIP-ViT-bigG-14','DFN5B-CLIP-ViT-H-14-378','metaclip-h14-fullcc2.5b'], help='CLIP模型名称')
    #parser.add_argument('--clip_models', nargs='+', default=['CLIP-ViT-bigG-14','DFN5B-CLIP-ViT-H-14-378'], help='CLIP模型名称')
    parser.add_argument('--open_clip_models', nargs='+', default=['ViT-H-14-CLIPA-336','ViT-SO400M-14-SigLIP','ViT-H-14','ViT-SO400M-14-SigLIP-384','convnext_xxlarge'], help='CLIP模型名称')
    #parser.add_argument('--open_clip_models', nargs='+', default=['ViT-SO400M-14-SigLIP'], help='CLIP模型名称')
    parser.add_argument('--eps', type=float, default=16/255, help='扰动上限')
    parser.add_argument('--alpha', type=float, default=0.5/255, help='PGD步长')
    parser.add_argument('--steps', type=int, default=300, help='迭代次数')

    parser.add_argument('--output_dir', default='your adv output path', help='对抗样本输出目录')
    parser.add_argument('--output_delta_dir', default='your adv delta output path', help='delta可视化输出目录')
    parser.add_argument('--output_delta_pt_dir', default='your adv delta_pt output path', help='delta_pt输出目录')
    parser.add_argument('--output_clean_dir', default='your clean output path', help='干净样本输出目录')
    parser.add_argument('--attack_dir',default='NIPS2017/images', help='待攻击的图像目录')
    parser.add_argument('--csv_file',default='./images_299.csv', help='包含 ImageId,TrueLabel,TargetClass 的 CSV 文件')
    parser.add_argument('--base_dir',default='your imagenet-1000-val path', help='ImageNet 000-999 子目录根目录')

    parser.add_argument('--log_file_path',default='your log file path', help='Log文件路径')

    parser.add_argument('--MI_PGD', action='store_false',help='默认为True,传参为False')
    parser.add_argument('--mu', type=float, default=0.8, help='MI-PGD动量') # 动量因子，可调 0.8–0.99

    parser.add_argument('--EMA', action='store_false',help='默认为True,传参为False')
    parser.add_argument('--ema_mu', type=float, default=0.99, help='EMA动量')

    parser.add_argument('--batch_size',default=1, type=int)
    parser.add_argument('--num_workers',default=4, type=int)

    parser.add_argument('--gpu_id',default=4, type=int)
    parser.add_argument('--attack_num',default=1000, type=int)

    parser.add_argument('--aug', action='store_false',help='默认为True,传参为False')
    parser.add_argument('--drop_patch_rate', type=float, default=0.0, help='drop patch率')
    parser.add_argument('--drop_block_rate', type=float, default=0.0, help='drop block率')

    parser.add_argument('--pca_dim',default=10, type=int, help='PCA子空间维度')


    args = parser.parse_args()
    # 环境准备
    seed_everything()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_delta_dir, exist_ok=True)
    os.makedirs(args.output_delta_pt_dir, exist_ok=True)
    #os.makedirs(args.output_clean_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file_path), exist_ok=True)
    
    logging.basicConfig(
        filename=args.log_file_path,
        filemode='a',  # 追加模式
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
        )
    
    print(args)
    logging.info(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 加载模型、处理器和损失
    clip_models, processors, criterions, input_sizes= [], [], [], []
    for model_name in args.clip_models:
        model_path = model_dict_transformers[model_name]
        model = CLIPModel.from_pretrained(model_path).to(device) 
        processor = CLIPProcessor.from_pretrained(model_path)
        wrapper = RegularizedTransformersCLIPWrapper(
            model,
            processor,
            drop_path_rate=args.drop_block_rate, 
            patch_drop_rate=args.drop_patch_rate
        ).to(device)
        clip_models.append(wrapper)
        processors.append(('transformers', processor))
        cfg = processor.image_processor.size
        if isinstance(cfg, int): 
            h=w=cfg
        else:
            h = cfg.get('height', cfg.get('shortest_edge'))
            w = cfg.get('width',  cfg.get('shortest_edge', h))
        input_sizes.append(max(h,w))
        print(f"{model_name} 已加载，输入尺寸 {h}")
        logging.info(f"{model_name} 已加载，输入尺寸 {h}")
    
    # 加载 open_clip 接口的 CLIP
    for name in args.open_clip_models:
        cfg = model_dict_open_clip[name]
        model, _, preprocess = create_model_and_transforms(
            name, pretrained=cfg['weights_path'], precision='fp32', device=device)
        if name in ['convnext_xxlarge']:
            wrapper = OpenCLIPWrapper(model, preprocess).to(device)
        else:
            wrapper = RegularizedOpenCLIPWrapper(
                model, preprocess,
                drop_path_rate=args.drop_block_rate, 
                patch_drop_rate=args.drop_patch_rate
            ).to(device)
        clip_models.append(wrapper)
        processors.append(('openclip', preprocess))
        input_sizes.append(preprocess.transforms[0].size)
        print(f"{name} 已加载，输入尺寸 {preprocess.transforms[0].size}")
        logging.info(f"{name} 已加载，输入尺寸 {preprocess.transforms[0].size}")

    dataset = PairedContrastiveDataset_299(
        attack_dir=args.attack_dir,
        csv_file=args.csv_file,
        base_dir=args.base_dir,
        num_pos_samples=50,
        num_neg_samples=50,
        transform=None
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda b: b[0], num_workers=args.num_workers)
    i = 0
    # 初始化 PCA 子空间
    pca_spaces = [AdaptivePCASpace(subspace_dim=args.pca_dim) for _ in clip_models]
    criterions = [VisualContrastiveLoss_enhance_PCA(model, pca_spaces[i]) for i, model in enumerate(clip_models)]
    model_group = list(zip(processors, clip_models, criterions, input_sizes))

    for data in tqdm(dataloader, desc="Attacking", total=len(dataloader)):
        i += 1
        if i > args.attack_num:
            print(f"已完成 {args.attack_num} 个样本的攻击，提前退出")
            logging.info(f"已完成 {args.attack_num} 个样本的攻击，提前退出")
            break
        orig_img = data['src_image']
        image_id = data['image_id'].rsplit('.', 1)[0]
        pos_images = data['pos_images']  # len = N_pos
        neg_images = data['neg_images']  # len = N_neg
        src_lbl = data['src_class']
        tgt_lbl = data['tgt_class']
        orig_W, orig_H = orig_img.size

        clean_name = f"{image_id}_{src_lbl}_to_{tgt_lbl}.png"
        orig_img.save(os.path.join(args.output_clean_dir, clean_name))
        if os.path.exists(os.path.join(args.output_dir, clean_name)):
            print(f"样本 {i} 已存在，跳过")
            logging.info(f"样本 {i} 已存在，跳过")
            continue
        # 2 为每个模型预先编码正负例
        all_pos_feats, all_neg_feats = [], []
        for (kind, proc), model in zip(processors, clip_models):
            if kind == 'transformers':
                pos_t = torch.stack([
                    proc(images=p, return_tensors='pt').pixel_values[0]
                    for p in pos_images
                ], dim=0).to(device)
                neg_t = torch.stack([
                    proc(images=n, return_tensors='pt').pixel_values[0]
                    for n in neg_images
                ], dim=0).to(device)
            else:  # openclip
                pos_t = torch.stack([proc(p) for p in pos_images], dim=0).to(device)
                neg_t = torch.stack([proc(n) for n in neg_images], dim=0).to(device)

            # 编码一次，缓存
            with torch.no_grad():
                pos_f = model.get_image_features(pixel_values=pos_t)  # [N_pos, D]
                neg_f = model.get_image_features(pixel_values=neg_t)  # [N_neg, D]
                # L2 归一化
                pos_f = F.normalize(pos_f, dim=-1)
                neg_f = F.normalize(neg_f, dim=-1)

            # 变形到 [1, N, D]
            all_pos_feats.append(pos_f.unsqueeze(0))
            all_neg_feats.append(neg_f.unsqueeze(0))
        print('正负例样本特征编码完毕')
        logging.info('正负例样本特征编码完毕')

        # 初始化共享 delta（在输入尺寸上优化）
        delta_base = torch.empty((1, 3, 299, 299), device=device)
        delta_base.uniform_(-1.0, 1.0)
        delta_base = delta_base * args.eps
        delta_base.requires_grad_(True)
        if args.MI_PGD:
            print(f'选择使用MI-PGD, momentum = {args.mu}')
            logging.info(f'选择使用MI-PGD, momentum = {args.mu}')
            # MI-PGD 动量项初始化
            momentum = torch.zeros_like(delta_base)
            mu = args.mu  
        
        x = TF.to_tensor(orig_img).unsqueeze(0).to(device)
        for step in range(args.steps):
            x_adv = x + delta_base  # [1,3,H,W]
            x_adv = torch.clamp(x_adv, 0, 1)
            
            if step % 10 == 0:
                with torch.no_grad():
                    for idx, (_, model, _, size) in enumerate(model_group):
                        x_adv_resized = F.interpolate(x_adv, size=(size, size), mode='bicubic', align_corners=False)

                        feat_adv = model.get_image_features(pixel_values=x_adv_resized).detach()  # [1, D]
                        pos_feat = all_pos_feats[idx].squeeze(0)  # [1, D]
                        neg_feat = all_neg_feats[idx].squeeze(0)  # [1, D]

                        pca_spaces[idx].update_subspace(torch.cat([feat_adv, pos_feat, neg_feat], dim=0))

                print(f"PCA子空间已更新,step={step}")

            total_loss = 0.0
            for idx, ((kind, proc), model, criterions, size) in enumerate(model_group):
                # 当前模型输入尺寸
                H_resize = W_resize = size
                if args.aug:
                    augmentor = PaperAugmentation_new((H_resize, W_resize),device=device)

                transform_post_aug = transforms.Compose(
                    [transforms.Resize(H_resize, interpolation=InterpolationMode.BICUBIC, antialias=True),
                    transforms.CenterCrop((H_resize, W_resize)),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711])])

                if args.aug:
                    x_adv_aug = augmentor.apply(x_adv)
                    x_adv_aug = transform_post_aug(x_adv_aug.squeeze(0)).unsqueeze(0)
                else:
                    x_adv_aug = transform_post_aug(x_adv.squeeze(0)).unsqueeze(0)
                
                pos_f = all_pos_feats[idx]    # 已缓存
                neg_f = all_neg_feats[idx]    # 已缓存

                loss_i = criterions.forward_feats(x_adv_aug, pos_f, neg_f)
                   
                #loss_i = loss_i / (loss_i.abs().detach().clamp(min=1.0)) 
                total_loss += loss_i

            # 更新共享 delta
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_([delta_base], max_norm=5.0)
                
            with torch.no_grad():
                # 获取梯度并归一化
                grad = delta_base.grad  # shape: [1, 3, H, W]
                grad_norm = grad / (grad.abs().mean() + 1e-8)
                grad_flat = grad_norm.view(grad_norm.size(0), -1)  # [1, C*H*W]

                
                grad_proj = grad_norm  # 未到达子空间投影阶段

                if args.MI_PGD:
                    momentum.mul_(mu).add_(grad_proj)  # μ·m_{t-1} + grad_proj
                    delta_base.data = (delta_base.data - args.alpha * momentum.sign()).clamp(-args.eps, args.eps)
                else:
                    delta_base.data = (delta_base.data - args.alpha * grad_proj.sign()).clamp(-args.eps, args.eps)

                delta_base.grad.zero_()

            #EMA 平滑扰动
            if args.EMA:
                if step == 0:
                    delta_avg = delta_base.clone().detach()
                elif step > args.steps * 0.75:
                    delta_avg.data = 0.9 * delta_avg.data + 0.1 * delta_base.data
                else:
                    delta_avg.data = delta_base.data.clone().detach()


            print(f"[{data['src_class']} → {data['tgt_class']}] Step {step+1}/{args.steps} | Loss: {total_loss.item():.4f}")
            logging.info(f"[{data['src_class']} → {data['tgt_class']}] Step {step+1}/{args.steps} | Loss: {total_loss.item():.4f}")

        
        src_img_tensor = TF.to_tensor(orig_img).unsqueeze(0).to(device)
        adv_img = torch.clamp(src_img_tensor + delta_avg, 0, 1).squeeze(0)

        adv_pil = TF.to_pil_image(adv_img.detach().cpu())
        adv_pil.save(os.path.join(args.output_dir, clean_name))

        # === 新增保存原始扰动张量 ===
        delta_pt_path = os.path.join(args.output_delta_pt_dir, clean_name.replace('.png', '.pt'))
        torch.save(delta_avg.detach().cpu(), delta_pt_path)

        delta_vis = (delta_avg - delta_avg.min()) / (delta_avg.max() - delta_avg.min() + 1e-8)
        delta_pil = TF.to_pil_image(delta_vis.squeeze(0).detach().cpu())
        delta_pil.save(os.path.join(args.output_delta_dir, clean_name))


if __name__ == "__main__":
    attack_pipeline()


    