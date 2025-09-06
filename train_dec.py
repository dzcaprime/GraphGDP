from __future__ import annotations
import os
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.temporal_decoder import TemporalDecoder


def _normalize_batch(batch: Dict[str, torch.Tensor],
                     device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准化批数据到稳定契约：
    - A: [B, N, N]
    - X: [B, T, N, C]
    仅支持 dataset 约定键：'adj'、'ts'。
    """
    A = batch['adj'].to(device)          # [B,1,N,N]
    X = batch['ts'].to(device)           # [B,N,T,C]
    A = A.squeeze(1).float()             # -> [B,N,N]
    X = X.permute(0, 2, 1, 3).contiguous().float()  # -> [B,T,N,C]
    return A, X


def _train_one_epoch(decoder: TemporalDecoder,
                     loader: DataLoader,
                     optimizer: Adam,
                     device: torch.device,
                     log_interval: int = 50,
                     grad_clip: float = 1.0,
                     epoch: int | None = None) -> float:
    """单轮训练：最小化每元素平均 -log p(X|A)，避免随 B/T/N/D 线性增长。"""
    decoder.train()
    total, count = 0.0, 0
    num_batches = len(loader)
    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        A, X = _normalize_batch(batch, device)   # A:[B,N,N], X:[B,T,N,C]
        _, loglik = decoder(A, X, return_loglik=True)
        if loglik is None:
            continue
        # 归一化为每元素平均 NLL：除以 B*(T-1)*N*D
        B, T, N, D = X.shape
        denom = float(B * max(T - 1, 1) * N * D)
        loss = (-loglik) / denom
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
        if log_interval and (step % log_interval == 0):
            if epoch is not None:
                print(f"epoch {epoch:03d} step {step:04d}/{num_batches-1:04d}: "
                      f"loss {loss.item():.6f}")
            else:
                print(f"train step {step}: loss {loss.item():.6f}")
    return total / max(count, 1)


@torch.no_grad()
def _eval_one_epoch(decoder: TemporalDecoder,
                    loader: DataLoader,
                    device: torch.device) -> float:
    """单轮验证：返回每元素平均 -log p(X|A)。"""
    decoder.eval()
    total, count = 0.0, 0
    for batch in loader:
        A, X = _normalize_batch(batch, device)
        _, loglik = decoder(A, X, return_loglik=True)
        if loglik is None:
            continue
        # 与训练一致的归一化
        B, T, N, D = X.shape
        denom = float(B * max(T - 1, 1) * N * D)
        loss = (-loglik) / denom
        total += loss.item()
        count += 1
    return total / max(count, 1)


def train_temporal_decoder(config, workdir: str, train_ds, eval_ds) -> TemporalDecoder:
    """
    预训练 TemporalDecoder（NRI 风格）。

    数据集稳定契约
    - adj: [B, 1, N, N]（DataLoader 批处理后）
    - ts:  [B, N, T, C]（单样本为 [N, T, C]）

    参数
    ----
    config : object
        模型/优化/训练配置对象。
    workdir : str
        用于保存检查点的目录。
    train_ds, eval_ds : Dataset
        返回包含键 {'adj', 'ts'} 的字典。

    返回
    ----
    TemporalDecoder
        已加载最佳权重的解码器。
    """
    device = torch.device(getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(workdir, exist_ok=True)
    save_path = os.path.join(workdir, "temporal_decoder_best.pth")

    # 仅依据 dataset 契约('ts':[N,T,C])推断特征维度 C
    sample = train_ds[0]
    X_s = sample['ts']
    if X_s.dim() != 3:
        raise ValueError("期望 'ts' 为 [N, T, C] 张量。")
    C = X_s.size(-1)

    # Decoder
    decoder = TemporalDecoder(
        n_in_node=C,
        msg_hid=config.model.nf,
        msg_out=config.model.nf // 2,
        n_hid=config.model.nf,
        do_prob=0.1,
        sigma_init=0.1
    ).to(device)

    # Optimizer & scheduler
    optimizer = Adam(decoder.parameters(), lr=getattr(config.optim, 'lr', 1e-3),
                     weight_decay=getattr(config.optim, 'weight_decay', 1e-5))
    scheduler = lr_scheduler.StepLR(optimizer,
                                    step_size=getattr(config.optim, 'lr_decay_step', 50),
                                    gamma=getattr(config.optim, 'lr_decay_gamma', 0.8))

    # DataLoaders
    batch_size = config.training.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    # Training loop
    best_val = float('inf')
    epochs = config.decoder.epochs
    log_interval = config.training.log_freq
    grad_clip = config.optim.grad_clip

    print("=== Starting Temporal Decoder Training (NRI-style) ===")
    for epoch in range(epochs):
        tr = _train_one_epoch(decoder, train_loader, optimizer, device,
                              log_interval=log_interval, grad_clip=grad_clip,
                              epoch=epoch)
        va = _eval_one_epoch(decoder, eval_loader, device)
        scheduler.step()

        # 明确区分：loglik（越大越好）与 loss/NLL（越小越好）
        train_loglik = -tr
        val_loglik = -va
        print(f"Epoch {epoch:03d} | train {train_loglik:.6f} loglik | "
              f"val {val_loglik:.6f} loglik | loss {tr:.6f}/{va:.6f}")

        if va < best_val:
            best_val = va
            torch.save({'model_state_dict': decoder.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val,
                        'n_in_node': C}, save_path)
            print(f"Saved best decoder to {save_path} (val_nll {best_val:.6f})")

    # Load best and return
    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        decoder.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best decoder from {save_path}")

    print("=== Temporal Decoder Training Completed ===")
    return decoder

