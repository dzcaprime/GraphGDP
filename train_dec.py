from __future__ import annotations
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from models.temporal_decoder import TemporalDecoder
from models.RNNdecoder import RNNDecoder


def _normalize_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    标准化批数据到稳定契约：
    - A: [B, N, N]
    - X: [B, T, N, C]
    仅支持 dataset 约定键：'adj'、'ts'。
    """
    A = batch["adj"].to(device=device, dtype=torch.float32)  # [B,N,N]
    X = batch["ts"].to(device)  # [B,T,N,C]
    A = A.squeeze(1) if A.dim() == 4 and A.size(1) == 1 else A  # [B,1,N,N] -> [B,N,N]
    return A, X


def _train_one_epoch(
    decoder: RNNDecoder,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    log_interval: int = 50,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    单轮训练：最小化每元素平均 NLL（-log p(X|A)）。
    - 使用 decoder 返回的 loglik（总对数似然），对其做元素均值归一化；
    - MSE 仅用于日志，不参与反传。
    """
    decoder.train()
    loss_train, mse_train = [], []

    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        A, X = _normalize_batch(batch, device)  # A:[B,N,N], X:[B,T,N,C]
        B, T, N, C = X.shape

        # 期望 decoder(A,X,return_loglik=True) 返回 (pred, loglik)
        pred, loglik = decoder(A, X, return_loglik=True, burn_in=True, burn_in_steps=5)
        if loglik is None:
            continue

        # 每元素平均 NLL：注意 X[:,1:] 是 T-1 个目标时刻
        n_elems = B * (T - 1) * N * C
        nll_loss = (-loglik) / max(n_elems, 1)

        # 仅记录，不回传
        mse = F.mse_loss(pred, X[:, 1:])

        nll_loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        optimizer.step()

        loss_train.append(float(nll_loss.item()))
        mse_train.append(float(mse.item()))

        if (step + 1) % log_interval == 0:
            print(
                f"[Train] step {step+1:04d}/{len(loader)} | nll {np.mean(loss_train):.6f} | mse {np.mean(mse_train):.6f}"
            )

    return (np.mean(loss_train), np.mean(mse_train))


@torch.no_grad()
def _eval_one_epoch(decoder: RNNDecoder, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    单轮验证：返回每元素平均 NLL 与 MSE（MSE 仅日志）。
    """
    decoder.eval()
    loss_eval, mse_eval = [], []

    for batch in loader:
        A, X = _normalize_batch(batch, device)
        B, T, N, C = X.shape

        pred, loglik = decoder(A, X, return_loglik=True, burn_in=True, burn_in_steps=5)
        if loglik is None:
            continue

        n_elems = B * (T - 1) * N * C
        nll = (-loglik) / max(n_elems, 1)
        mse = F.mse_loss(pred, X[:, 1:])

        loss_eval.append(float(nll.item()))
        mse_eval.append(float(mse.item()))

    return (float(np.mean(loss_eval)), float(np.mean(mse_eval)))


def train_temporal_decoder(config, workdir: str, train_ds, eval_ds) -> RNNDecoder:
    """
    预训练 TemporalDecoder（NRI 风格）。

    数据集稳定契约
    - adj: [B, 1, N, N] 或 [B, N, N]（DataLoader 批处理后）
    - ts:  [B, T, N, C] 或 [B, N, T, C]（本函数内统一到 [B, T, N, C]）
      单样本可为 [N, T, C] 或 [T, N, C]，仅用于推断 C。

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
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(workdir, exist_ok=True)
    save_path = os.path.join(workdir, "temporal_decoder_best.pth")
    sample = train_ds[0]
    X_s = sample["ts"]
    if X_s.dim() != 3:
        raise ValueError("期望 'ts' 为三维样本张量：[N, T, C] 或 [T, N, C]。")
    C = X_s.size(-1)  # 无论 [N,T,C] 或 [T,N,C]，最后一维均为 C

    # —— Decoder ——
    decoder = RNNDecoder(
        n_in_node=C,
        msg_hid=config.decoder.msg_hidden,
        msg_out=max(1, getattr(config.model, "nf", 128) // 2),
        n_hid=config.decoder.n_hidden,
        do_prob=getattr(config.model, "dropout", 0.1),
        sigma_init=getattr(config.model, "sigma_init", 0.1),
    ).to(device)

    # —— Optimizer & scheduler ——
    optimizer = Adam(
        decoder.parameters(),
        lr=getattr(config.optim, "lr", 1e-3),
        weight_decay=getattr(config.optim, "weight_decay", 1e-5),
    )
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=getattr(config.optim, "lr_decay_step", 50),
        gamma=getattr(config.optim, "lr_decay_gamma", 0.8),
    )

    # —— DataLoaders ——
    batch_size = getattr(config.training, "batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    # —— Training loop ——
    best_val = float("inf")
    epochs = getattr(config.decoder, "epochs", 100)
    log_interval = getattr(config.training, "log_freq", 50)
    grad_clip = getattr(config.optim, "grad_clip", 1.0)

    print("=== Starting Temporal Decoder Training (NRI-style, Route A) ===")
    for epoch in range(epochs):
        nll_tr, mse_tr = _train_one_epoch(
            decoder, train_loader, optimizer, device, log_interval=log_interval, grad_clip=grad_clip
        )
        nll_va, mse_va = _eval_one_epoch(decoder, eval_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train nll {nll_tr:.6f} mse {mse_tr:.6f} | val nll {nll_va:.6f} mse {mse_va:.6f}")

        if nll_va < best_val:
            best_val = nll_va
            torch.save(
                {"model_state_dict": decoder.state_dict(), "epoch": epoch, "val_loss": best_val, "n_in_node": C},
                save_path,
            )
            print(f"Saved best decoder to {save_path} (val_nll {best_val:.6f})")

    # —— Load best and return ——
    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        decoder.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best decoder from {save_path}")

    print("=== Temporal Decoder Training Completed ===")
    return decoder
