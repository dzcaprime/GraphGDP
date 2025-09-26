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


def _normalize_batch(
    batch: Dict[str, Tensor], device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    标准化批数据，消除所有 special cases：
      - A: 邻接矩阵，强制 [B,N,N]
      - X: 时序特征，强制 [B,T,N,C]
      - mask_adj: 邻接掩码，强制 [B,N,N]

    Parameters
    ----------
    batch : dict
        包含键 'adj'、'mask'、'ts'。
    device : torch.device
        目标设备。

    Returns
    -------
    A : Tensor
        邻接矩阵，形状 [B,N,N]。
    X : Tensor
        时序张量，形状 [B,T,N,C]。
    mask_adj : Tensor
        邻接掩码，形状 [B,N,N]。
    """
    A = batch["adj"].to(device=device, dtype=torch.float32)  # [B,1,N,N] 或 [B,N,N]
    if A.dim() == 4 and A.size(1) == 1:
        A = A.squeeze(1)  # [B,N,N]

    X = batch["ts"].to(device=device, dtype=torch.float32)  # [B,T,N,C]

    mask_adj = batch["mask"].to(device=device, dtype=torch.float32)  # [B,1,N,N] 或 [B,N,N]
    if mask_adj.dim() == 4 and mask_adj.size(1) == 1:
        mask_adj = mask_adj.squeeze(1)  # [B,N,N]

    return A, X, mask_adj


def _add_sde_noise(
    A: Tensor,
    sde,
    device: torch.device,
    mask_adj: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[Tensor, Tensor]:
    """
    对邻接矩阵 A 引入噪声，返回 (A_t, t)。
    mask_adj: 如果给定，按 [B,N,N] 层面屏蔽掉对角与已知边。
    """
    B, N, _ = A.shape
    # 采样 t
    t = torch.rand(B, device=device) * (sde.T - eps) + eps  # [B]
    # 对称化高斯噪声
    z = torch.randn_like(A)
    z = torch.tril(z, -1); z = z + z.transpose(-1, -2)

    mean, std = sde.marginal_prob(A, t)  # mean/std 形状 [B,N,N]
    mean = torch.tril(mean, -1); mean = mean + mean.transpose(-1, -2)

    if mask_adj is not None:
        mean = mean * mask_adj
        z = z * mask_adj

    A_t = mean + std.view(-1,1,1) * z
    return A_t, t


def _train_one_epoch(
    decoder: RNNDecoder,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    sde,  # 新增：传入 SDE 实例
    log_interval: int = 50,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    """
    单轮训练：最小化每元素平均 NLL（-log p(X|A_t)）。
    - 对邻接矩阵加扩散噪声，送入解码器。
    """
    decoder.train()
    loss_train, mse_train = [], []

    for step, batch in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        A, X, mask = _normalize_batch(batch, device)  # A:[B,N,N], X:[B,T,N,C]
        B, T, N, C = X.shape

        # 对邻接加噪声
        A_t, t = _add_sde_noise(A, sde, device, mask)

        # 期望 decoder(A_t, X, return_loglik=True) 返回 (pred, loglik)
        pred, loglik = decoder(A_t, X, return_loglik=True, burn_in=True, burn_in_steps=5)
        if loglik is None:
            continue

        # 每元素平均 NLL
        n_elems = B * (T - 1) * N * C
        nll_loss = (-loglik) / max(n_elems, 1)

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
def _eval_one_epoch(decoder: RNNDecoder, loader: DataLoader, device: torch.device, sde) -> Tuple[float, float]:
    """
    单轮验证：对邻接加扩散噪声，返回每元素平均 NLL 与 MSE。
    """
    decoder.eval()
    loss_eval, mse_eval = [], []

    for batch in loader:
        A, X, mask = _normalize_batch(batch, device)
        B, T, N, C = X.shape

        # 对邻接加噪声
        A_t, t = _add_sde_noise(A, sde, device, mask)

        pred, loglik = decoder(A_t, X, return_loglik=True, burn_in=True, burn_in_steps=5)
        if loglik is None:
            continue

        n_elems = B * (T - 1) * N * C
        nll = (-loglik) / max(n_elems, 1)
        mse = F.mse_loss(pred, X[:, 1:])

        loss_eval.append(float(nll.item()))
        mse_eval.append(float(mse.item()))

    return (float(np.mean(loss_eval)), float(np.mean(mse_eval)))


def train_temporal_decoder(config, workdir: str, train_ds, eval_ds, sde=None) -> RNNDecoder:
    """
    预训练 TemporalDecoder（NRI 风格）。
    - 新增参数 sde：用于加噪声训练。
    """
    device = torch.device(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(workdir, exist_ok=True)
    save_path = os.path.join(workdir, "temporal_decoder_best.pth")
    sample = train_ds[0]
    X_s = sample["ts"]
    if X_s.dim() != 3:
        raise ValueError("期望 'ts' 为三维样本张量：[N, T, C] 或 [T, N, C]。")
    C = X_s.size(-1)

    decoder = RNNDecoder(
        n_in_node=C,
        msg_hid=config.decoder.msg_hidden,
        msg_out=max(1, getattr(config.model, "nf", 128) // 2),
        n_hid=config.decoder.n_hidden,
        do_prob=getattr(config.model, "dropout", 0.1),
        sigma_init=getattr(config.model, "sigma_init", 0.1),
    ).to(device)

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

    batch_size = getattr(config.training, "batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    epochs = getattr(config.decoder, "epochs", 100)
    log_interval = getattr(config.training, "log_freq", 50)
    grad_clip = getattr(config.optim, "grad_clip", 1.0)

    # —— SDE 构造（如果未传入则尝试从 config 构造） ——
    if sde is None:
        from sde_lib import VPSDE

        sde = VPSDE(
            beta_min=getattr(config.model, "beta_min", 0.1),
            beta_max=getattr(config.model, "beta_max", 20.0),
            N=getattr(config.model, "num_scales", 1000),
        )

    print("=== Starting Temporal Decoder Training (NRI-style, Route A) ===")
    for epoch in range(epochs):
        nll_tr, mse_tr = _train_one_epoch(
            decoder, train_loader, optimizer, device, sde, log_interval=log_interval, grad_clip=grad_clip
        )
        nll_va, mse_va = _eval_one_epoch(decoder, eval_loader, device, sde)
        scheduler.step()

        print(f"Epoch {epoch:03d} | train nll {nll_tr:.6f} mse {mse_tr:.6f} | val nll {nll_va:.6f} mse {mse_va:.6f}")

        if nll_va < best_val:
            best_val = nll_va
            torch.save(
                {"model_state_dict": decoder.state_dict(), "epoch": epoch, "val_loss": best_val, "n_in_node": C},
                save_path,
            )
            print(f"Saved best decoder to {save_path} (val_nll {best_val:.6f})")

    if os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=device)
        decoder.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best decoder from {save_path}")

    print("=== Temporal Decoder Training Completed ===")
    return decoder
