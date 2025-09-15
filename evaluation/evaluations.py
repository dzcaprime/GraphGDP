import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,           # 已使用
    average_precision_score, # AUC-PR
    precision_recall_curve,  # F1@最佳阈值
)

def _compute_batch_auroc(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    使用 sklearn 计算一批样本的 AUROC（逐样本 -> 再平均）。
    约定输入均为 [B, 1, N, N]，仅统计下三角、去对角，且受 mask 限制。
    """
    # 统一到 [B, 1, N, N]
    if preds.dim() == 3:
        preds = preds.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    B, _, N, _ = preds.shape
    tril = torch.tril(torch.ones(N, N, device=preds.device, dtype=torch.bool), diagonal=-1)

    aurocs = []
    for i in range(B):
        valid = (mask[i, 0] > 0.5) & tril
        y_score = preds[i, 0][valid].detach().cpu().numpy()
        y_true = (target[i, 0][valid] > 0.5).float().detach().cpu().numpy()
        # 极端情形：全正或全负，返回 0.5 避免报错
        if y_true.sum() == 0 or y_true.sum() == y_true.size:
            aurocs.append(0.5)
        else:
            aurocs.append(float(roc_auc_score(y_true, y_score)))
    return float(np.mean(aurocs)) if len(aurocs) > 0 else 0.0

def _compute_batch_aupr(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """
    批量 AUC-PR（Average Precision，逐样本->再平均）。
    约定：输入为 [B,1,N,N] 或 [B,N,N]；仅统计下三角、去对角，受 mask 限制。
    边界：无正样本 -> 0.0；全正样本 -> 1.0；无有效样本 -> 0.0。
    """
    if preds.dim() == 3: preds = preds.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    if mask.dim() == 3: mask = mask.unsqueeze(1)

    B, _, N, _ = preds.shape
    tril = torch.tril(torch.ones(N, N, device=preds.device, dtype=torch.bool), diagonal=-1)

    aps = []
    for i in range(B):
        valid = (mask[i, 0] > 0.5) & tril
        y_score = preds[i, 0][valid].detach().cpu().numpy()
        y_true_t = (target[i, 0][valid] > 0.5).float()
        y_true = y_true_t.detach().cpu().numpy()

        total = int(y_true_t.numel())
        pos_cnt = int(y_true_t.sum().item())
        if total == 0:
            aps.append(0.0); continue
        if pos_cnt == 0:
            aps.append(0.0); continue
        if pos_cnt == total:
            aps.append(1.0); continue

        aps.append(float(average_precision_score(y_true, y_score)))

    return float(np.mean(aps)) if len(aps) > 0 else 0.0

def _compute_batch_best_f1(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    """
    批量 F1@最佳阈值（基于 PR 曲线阈值）。返回 (mean_best_f1, mean_best_tau)。
    边界：无样本/无正样本 -> F1=0、tau=0.5；全正 -> F1=1、tau=0.0。
    """
    if preds.dim() == 3: preds = preds.unsqueeze(1)
    if target.dim() == 3: target = target.unsqueeze(1)
    if mask.dim() == 3: mask = mask.unsqueeze(1)

    B, _, N, _ = preds.shape
    tril = torch.tril(torch.ones(N, N, device=preds.device, dtype=torch.bool), diagonal=-1)

    best_f1_list, best_tau_list = [], []
    for i in range(B):
        valid = (mask[i, 0] > 0.5) & tril
        y_score = preds[i, 0][valid].detach().cpu().numpy()
        y_true_t = (target[i, 0][valid] > 0.5).float()
        y_true = y_true_t.detach().cpu().numpy()

        total = int(y_true_t.numel())
        pos_cnt = int(y_true_t.sum().item())
        if total == 0:
            best_f1_list.append(0.0); best_tau_list.append(0.5); continue
        if pos_cnt == 0:
            best_f1_list.append(0.0); best_tau_list.append(0.5); continue
        if pos_cnt == total:
            best_f1_list.append(1.0); best_tau_list.append(0.0); continue

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        f1 = (2.0 * precision * recall) / np.maximum(precision + recall, 1e-12)
        idx = int(np.nanargmax(f1))
        best_f1 = float(f1[idx])
        if thresholds.size > 0:
            tau = float(min(1.0, thresholds[0] + 1e-12)) if idx == 0 else float(np.clip(thresholds[idx - 1], 0.0, 1.0))
        else:
            tau = 0.5

        best_f1_list.append(best_f1)
        best_tau_list.append(tau)

    return (
        float(np.mean(best_f1_list)) if best_f1_list else 0.0,
        float(np.mean(best_tau_list)) if best_tau_list else 0.5,
    )

def compute_edge_metrics(preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    """
    统一计算：AUC-PR（主）、AUC-ROC、F1@最佳阈值；逐样本->再平均。
    返回: {'aupr', 'auroc', 'f1_best', 'best_threshold'}。
    """
    aupr = _compute_batch_aupr(preds, target, mask)
    auroc = _compute_batch_auroc(preds, target, mask)
    f1_best, best_tau = _compute_batch_best_f1(preds, target, mask)
    return {'aupr': aupr, 'auroc': auroc, 'f1_best': f1_best, 'best_threshold': best_tau}