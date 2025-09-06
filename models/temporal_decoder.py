"""
时序解码器模块：基于NRI的MLPDecoder改造，适配GraphGDP的软邻接矩阵输入。

设计原则：
- 输入对称零对角软邻接矩阵A，保持可导性
- 基于图结构预测时序演化
- 高斯似然便于梯度计算和反向传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm (from NRI)."""

    def __init__(self, n_in: int, n_hid: int, n_out: int, do_prob: float = 0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class TemporalDecoder(nn.Module):
    """
    时序解码器（RNN 版）：在隐藏状态上进行消息传递并使用 GRU 风格门控更新，
    以软邻接矩阵 A 为权重进行聚合，保持端到端可导。

    兼容性说明
    ----------
    - 保留类名、forward/compute_likelihood_gradient 接口与返回值不变。
    - 构造参数 msg_out 不再作为外部输出维度使用；消息最终维度固定为 n_hid。
      为了向后兼容，仍接受该参数但不会影响输出维度。

    设计要点
    ----------
    - 消息传递在隐藏状态 H 上执行，更贴近 NRI 中的 RNNDecoder。
    - 门控结构与 RNNDecoder 等价（GRU-style），但以软 A 聚合而非离散边类型。
    - 仍以高斯似然训练，稳定、可微。
    """

    def __init__(
        self,
        n_in_node: int,
        msg_hid: int,
        msg_out: int,  # 兼容保留；最终消息维度对齐到 n_hid
        n_hid: int,
        do_prob: float = 0.0,
        sigma_init: float = 0.1,
    ) -> None:
        super(TemporalDecoder, self).__init__()
        self.n_in_node = n_in_node
        self.n_hid = n_hid
        self.msg_out_shape = n_hid  # 统一到隐状态维度

        # 在隐藏状态上进行消息传递（参考 RNNDecoder，但用软 A）
        self.msg_fc1 = nn.Linear(2 * n_hid, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, n_hid)

        # GRU 风格门控：输入->隐状态、隐状态->隐状态
        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        # 输出 MLP（基于隐状态），预测输入的增量
        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        # 可学习噪声方差（对数参数化）
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
        self.dropout_prob = do_prob

        print("Using learned GraphGDP temporal decoder (RNN-style).")

    def project_adjacency(self, A: torch.Tensor) -> torch.Tensor:
        """确保 A 对称且零对角，并限制到 [0, 1]，保持梯度流动。"""
        A_sym = (A + A.transpose(-2, -1)) / 2
        A_sym = A_sym - torch.diag_embed(
            torch.diagonal(A_sym, dim1=-2, dim2=-1)
        )
        return torch.clamp(A_sym, 0, 1)

    def _message_passing(
        self, hidden: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """
        在隐藏状态上执行一次消息传递并聚合到接收节点。

        Parameters
        ----------
        hidden : torch.Tensor
            [B, N, H]，各节点隐状态。
        A : torch.Tensor
            [B, N, N]，对称零对角软邻接矩阵。

        Returns
        -------
        torch.Tensor
            [B, N, H]，聚合后的消息。
        """
        B, N, H = hidden.shape
        senders_h = hidden.unsqueeze(2).expand(-1, -1, N, -1)   # [B,N,N,H]
        receivers_h = hidden.unsqueeze(1).expand(-1, N, -1, -1) # [B,N,N,H]
        pre_msg = torch.cat([senders_h, receivers_h], dim=-1)   # [B,N,N,2H]

        msg = torch.tanh(self.msg_fc1(pre_msg))
        msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
        msg = torch.tanh(self.msg_fc2(msg))                     # [B,N,N,H]

        edge_w = A.unsqueeze(-1)                                # [B,N,N,1]
        weighted = msg * edge_w                                 # [B,N,N,H]

        # 聚合到接收节点（与旧实现一致：对 senders 维求和）
        agg = weighted.sum(dim=1)                               # [B,N,H]
        return agg

    def single_step_forward(
        self, x_t: torch.Tensor, hidden: torch.Tensor, A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步前向：基于当前观测 x_t、上一隐藏状态 hidden 和图结构 A，
        进行消息传递与门控更新，输出下一步预测与新隐藏状态。

        Parameters
        ----------
        x_t : torch.Tensor
            [B, N, D]，当前时刻输入。
        hidden : torch.Tensor
            [B, N, H]，上一时刻隐藏状态。
        A : torch.Tensor
            [B, N, N]，软邻接矩阵。

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (pred, hidden_new)，分别为 [B,N,D] 与 [B,N,H]。
        """
        agg = self._message_passing(hidden, A)                  # [B,N,H]

        r = torch.sigmoid(self.input_r(x_t) + self.hidden_r(agg))
        i = torch.sigmoid(self.input_i(x_t) + self.hidden_i(agg))
        n = torch.tanh(self.input_n(x_t) + r * self.hidden_h(agg))
        hidden_new = (1 - i) * n + i * hidden                   # [B,N,H]

        # 基于隐藏状态生成对 x_t 的残差预测
        pred = F.dropout(F.relu(self.out_fc1(hidden_new)),
                         p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)),
                         p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)                               # [B,N,D]
        pred = x_t + pred                                       # 残差
        return pred, hidden_new

    def forward(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        pred_steps: int = 1,          # 兼容保留；此实现总是逐步 teacher forcing
        return_loglik: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：基于图结构 A 与观测序列 X 进行逐步预测。

        Parameters
        ----------
        A : torch.Tensor
            [B, N, N]，软邻接矩阵。
        X : torch.Tensor
            [B, T, N, D]，输入时序。
        pred_steps : int
            兼容保留，当前实现按时间步逐步预测（teacher forcing）。
        return_loglik : bool
            是否返回对数似然。

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            X_pred: [B, T-1, N, D]，逐步预测；
            loglik: 标量对数似然（可选）。
        """
        B, T, N, D = X.shape
        A = self.project_adjacency(A)
        
        assert A.size(-1) == X.size(2)

        # 初始化隐藏状态
        hidden = torch.zeros(B, N, self.n_hid, device=X.device)

        preds = []
        total_loglik = torch.tensor(0.0, device=X.device) if return_loglik else None

        for t in range(T - 1):
            x_t = X[:, t]                                      # [B,N,D]
            pred_t, hidden = self.single_step_forward(x_t, hidden, A)
            preds.append(pred_t)

            if return_loglik:
                target = X[:, t + 1]
                sigma = torch.exp(self.log_sigma)
                # 高斯负对数似然：向量化以稳定数值
                mse = ((target - pred_t) ** 2).sum()
                const = N * D * B * torch.log(sigma * (2 * torch.pi) ** 0.5)
                neg_loglik = mse / (2 * sigma**2) + const
                total_loglik = total_loglik - neg_loglik

        X_pred = torch.stack(preds, dim=1)                     # [B,T-1,N,D]
        if return_loglik:
            return X_pred, total_loglik
        return X_pred, None

    def compute_likelihood_gradient(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        计算对数似然关于 A 的梯度，用于图结构引导或采样。

        Parameters
        ----------
        A : torch.Tensor
            [B, N, N]，软邻接矩阵，需可导。
        X : torch.Tensor
            [B, T, N, D]，输入时序。

        Returns
        -------
        torch.Tensor
            与 A 同形状的梯度。
        """
        if not A.requires_grad:
            A = A.requires_grad_()
        _, loglik = self.forward(A, X, return_loglik=True)
        if loglik is not None:
            grad = torch.autograd.grad(
                outputs=loglik, inputs=A, retain_graph=False, create_graph=False
            )[0]
            return grad
        return torch.zeros_like(A)
