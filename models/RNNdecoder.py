import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class RNNDecoder(nn.Module):
    """
    时序解码器（自回归版，支持 burn-in）：
    在现有 GraphGDP TemporalDecoder 基础上改造，允许部分时间步使用预测结果作为输入，
    提升对长时序依赖和 A 敏感性。

    兼容性：
    - forward / compute_likelihood_gradient 接口保持一致。
    - 返回 (X_pred, loglik)。
    """

    def __init__(
        self,
        n_in_node: int,
        msg_hid: int,
        msg_out: int,  # 保留兼容
        n_hid: int,
        do_prob: float = 0.0,
        sigma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_in_node = n_in_node
        self.n_hid = n_hid
        self.msg_out_shape = n_hid

        # 消息传递
        self.msg_fc1 = nn.Linear(2 * n_hid, msg_hid)
        self.msg_fc2 = nn.Linear(msg_hid, n_hid)

        # GRU 门控
        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.msg_r = nn.Linear(n_hid, n_hid, bias=False)
        self.msg_i = nn.Linear(n_hid, n_hid, bias=False)
        self.msg_n = nn.Linear(n_hid, n_hid, bias=False)

        # 输出
        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
        self.dropout_prob = do_prob

        print("Using autoregressive GraphGDP temporal decoder (RNN-style).")

    def project_adjacency(self, A: torch.Tensor, temp: float = 1.0, eps: float = 1e-6):
        A = 0.5 * (A + A.transpose(-2, -1))
        A = torch.sigmoid(A / temp)
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0).expand_as(A)
        A = A * (1.0 - I)
        deg = A.sum(dim=-1, keepdim=True) + eps
        return A / deg

    def _message_passing(self, hidden: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, H = hidden.shape
        senders_h = hidden.unsqueeze(2).expand(-1, -1, N, -1)
        receivers_h = hidden.unsqueeze(1).expand(-1, N, -1, -1)
        pre_msg = torch.cat([senders_h, receivers_h], dim=-1)

        msg = torch.tanh(self.msg_fc1(pre_msg))
        msg = F.dropout(msg, p=self.dropout_prob, training=self.training)
        msg = torch.tanh(self.msg_fc2(msg))

        edge_w = A.unsqueeze(-1)
        weighted = msg * edge_w
        return weighted.sum(dim=1)  # [B,N,H]

    def single_step_forward(
        self, x_t: torch.Tensor, hidden: torch.Tensor, A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agg = self._message_passing(hidden, A)
        r = torch.sigmoid(self.input_r(x_t) + self.hidden_r(hidden) + self.msg_r(agg))
        i = torch.sigmoid(self.input_i(x_t) + self.hidden_i(hidden) + self.msg_i(agg))
        n = torch.tanh(self.input_n(x_t) + r * self.hidden_h(hidden) + self.msg_n(agg))
        hidden_new = (1 - i) * n + i * hidden

        pred = F.dropout(F.relu(self.out_fc1(hidden_new)), p=self.dropout_prob, training=self.training)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob, training=self.training)
        pred = self.out_fc3(pred)
        pred = x_t + pred
        return pred, hidden_new

    def forward(
        self,
        A: torch.Tensor,
        X: torch.Tensor,
        pred_steps: int = 1,  # 保留兼容
        return_loglik: bool = True,
        burn_in: bool = False,
        burn_in_steps: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T, N, D = X.shape
        A = self.project_adjacency(A)
        hidden = torch.zeros(B, N, self.n_hid, device=X.device)

        preds, total_loglik = [], torch.tensor(0.0, device=X.device) if return_loglik else None

        for t in range(T - 1):
            if not burn_in:
                # 原始：teacher forcing
                x_t = X[:, t]
            else:
                if t < burn_in_steps:
                    x_t = X[:, t]
                else:
                    x_t = preds[-1]  # 使用自预测

            pred_t, hidden = self.single_step_forward(x_t, hidden, A)
            preds.append(pred_t)

            if return_loglik:
                target = X[:, t + 1]
                sigma = torch.exp(self.log_sigma).clamp(min=1e-4, max=10.0)
                diff2 = (target - pred_t).pow(2).sum()
                count = B * N * D
                nll_t = 0.5 * count * torch.log(2 * torch.pi * sigma * sigma) + diff2 / (2 * sigma * sigma)
                total_loglik = total_loglik - nll_t

        X_pred = torch.stack(preds, dim=1)
        return (X_pred, total_loglik) if return_loglik else (X_pred, None)

    def compute_likelihood_gradient_backup(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        if not A.requires_grad:
            A = A.requires_grad_()
        _, loglik = self.forward(A, X, return_loglik=True)
        if loglik is not None:
            grad = torch.autograd.grad(loglik, A, retain_graph=False, create_graph=False)[0]
            return grad
        return torch.zeros_like(A)
    
    def compute_likelihood_gradient(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        为 CG 引导取梯度：对“已投影”的 A 取 ∇_A loglik，避免把投影的非线性夹进链式。
        """
        # <<< 固定投影，不让梯度穿过投影
        with torch.no_grad():
            A_clean = self.project_adjacency(A)

        A_var = A_clean.clone().detach().requires_grad_(True)

        # 复用 forward 主体，但不要再次投影：直接走单步循环体
        N = A.size(-1)
        B, T, N_x, D = X.shape
        assert N_x == N

        hidden = torch.zeros(B, N, self.n_hid, device=X.device, dtype=X.dtype)

        total_loglik = torch.tensor(0.0, device=X.device, dtype=X.dtype)
        for t in range(T - 1):
            x_t = X[:, t]
            pred_t, hidden = self.single_step_forward(x_t, hidden, A_var)
            target = X[:, t + 1]
            sigma = torch.exp(self.log_sigma).clamp(min=1e-4, max=10.0)
            diff2 = (target - pred_t).pow(2).sum()
            count = B * N * D
            const = 0.5 * count * torch.log(torch.tensor(2*math.pi, dtype=X.dtype, device=X.device) * sigma * sigma)
            nll_t = const + diff2 / (2 * sigma * sigma)
            total_loglik = total_loglik - nll_t

        grad = torch.autograd.grad(total_loglik, A_var, retain_graph=False, create_graph=False)[0]
        return grad