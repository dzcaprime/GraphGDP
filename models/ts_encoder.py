import torch
import torch.nn as nn
from typing import Optional


class TemporalEncoder(nn.Module):
    """
    将节点级时间序列编码为静态节点嵌入。
    输入: x_ts [B, T, N, Din]
    流程:
      1) 展平批次与节点 -> (B*N, T, Din)
      2) 过单层或多层 GRU/LSTM (这里用 GRU) 取最后隐状态 -> (B*N, H)
      3) 线性投影到 out_dim（可与隐层相同）
      4) reshape 回 [B, N, out_dim]
    设计取舍:
      - 仅取最后隐状态：足够表达时间汇聚，避免存储整段序列
      - 可替换为 TCN/Transformer：保持接口一致，后续易扩展
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}.get(rnn_type.lower())
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")
        self.rnn_type = rnn_type.lower()

        self.rnn = rnn_cls(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        proj_in = hidden_dim * self.num_directions
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(proj_in, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_ts : torch.Tensor
            时间序列输入，形状 [B,T,N,D]。
        Returns
        -------
        torch.Tensor
            每节点静态嵌入 [B,N,out_dim]。
        """
        if x_ts.dim() != 4:
            raise ValueError(f"TemporalEncoder expects [B,T,N,D], got {tuple(x_ts.shape)}")
        B, T, N, D = x_ts.shape
        x_seq = x_ts.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)  # (B*N,T,D)

        out, h = self.rnn(x_seq)  # h: (num_layers*dir, B*N, H)
        if self.rnn_type == "lstm":
            h = h[0]
        last = h.view(self.rnn.num_layers, self.num_directions, B * N, self.hidden_dim)[-1]
        last = last.transpose(0, 1).reshape(B * N, -1)  # (B*N, H*dir)

        emb = self.proj(last).view(B, N, self.out_dim)
        return emb
