import torch.nn as nn
import torch
import functools
from . import utils, layers, gnns
from .ts_encoder import TemporalEncoder  # 新增：时间序列编码器

get_act = layers.get_act
conv1x1 = layers.conv1x1


@utils.register_model(name="PGSN")
class PGSN(nn.Module):
    """Position enhanced graph score network."""

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.act = act = get_act(config)

        # get model construction paras
        self.nf = nf = config.model.nf
        self.num_gnn_layers = num_gnn_layers = config.model.num_gnn_layers
        dropout = config.model.dropout
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        self.rw_depth = rw_depth = config.model.rw_depth
        self.edge_th = config.model.edge_th

        # ====== 时间条件（可选）======
        self.use_ts = config.model.ts_cond
        self.cfg_uncond_prob = config.model.cfg_uncond_prob if self.use_ts else 0.0
        if self.use_ts:
            ts_in = config.data.ts_features
            ts_hid = config.model.ts_hid
            ts_dropout = config.model.ts_dropout
            self.ts_fuse_mode = config.model.ts_fuse.lower()
            if self.ts_fuse_mode != "concat":
                raise ValueError(f"当前仅实现 concat 融合，收到 {self.ts_fuse_mode}")
            # TemporalEncoder 输出维度设为 ts_hid，后续线性映射回 nf
            self.temporal_encoder = TemporalEncoder(
                in_dim=ts_in,
                hidden_dim=ts_hid,
                out_dim=ts_hid,
                dropout=ts_dropout,
                rnn_type="gru",
                num_layers=1,
            )
            self.ts_fuse = nn.Sequential(
                nn.Dropout(ts_dropout),
                nn.Linear(nf + ts_hid, nf),
                nn.GELU(),
                nn.LayerNorm(nf),
            )
            # 若需要做 CFG，则增加 cond/uncond 指示嵌入 (2 -> nf*4)
            if self.cfg_uncond_prob > 0:
                self.cond_embed = nn.Linear(2, nf * 4)
            else:
                self.cond_embed = None
        # ====== 其余模块 ======
        modules = []
        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "positional":
            embed_dim = nf
        else:
            raise ValueError(f"embedding type {embedding_type} unknown.")

        # timestep embedding layers
        modules.append(nn.Linear(embed_dim, nf * 4))
        modules.append(nn.Linear(nf * 4, nf * 4))

        # graph size condition embedding
        self.size_cond = size_cond = config.model.size_cond
        if size_cond:
            self.size_onehot = functools.partial(nn.functional.one_hot, num_classes=config.data.max_node + 1)
            modules.append(nn.Linear(config.data.max_node + 1, nf * 4))
            modules.append(nn.Linear(nf * 4, nf * 4))

        channels = config.data.num_channels
        assert channels == 1, "Without edge features."

        # degree onehot
        self.degree_max = self.config.data.max_node // 2
        self.degree_onehot = functools.partial(nn.functional.one_hot, num_classes=self.degree_max + 1)

        # project edge features
        modules.append(conv1x1(channels, nf // 2))
        modules.append(conv1x1(rw_depth + 1, nf // 2))

        # project node features
        self.x_ch = nf
        self.pos_ch = nf // 2
        modules.append(nn.Linear(self.degree_max + 1, self.x_ch))
        modules.append(nn.Linear(rw_depth, self.pos_ch))

        # GNN
        modules.append(
            gnns.pos_gnn(
                act,
                self.x_ch,
                self.pos_ch,
                nf,
                config.data.max_node,
                config.model.graph_layer,
                num_gnn_layers,
                heads=config.model.heads,
                edge_dim=nf // 2,
                temb_dim=nf * 4,
                dropout=dropout,
                attn_clamp=config.model.attn_clamp,
            )
        )

        # output
        modules.append(conv1x1(nf // 2, nf // 2))
        modules.append(conv1x1(nf // 2, channels))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, *args, **kwargs):
        mask = kwargs["mask"]
        # 规范化 mask：去 NaN/Inf 并夹紧到 [0,1]，避免传播异常值
        mask = torch.nan_to_num(
            mask.to(x.dtype), nan=0.0, posinf=1.0, neginf=0.0
        ).clamp(0.0, 1.0)

        modules = self.all_modules
        m_idx = 0

        # Sinusoidal positional embeddings
        timesteps = time_cond
        temb = layers.get_timestep_embedding(timesteps, self.nf)

        # time embedding
        temb = modules[m_idx](temb)
        m_idx += 1
        temb = modules[m_idx](self.act(temb))
        m_idx += 1

        if self.size_cond:
            with torch.no_grad():
                node_mask = utils.mask_adj2node(mask.squeeze(1))  # [B, N]
                num_node = torch.sum(node_mask, dim=-1)  # [B]
                # 安全化：去 NaN/Inf -> 夹到 [0, max_node] -> 转 long
                num_node = torch.nan_to_num(
                    num_node, nan=0.0, posinf=0.0, neginf=0.0
                )
                max_node = self.config.data.max_node
                num_node = num_node.clamp(0, max_node).to(torch.long)
                num_node = self.size_onehot(num_node).to(torch.float)

            num_node_emb = modules[m_idx](num_node)
            m_idx += 1
            num_node_emb = modules[m_idx](self.act(num_node_emb))
            m_idx += 1
            temb = temb + num_node_emb

        if not self.config.data.centered:
            # rescale the input data to [-1, 1]
            x = x * 2.0 - 1.0
        # 数值清理：去除 NaN/Inf，并夹紧到 [-1,1]，防止后续投影/卷积被污染
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

        with torch.no_grad():
            # continuous-valued graph adjacency matrices
            cont_adj = ((x + 1.0) / 2.0).clone()
            cont_adj = (cont_adj * mask).squeeze(1)  # [B, N, N]
            cont_adj = cont_adj.clamp(min=0.0, max=1.0)
            if self.edge_th > 0.0:
                cont_adj[cont_adj < self.edge_th] = 0.0

            # 进一步数值稳定：去 NaN/Inf 并夹紧
            cont_adj = torch.nan_to_num(
                cont_adj, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)

            # discretized graph adjacency matrices
            adj = x.squeeze(1).clone()  # [B, N, N]
            adj[adj >= 0.0] = 1.0
            adj[adj < 0.0] = 0.0
            adj = adj * mask.squeeze(1)

        # extract RWSE and Shortest-Path Distance
        x_pos, spd_onehot = utils.get_rw_feat(self.rw_depth, adj)

        # edge [B, N, N, F]
        dense_edge_ori = modules[m_idx](x).permute(0, 2, 3, 1)
        m_idx += 1
        dense_edge_spd = modules[m_idx](spd_onehot).permute(0, 2, 3, 1)
        m_idx += 1

        # Use Degree as node feature
        x_degree = torch.sum(cont_adj, dim=-1)  # [B, N]
        # 安全化度数：去 NaN/Inf -> 夹到 [0, degree_max] -> 转 long 并二次 clamp
        x_degree = torch.nan_to_num(
            x_degree, nan=0.0, posinf=float(self.degree_max), neginf=0.0
        ).clamp(0.0, float(self.degree_max))
        x_degree_long = x_degree.to(torch.long).clamp(0, self.degree_max)
        x_degree = self.degree_onehot(x_degree_long).to(torch.float)  # [B,N,K]
        x_degree = modules[m_idx](x_degree)  # [B,N,nf]
        m_idx += 1

        # ---------- 条件指示 (CFG 训练) ----------
        cond_mask = None  # True 表示使用 ts 条件；False 表示无条件
        if self.use_ts and self.cfg_uncond_prob > 0:
            ts_in_batch = kwargs.get("ts", None)
            if self.training and ts_in_batch is not None:
                # 采样 Bernoulli：1 表示“无条件”(drop)，转成 cond_mask False
                drop = torch.bernoulli(
                    torch.full((x.size(0),), self.cfg_uncond_prob, device=x.device)
                ).to(torch.bool)
                cond_mask = ~drop  # True=有条件
            elif ts_in_batch is not None:
                cond_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            else:
                cond_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

            if self.cond_embed is not None:
                # one-hot: [uncond, cond]
                onehot = torch.stack([~cond_mask, cond_mask], dim=1).to(torch.float32)
                temb = temb + self.cond_embed(onehot)

        # ====== 时间序列条件融合（concat + CFG dropout）======
        if self.use_ts:
            ts_raw = kwargs.get("ts", None)
            assert ts_raw is not None, "需要提供时间序列 ts 条件"
            # expect ts_raw shape [B, T, N, D]
            assert ts_raw.size(2) == x_degree.size(1), "ts节点数应与x节点数匹配"
            # 简化：编码全 batch；若需进一步提速，可只编码 cond_mask==True 子集
            ts_emb = self.temporal_encoder(ts_raw)  # [B,N,ts_hid]
            assert ts_emb.size(1) == x_degree.size(1), "ts_emb节点数应与x节点数匹配"
            if cond_mask is None or cond_mask.all():
                x_degree = self.ts_fuse(torch.cat([x_degree, ts_emb], dim=-1))
            elif (~cond_mask).all():
                pass  # 全部无条件直接跳过
            else:
                fused = self.ts_fuse(torch.cat([x_degree[cond_mask], ts_emb[cond_mask]], dim=-1))
                x_deg_clone = x_degree.clone()
                x_deg_clone[cond_mask] = fused
                x_degree = x_deg_clone
        # ====== 位置编码 ======
        x_pos = modules[m_idx](x_pos)
        m_idx += 1

        # Dense to sparse node [BxN, -1]
        x_degree = x_degree.reshape(-1, self.x_ch)
        x_pos = x_pos.reshape(-1, self.pos_ch)

        # 从 batched 稠密邻接安全构造 edge_index
        B, N, _ = cont_adj.shape
        dense_index = cont_adj.nonzero(as_tuple=True)
        idx = cont_adj.nonzero(as_tuple=False)
        if idx.numel() == 0:
            device = cont_adj.device
            edge_index = torch.zeros((2, 1), dtype=torch.long, device=device)
        else:
            b = idx[:, 0]; i = idx[:, 1]; j = idx[:, 2]
            row = i + b * N
            col = j + b * N
            edge_index = torch.stack((row, col), dim=0)

        # Run GNN layers
        h_dense_edge = modules[m_idx](
            x_degree, x_pos, edge_index,
            dense_edge_ori, dense_edge_spd, dense_index, temb
        )
        m_idx += 1

        # Output
        h = self.act(modules[m_idx](self.act(h_dense_edge)))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        # make edge estimation symmetric
        h = (h + h.transpose(2, 3)) / 2.0 * mask

        assert m_idx == len(modules)

        return h