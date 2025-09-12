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
        self.heads = int(config.model.heads)
        self.use_ts = config.model.ts_cond
        self.cfg_uncond_prob = config.model.cfg_uncond_prob if self.use_ts else 0.0
        if self.use_ts:
            ts_in = config.data.ts_features
            ts_hid = config.model.ts_hid
            ts_dropout = config.model.ts_dropout
            # 支持 concat/film/both 三种模式
            self.ts_fuse_mode = getattr(config.model, "ts_fuse", "concat").lower()
            assert self.ts_fuse_mode in {"concat", "film", "both"}, f"不支持的 ts_fuse: {self.ts_fuse_mode}"

            # TemporalEncoder 输出维度设为 ts_hid
            self.temporal_encoder = TemporalEncoder(
                in_dim=ts_in,
                hidden_dim=ts_hid,
                out_dim=ts_hid,
                dropout=ts_dropout,
                rnn_type="gru",
                num_layers=1,
            )
            # Concat
            if self.ts_fuse_mode in {"concat", "both"}:
                self.ts_fuse = nn.Sequential(
                    nn.Dropout(ts_dropout),
                    nn.Linear(nf + ts_hid, nf),
                    nn.GELU(),
                    nn.LayerNorm(nf),
                )
            else:
                self.ts_fuse = None

            # 仅在使用 film 或 both 模式时构建 FiLM 配置
            if self.ts_fuse_mode in {"film", "both"}:
                # 简洁方式读取配置
                self.film_node = bool(config.model.film.enable_node)
                self.film_edge = bool(config.model.film.enable_edge)
                self.film_attn_bias = bool(config.model.film.enable_attn_bias)
                self.film_hidden = int(config.model.film.hidden)
                self.film_dropout = float(config.model.film.dropout)
                self.film_gamma_range = tuple(config.model.film.gamma_range)
                self.film_beta_scale = float(config.model.film.beta_scale)
                self.film_warmup = float(config.model.film.warmup)

                # 构建 FiLM 头
                if self.film_node or self.film_edge or self.film_attn_bias:
                    self._build_film_heads(ts_hid)
            else:
                # 非 film/both 模式显式禁用 FiLM
                self.film_node = False
                self.film_edge = False
                self.film_attn_bias = False

            # CFG 条件指示嵌入
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

    def _build_linear_zero(self, in_dim: int, out_dim: int) -> nn.Linear:
        lin = nn.Linear(in_dim, out_dim)
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)
        return lin

    def _build_head(self, in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, self.film_hidden),
            nn.GELU(),
            nn.Dropout(self.film_dropout),
            self._build_linear_zero(self.film_hidden, out_dim),
        )

    def _build_film_heads(self, ts_hid: int) -> None:
        if self.film_node:
            self.film_node_x = self._build_head(ts_hid, 2 * self.nf)
            self.film_node_p = self._build_head(ts_hid, 2 * (self.nf // 2))
        if self.film_edge:
            self.film_edge_node = self._build_head(ts_hid, 2 * (self.nf // 2))
        if self.film_attn_bias:
            self.film_attn_node = self._build_head(ts_hid, self.heads)

    def _map_gamma(self, raw: torch.Tensor) -> torch.Tensor:
        gmin, gmax = self.film_gamma_range
        return gmin + (gmax - gmin) * raw.sigmoid()

    def _map_beta(self, raw: torch.Tensor) -> torch.Tensor:
        return self.film_beta_scale * torch.tanh(raw)

    def _apply_conditional_modulation(
        self, tensor: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, cond_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """
        Apply FiLM modulation conditionally based on mask.

        Design Notes
        ------------
        - Normalizes conditional application to eliminate scattered if/else
        - Handles all mask states uniformly through the main path
        """
        if cond_mask is None or cond_mask.all():
            return gamma * tensor + beta
        elif (~cond_mask).all():
            return tensor
        else:
            result = tensor.clone()
            result[cond_mask] = gamma[cond_mask] * tensor[cond_mask] + beta[cond_mask]
            return result

    def _encode_temporal_sequence(self, ts_raw: torch.Tensor) -> torch.Tensor:
        """
        Encode temporal sequence into static node embeddings.

        Parameters
        ----------
        ts_raw : torch.Tensor
            Input time series [B, T, N, D]

        Returns
        -------
        torch.Tensor
            Node embeddings [B, N, ts_hid]
        """
        if ts_raw.dim() != 4:
            raise ValueError(f"Expected [B,T,N,D], got {tuple(ts_raw.shape)}")
        return self.temporal_encoder(ts_raw)

    def forward(self, x, time_cond, *args, **kwargs):
        mask = kwargs["mask"]
        # 规范化 mask：去 NaN/Inf 并夹紧到 [0,1]，避免传播异常值
        mask = torch.nan_to_num(mask.to(x.dtype), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

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
                num_node = torch.nan_to_num(num_node, nan=0.0, posinf=0.0, neginf=0.0)
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
            cont_adj = torch.nan_to_num(cont_adj, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)

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
        x_degree = torch.nan_to_num(x_degree, nan=0.0, posinf=float(self.degree_max), neginf=0.0).clamp(
            0.0, float(self.degree_max)
        )
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
                drop = torch.bernoulli(torch.full((x.size(0),), self.cfg_uncond_prob, device=x.device)).to(torch.bool)
                cond_mask = ~drop  # True=有条件
            elif ts_in_batch is not None:
                cond_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            else:
                cond_mask = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)

            if self.cond_embed is not None:
                # one-hot: [uncond, cond]
                onehot = torch.stack([~cond_mask, cond_mask], dim=1).to(torch.float32)
                temb = temb + self.cond_embed(onehot)

        # ====== 统一时间序列编码与融合 ======
        ts_emb = None
        if self.use_ts:
            ts_raw = kwargs.get("ts", None)
            if ts_raw is not None and (cond_mask is None or cond_mask.any()):
                ts_emb = self._encode_temporal_sequence(ts_raw)
                ts_emb = torch.nan_to_num(ts_emb, nan=0.0, posinf=0.0, neginf=0.0)

            # Concat 融合路径
            if self.ts_fuse_mode in {"concat", "both"} and ts_emb is not None:
                assert ts_emb.shape[:2] == x_degree.shape[:2], f"ts_emb {ts_emb.shape} vs x_degree {x_degree.shape}"
                x_degree = self._apply_conditional_modulation(
                    x_degree,
                    torch.ones_like(x_degree),
                    self.ts_fuse(torch.cat([x_degree, ts_emb], dim=-1)) - x_degree,
                    cond_mask,
                )

        # ====== 位置编码 ======
        x_pos = modules[m_idx](x_pos)
        m_idx += 1

        # ====== FiLM-A: 节点级调制 ======
        if self.use_ts and self.ts_fuse_mode in {"film", "both"} and self.film_node and ts_emb is not None:

            # 生成调制参数
            gx_bx = self.film_node_x(ts_emb)  # [B,N,2*nf]
            gp_bp = self.film_node_p(ts_emb)  # [B,N,2*(nf//2)]

            gx, bx = torch.split(gx_bx, [self.nf, self.nf], dim=-1)
            gp, bp = torch.split(gp_bp, [self.nf // 2, self.nf // 2], dim=-1)

            # 应用映射函数
            gx = self._map_gamma(gx)
            gp = self._map_gamma(gp)
            bx = self._map_beta(bx)
            bp = self._map_beta(bp)

            # 条件化应用
            x_degree = self._apply_conditional_modulation(x_degree, gx, bx, cond_mask)
            x_pos = self._apply_conditional_modulation(x_pos, gp, bp, cond_mask)

        # ====== FiLM-B: 边级调制 ======
        if self.use_ts and self.ts_fuse_mode in {"film", "both"} and self.film_edge and ts_emb is not None:

            # 生成节点级参数并组合为边级
            ge_be = self.film_edge_node(ts_emb)  # [B,N,2*(nf//2)]
            F = dense_edge_ori.size(-1)
            ge, be = torch.split(ge_be, [F, F], dim=-1)

            # 外和组合保持对称性
            ge_ij = ge.unsqueeze(2) + ge.unsqueeze(1)  # [B,N,N,F]
            be_ij = be.unsqueeze(2) + be.unsqueeze(1)  # [B,N,N,F]

            # 应用映射
            ge_ij = self._map_gamma(ge_ij)
            be_ij = self._map_beta(be_ij)

            # 条件化应用到边特征
            dense_edge_ori = self._apply_conditional_modulation(dense_edge_ori, ge_ij, be_ij, cond_mask)
            dense_edge_spd = self._apply_conditional_modulation(dense_edge_spd, ge_ij, be_ij, cond_mask)

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
            b = idx[:, 0]
            i = idx[:, 1]
            j = idx[:, 2]
            row = i + b * N
            col = j + b * N
            edge_index = torch.stack((row, col), dim=0)

        # ====== FiLM-C：注意力偏置 ======
        attn_bias_sparse = None
        if (
            self.use_ts
            and self.ts_fuse_mode in {"film", "both"}
            and self.film_attn_bias
            and ts_emb is not None
            and idx.numel() > 0
        ):

            try:
                bnode = self.film_attn_node(ts_emb)  # [B,N,H]
                b_i = bnode[b, i, :]  # [E,H]
                b_j = bnode[b, j, :]  # [E,H]
                attn_bias_sparse = self._map_beta(b_i + b_j)  # [E,H]

                # 条件化门控
                if cond_mask is not None and not cond_mask.all():
                    # 为稀疏边构建条件掩码
                    edge_cond_mask = cond_mask[b]
                    attn_bias_sparse = attn_bias_sparse * edge_cond_mask.unsqueeze(-1)

            except (IndexError, RuntimeError) as e:
                # 边索引异常时安全回退
                attn_bias_sparse = None

        # Run GNN layers
        h_dense_edge = modules[m_idx](
            x_degree,
            x_pos,
            edge_index,
            dense_edge_ori,
            dense_edge_spd,
            dense_index,
            temb,
            attn_bias_sparse=attn_bias_sparse,  # 新增
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
