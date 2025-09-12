import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # 与 vp_nri 一致
    config.model_type = "sde_with_decoder"

    # ==============Decoder=============
    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.ckpt = "/home/lxx/open_source/GraphGDP/work/temporal/temporal_decoder_best.pth"  # 若指定则加载该 ckpt 的解码器
    decoder.pretrained = True  # 若 True 则加载预训练解码器
    decoder.epochs = 500  # 解码器训练轮数

    # ============== Training ==============
    config.training = training = ml_collections.ConfigDict()
    training.sde = "vpsde"
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 256
    training.eval_batch_size = 256
    training.n_iters = 1500000
    training.snapshot_freq = 10000
    training.log_freq = 500
    training.eval_freq = 5000
    training.snapshot_freq_for_preemption = 5000
    training.snapshot_sampling = True # sampling in training snapshots
    training.likelihood_weighting = False

    # 本文件特有：解码器与引导采样配置（保留）
    training.test_guided_sampling = True

    # ============== Sampling ==============
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = "guided_pc"
    sampling.predictor = "euler_maruyama"
    sampling.corrector = "none"
    sampling.rtol = 1e-5
    sampling.atol = 1e-5
    sampling.ode_method = "dopri5"  # 或 'rk4'
    sampling.ode_step = 0.01

    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.vis_row = 4
    sampling.vis_col = 4
    sampling.guidance_weight = 0.0  # decoder引导采样权重
    sampling.cfg_scale = 2.0  # CFG guidance scale (w)，典型 1.5~2.5

    # ============== Evaluation ==============
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 60
    evaluate.end_ckpt = 110
    evaluate.batch_size = 1000
    evaluate.enable_sampling = True
    evaluate.num_samples = 1000
    evaluate.mmd_distance = "RBF"
    evaluate.max_subgraph = False
    evaluate.save_graph = False

    # ============== Data ==============
    config.data = data = ml_collections.ConfigDict()
    data.split_ratio = 0.8
    data.centered = False
    data.dequantization = False
    data.root = "/home/lxx/open_source/GraphGDP/data"
    data.name = "nri_springs10"
    data.temporal = True
    data.ts_features = 4
    data.max_node = 10
    data.num_channels = 1

    # ============== Model ==============
    config.model = model = ml_collections.ConfigDict()
    model.name = "PGSN"
    model.ema_rate = 0.9999
    model.nonlinearity = "swish"
    model.nf = 128
    model.num_gnn_layers = 4
    model.size_cond = False
    model.embedding_type = "positional"
    model.rw_depth = 16
    model.graph_layer = "PosTransLayer"  # 与 vp_nri 对齐
    model.edge_th = -1.0
    model.heads = 8
    model.attn_clamp = False
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 5.0
    model.dropout = 0.0
    model.ts_cond = True
    model.ts_in = 4  # 输入时间序列单步特征维度（需与 data.ts_features 对齐）

    # ---- 以下为 PGSN 中 TemporalEncoder / 融合模块所需新增字段 ----
    model.ts_hid = 64  # 隐藏/输出维度；PGSN 内若未提供则默认 nf//2，这里显式给出以保证可重复
    model.ts_dropout = 0.1  # 时间序列编码与融合过程中的 dropout，映射到 TemporalEncoder 与融合 MLP
    model.ts_fuse = "both"  # 融合策略（concat, film, both）
    model.cfg_uncond_prob = 0.12  # p_uncond：训练阶段随机无条件概率 (0.1~0.15 推荐)

    #===============FiLM================
    model.film = film = ml_collections.ConfigDict()
    film.enable_node = True       # 节点级 FiLM (A)
    film.enable_edge = True      # 边级 FiLM (B)
    film.enable_attn_bias = True # 注意力偏置 FiLM (C)
    film.hidden = 64              # FiLM 头隐藏维度
    film.dropout = 0.1            # FiLM 头 dropout
    film.gamma_range = [0.5, 1.5] # gamma 缩放范围
    film.beta_scale = 0.1         # beta 偏置尺度
    film.warmup = 0.1             # warmup 比例

    # ============== Optim ==============
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.0
    optim.lr_decay_step = 50
    optim.lr_decay_gamma = 0.8

    # ============== Misc ==============
    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    return config
