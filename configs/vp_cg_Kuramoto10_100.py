from tkinter import NO
import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    # 与 vp_nri 一致
    config.model_type = "sde_with_decoder"

    # ==============Decoder=============
    config.decoder = decoder = ml_collections.ConfigDict()
    # decoder.ckpt = None  # 若指定则加载该 ckpt 的解码器
    decoder.ckpt = "/home/lxx/open_source/GraphGDP/work/vp_cg_Kuramoto10_100/temporal_decoder_best.pth"
    decoder.pretrained = True  # 若 True 则加载预训练解码器
    decoder.epochs = 300  # 解码器训练轮数
    decoder.n_hidden=128
    decoder.msg_hidden=128

    # ============== Training ==============
    config.training = training = ml_collections.ConfigDict()
    training.sde = "vpsde"
    training.continuous = True
    training.reduce_mean = True
    training.batch_size = 128
    training.eval_batch_size = 128
    training.n_iters = 1500000
    training.snapshot_freq = 5000
    training.log_freq = 1000
    training.eval_freq = 5000
    training.snapshot_freq_for_preemption = 5000
    training.snapshot_sampling = True
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
    sampling.guidance_weight = 0.5  # 引导采样权重

    # ============== Evaluation ==============
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 5
    evaluate.end_ckpt = 100
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
    data.name = "_Kuramoto"
    data.temporal = True
    data.ts_features = 4
    data.max_node = 10
    data.delta_t = 100
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
    model.num_scales = 1000 # VPSDE sampling steps
    model.beta_min = 0.1
    model.beta_max = 5.0
    model.dropout = 0.0
    model.ts_cond = True
    model.ts_in = 4

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
