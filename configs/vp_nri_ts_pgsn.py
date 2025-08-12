import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    config.model_type = 'sde'
    config.seed = 42

    config.data = ml_collections.ConfigDict()
    config.data.name = 'nri_springs10'   # 触发分支
    config.data.temporal = True
    config.data.root = '/home/lxx/open_source/GraphGDP/NRI/data'
    config.data.max_node = 10
    config.data.num_channels = 1         # 输入仍是邻接矩阵
    config.data.ts_features = 4          # ts 的通道数 = 4
    config.data.centered = False
    config.data.dequantization = False

    config.model = ml_collections.ConfigDict()
    config.model.name = 'PGSN'
    config.model.nf = 128
    config.model.num_gnn_layers = 6
    config.model.dropout = 0.0
    config.model.embedding_type = 'positional'
    config.model.rw_depth = 8
    config.model.edge_th = 0.0
    config.model.size_cond = True
    config.model.graph_layer = 'PosTransLayer'
    config.model.heads = 4
    config.model.attn_clamp = False
    config.model.ema_rate = 0.999
    config.model.num_scales = 1000
    config.model.beta_min = 0.1
    config.model.beta_max = 5.0
    config.model.nonlinearity = 'elu'
    # 开启时序条件
    config.model.ts_cond = True
    config.model.ts_in = 4

    config.training = ml_collections.ConfigDict()
    config.training.sde = 'vpsde'
    config.training.n_iters = 1500000
    config.training.batch_size = 64
    config.training.eval_batch_size = 32
    config.training.snapshot_freq = 10000
    config.training.snapshot_freq_for_preemption = 2000
    config.training.snapshot_sampling = False
    config.training.continuous = True
    config.training.reduce_mean = True
    config.training.likelihood_weighting = True
    config.training.log_freq = 200
    config.training.eval_freq = 5000
    
    # sampling（补齐常见字段，后续启用采样/评估时可直接使用）
    config.sampling = ml_collections.ConfigDict()
    config.sampling.method = 'pc'
    config.sampling.predictor = 'reverse_diffusion'
    config.sampling.corrector = 'langevin'
    config.sampling.n_steps_each = 1
    config.sampling.noise_removal = True
    config.sampling.probability_flow = False
    config.sampling.snr = 0.16
    config.sampling.rtol = 1e-5
    config.sampling.atol = 1e-5
    config.sampling.ode_method = 'dopri5'  # 'rk4'
    config.sampling.ode_step = 0.01
    config.sampling.vis_row = 4
    config.sampling.vis_col = 4

    config.eval = ml_collections.ConfigDict()
    config.eval.enable_sampling = False  # 先专注训练
    config.eval.begin_ckpt = 1
    config.eval.end_ckpt = 1
    config.eval.batch_size = 32
    config.eval.num_samples = 100
    config.eval.save_graph = False
    config.eval.mmd_distance = 'RBF'
    config.eval.max_subgraph = False
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-5
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 1000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config