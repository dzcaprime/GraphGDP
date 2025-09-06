import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
import torch
import numpy as np
import random
import torch.nn as nn  # 新增
from typing import Optional  # 新增
import logging
import time
from absl import flags
from torch.utils import tensorboard
from torch_geometric.loader import DataLoader
import pickle
import networkx as nx  # 兼容 NetworkX 3.x

if not hasattr(nx, "from_numpy_matrix"):
    nx.from_numpy_matrix = nx.from_numpy_array

from models import pgsn
import losses
import sampling
from models import utils as mutils
from models.temporal_decoder import TemporalDecoder  # 新增：导入时序解码器
from models.ema import ExponentialMovingAverage
import datasets
from evaluation import get_stats_eval, get_nn_eval
import sde_lib
import visualize
from utils import *
from torch_geometric.data import Data  # 增加：用于构造评估用 Data 对象
from train_dec import train_temporal_decoder  # 新增：导入解码器训练函数


def edge_accuracy(preds, target, threshold=0.5):
    """Compute edge accuracy for predicted vs target adjacency matrices.

    Args:
        preds: [B, 1, N, N] or [B, N, N] predicted adjacency (continuous).
        target: [B, 1, N, N] or [B, N, N] target adjacency (binary).
        threshold: Threshold for binarizing predictions.

    Returns:
        Accuracy as a scalar tensor.
    """
    preds = preds.squeeze(1) if preds.dim() == 4 else preds  # [B, N, N]
    target = target.squeeze(1) if target.dim() == 4 else target  # [B, N, N]
    preds_binary = (preds > threshold).float()
    correct = preds_binary.eq(target).float().sum()
    total = target.numel()
    return correct / total


def _build_sde(config):
    sde_name = config.training.sde.lower()
    if sde_name == "vpsde":
        return sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
    else:
        raise ValueError(f"Unknown SDE: {sde_name}")


def _project_adjacency(A: torch.Tensor, mask: Optional[torch.Tensor] = None):
    # A: [B,1,N,N] or [B,N,N]
    if A.dim() == 4:
        A = A.squeeze(1)
    A = 0.5 * (A + A.transpose(-1, -2))
    A = A - torch.diag_embed(torch.diagonal(A, dim1=-2, dim2=-1))
    if mask is not None:
        A = A * mask.squeeze(1)
    return A.unsqueeze(1)


@torch.no_grad()
def _tweedie_denoise(sde, A_t, t, score, eps=1e-8):
    # A_t, score: [B,1,N,N]; t: [B] 或标量
    mean, std = sde.marginal_prob(torch.zeros_like(A_t), t)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)
    return (A_t + (std**2) * score) / (mean + eps)


def _compute_joint_regularizer(
    decoder: nn.Module,
    sde,
    model: nn.Module,
    batch: dict,
    guidance_weight: float = 1.0,
    eps: float = 1e-8,
    continuous: bool = True,
):
    """
    路线 C：对当前去噪估计 A0_hat 施加 -λ log pψ(X|A0_hat) 正则，不回传穿过 decoder。
    返回标量 joint_loss（已乘以 guidance_weight）。
    需要 batch 至少包含：
      - 'adj': [B,1,N,N] 真图 A0
      - 'mask': [B,1,N,N] 掩码
      - 'ts': 观测时序，传给 decoder
    """
    if decoder is None or guidance_weight <= 0.0:
        return torch.tensor(0.0, device=batch["adj"].device)

    A0 = batch["adj"]  # [B,1,N,N]
    mask = batch.get("mask", torch.ones_like(A0))
    ts = batch.get("ts", None)

    # 采样一个随机 t，构造 A_t 并通过 score 得到 A0_hat
    B = A0.size(0)
    device = A0.device
    t = torch.rand(B, device=device) * (sde.T - 1e-5) + 1e-5  # (0, T]
    z = torch.randn_like(A0)
    mean, std = sde.marginal_prob(torch.zeros_like(A0), t)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)
    A_t = mean * A0 + std * z

    score_fn = mutils.get_score_fn(sde, model, train=True, continuous=continuous)
    with torch.no_grad():
        score = score_fn(A_t, t, mask=mask, ts=ts)
        A0_hat = _tweedie_denoise(sde, A_t, t, score)
        A0_hat = _project_adjacency(A0_hat, mask=mask).squeeze(1)  # [B,N,N]

    # 只对 A0_hat 的副本求 decoder 前向，取 loglik，作为数值正则
    decoder.eval()
    pred, loglik = decoder(A0_hat.detach(), ts, return_loglik=True)
    if loglik is None:
        return torch.tensor(0.0, device=device)
    joint_loss = -guidance_weight * loglik.mean()
    return joint_loss


FLAGS = flags.FLAGS


def set_random_seed(config):
    seed = config.seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sde_train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries.
            If this contains checkpoint training will be resumed from the latest checkpoint.
    """

    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    tb_dir = os.path.join(workdir, "tensorboard")
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directly
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(os.path.dirname(checkpoint_meta_dir)):
        os.makedirs(os.path.dirname(checkpoint_meta_dir))
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state["step"])

    # Build dataloader and iterators
    train_ds, eval_ds, test_ds, n_node_pmf = datasets.get_dataset(config)
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)

    train_iter = iter(train_loader)
    # create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # Build sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.eval_batch_size,
            config.data.num_channels,
            config.data.max_node,
            config.data.max_node,
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    logging.info("Starting training loop at step %d." % (initial_step,))

    # 小工具：从数据集中随机抽取一批 ts 作为条件，形状 [B, 4, L]（若非时序则返回 None）
    def _prepare_ts_batch(ds, batch_size, device):
        if not config.data.temporal:
            return None
        # 直接索引 Dataset，聚合 ts=[4, L] -> [B, 4, L]
        ts_list = []
        for _ in range(batch_size):
            item = ds[np.random.randint(len(ds))]
            if isinstance(item, dict) and "ts" in item:
                ts_list.append(item["ts"].unsqueeze(0))
        if len(ts_list) == 0:
            return None
        return torch.cat(ts_list, dim=0).to(device)

    for step in range(initial_step, num_train_steps + 1):
        try:
            graphs = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            graphs = next(train_iter)
        batch = {
            "adj": scaler(graphs["adj"]).to(config.device),
            "mask": graphs["mask"].to(config.device),
            "ts": (graphs.get("ts", None).to(config.device) if graphs.get("ts", None) is not None else None),
        }
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            for eval_graphs in eval_loader:
                eval_batch = {
                    "adj": scaler(eval_graphs["adj"]).to(config.device),
                    "mask": eval_graphs["mask"].to(config.device),
                    "ts": (
                        eval_graphs.get("ts", None).to(config.device)
                        if eval_graphs.get("ts", None) is not None
                        else None
                    ),
                }
                eval_loss = eval_step_fn(state, eval_batch)
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
                writer.add_scalar("eval_loss", eval_loss.item(), step)
            for test_graphs in test_loader:
                test_batch = {
                    "adj": scaler(test_graphs["adj"]).to(config.device),
                    "mask": test_graphs["mask"].to(config.device),
                    "ts": (
                        test_graphs.get("ts", None).to(config.device)
                        if test_graphs.get("ts", None) is not None
                        else None
                    ),
                }
                test_loss = eval_step_fn(state, test_batch)
                logging.info("step: %d, test_loss: %.5e" % (step, test_loss.item()))
                writer.add_scalar("test_loss", test_loss.item(), step)

        # Save a checkpoint periodically and generate samples
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:

            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"), state)

            # Generate and save samples
            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                # 准备时序条件（优先用 eval_ds）
                ts_cond = _prepare_ts_batch(
                    eval_ds if config.data.temporal else train_ds,
                    config.training.eval_batch_size,
                    config.device,
                )
                sample, sample_steps, sample_nodes = sampling_fn(score_model, n_node_pmf, ts=ts_cond)
                sample_list = adj2graph(sample, sample_nodes)
                ema.restore(score_model.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                if not os.path.exists(this_sample_dir):
                    os.makedirs(this_sample_dir)
                # graph visualization and save figs
                visualize.visualize_graphs(sample_list, this_sample_dir, config)


def sde_evaluate(config, workdir, eval_folder="eval"):
    """Evaluate trained models.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints.
        eval_folder: The subfolder for storing evaluation results. Default to "eval".
    """

    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build data pipeline
    train_ds, _, test_ds, n_node_pmf = datasets.get_dataset(config)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)

    # Creat data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=config.model.sigma_min,
            sigma_max=config.model.sigma_max,
            N=config.model.num_scales,
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (
            config.eval.batch_size,
            config.data.num_channels,
            config.data.max_node,
            config.data.max_node,
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
        eval_stats_fn = get_stats_eval(config)
        nn_eval_fn = get_nn_eval(config)

    # Begin evaluation
    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))

    # 小工具：准备评估用 ts 条件与其来源索引
    def _prepare_ts_batch(ds, batch_size, device):
        if not config.data.temporal:
            return None, None
        idxs = np.random.randint(len(ds), size=batch_size).tolist()
        ts_list = []
        for i in idxs:
            item = ds[i]
            if isinstance(item, dict) and "ts" in item:
                ts_list.append(item["ts"].unsqueeze(0))
        if len(ts_list) == 0:
            return None, idxs
        return torch.cat(ts_list, dim=0).to(device), idxs

    # 适配器：将 dict 样本按需转换为 PyG Data（只用于 evaluator），并支持子集索引
    class _DictToPyGDataset(torch.utils.data.Dataset):
        def __init__(self, base_ds, indices=None):
            self.base = base_ds
            self.idxs = list(indices) if indices is not None else None

        def __len__(self):
            return len(self.base) if self.idxs is None else len(self.idxs)

        def __getitem__(self, idx):
            if self.idxs is not None:
                idx = self.idxs[idx]
            item = self.base[idx]
            adj = item["adj"]
            if not torch.is_tensor(adj):
                adj = torch.from_numpy(adj)
            A = (adj.squeeze(0) > 0.5).to(torch.bool)  # [N,N]
            N = A.shape[-1]
            A = A.clone()
            A.fill_diagonal_(False)  # 移除自环
            edge_index = A.nonzero(as_tuple=False).t().contiguous()  # [2,E]
            return Data(num_nodes=N, edge_index=edge_index)

    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        # Wait if the target checkpoint doesn't exist yet
        waiting_message_printed = False
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        while not os.path.exists(ckpt_filename):
            if not waiting_message_printed:
                logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
                waiting_message_printed = True
            time.sleep(60)

        # Wait for 2 additional mins in case the file exists but is not ready for reading
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        try:
            state = restore_checkpoint(ckpt_path, state, device=config.device)
        except:
            time.sleep(60)
            try:
                state = restore_checkpoint(ckpt_path, state, device=config.device)
            except:
                time.sleep(120)
                state = restore_checkpoint(ckpt_path, state, device=config.device)

        ema.copy_to(score_model.parameters())

        # Generate samples and compute MMD stats
        if config.eval.enable_sampling:
            num_sampling_rounds = int(np.ceil(config.eval.num_samples / config.eval.batch_size))
            all_samples, eval_indices = [], []
            for r in range(num_sampling_rounds):
                logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
                # 为每个 round 抽一批时序条件，并记录来源索引
                ts_cond, idxs = _prepare_ts_batch(
                    test_ds if config.data.temporal else train_ds,
                    config.eval.batch_size,
                    config.device,
                )
                sample, sample_steps, sample_nodes = sampling_fn(score_model, n_node_pmf, ts=ts_cond)
                logging.info("sample steps: %d" % sample_steps)
                sample_list = adj2graph(sample, sample_nodes)
                all_samples += sample_list
                if idxs is not None:
                    eval_indices += idxs
            # 截断到指定样本数
            all_samples = all_samples[: config.eval.num_samples]
            eval_indices = eval_indices[: len(all_samples)] if len(eval_indices) else eval_indices

            # 基于相同索引构造“真实图子集”作为参考
            if isinstance(test_ds[0], dict):
                ref_ds = _DictToPyGDataset(test_ds, indices=eval_indices if eval_indices else None)
            else:
                try:
                    from torch.utils.data import Subset

                    ref_ds = Subset(test_ds, eval_indices) if eval_indices else test_ds
                except Exception:
                    ref_ds = test_ds  # 兜底

            # save the graphs
            sampler_name = config.sampling.method

            if config.eval.save_graph:
                graph_file = os.path.join(eval_dir, sampler_name + "_ckpt_{}.pkl".format(ckpt))
                with open(graph_file, "wb") as f:
                    pickle.dump(all_samples, f)

            # evaluate（对子集进行分布级评估）
            eval_results = eval_stats_fn(ref_ds, all_samples)
            all_res = []
            for key, values in eval_results.items():
                all_res.append(values)
                logging.info("sampling -- ckpt: {}, {}: {:.6f}".format(ckpt, key, values))
            logging.info("sampling -- ckpt: {}, {}: {:.6f}".format(ckpt, "mean", np.mean(all_res)))
            # Draw and save the graph visualize figs
            this_sample_dir = os.path.join(eval_dir, sampler_name + "_ckpt_{}".format(ckpt))
            if not os.path.exists(this_sample_dir):
                os.makedirs(this_sample_dir)
            visualize.visualize_graphs(all_samples[:32], this_sample_dir, config, remove=False)

            # NN-based metric（同样使用子集）
            nn_eval_results = nn_eval_fn(ref_ds, all_samples)
            for key, values in nn_eval_results.items():
                logging.info(
                    "sampling -- ckpt: {}, {} mean: {:.6f} std: {:.6f}".format(ckpt, key, values[0], values[1])
                )

        # Compute paired edge accuracy for the entire test set
        if isinstance(test_ds[0], dict):
            batch_size = config.eval.batch_size
            all_pred_adjs = []
            all_true_adjs = []
            for start in range(0, len(test_ds), batch_size):
                end = min(start + batch_size, len(test_ds))
                ts_batch = torch.stack([test_ds[i]["ts"] for i in range(start, end)]).to(config.device)
                true_adj_batch = torch.stack([test_ds[i]["adj"] for i in range(start, end)]).to(config.device)
                # Create sampling function for this batch
                sampling_shape = (
                    end - start,
                    config.data.num_channels,
                    config.data.max_node,
                    config.data.max_node,
                )
                sampling_fn_batch = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
                # Sample with paired ts
                pred_adj_batch, _, _ = sampling_fn_batch(score_model, n_node_pmf, ts=ts_batch)
                all_pred_adjs.append(pred_adj_batch)
                all_true_adjs.append(true_adj_batch)
            pred_adj_batch = torch.cat(all_pred_adjs, dim=0)
            true_adj_batch = torch.cat(all_true_adjs, dim=0)
            # Compute edge accuracy
            acc = edge_accuracy(pred_adj_batch, true_adj_batch)
            logging.info("paired_edge_accuracy: {:.6f}".format(acc.item()))


def sde_train_with_decoder(config, workdir):
    """
    两阶段训练管线：
      1) 预训练时序解码器 ψ（可跳过并从 ckpt 加载）
      2) 训练先验 score θ（常规 DSM/EDM），可选加入路线 C 的联合正则
      3) 评估/采样期支持路线 B 的似然引导
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建目录与日志
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # 数据集
    train_ds, eval_ds, test_ds, n_node_pmf = datasets.get_dataset(config)
    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(config.device)
    # 归一化器与其逆（与 sde_train 保持一致）
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # 新增：统一构造 batch 的小工具，假设输入一定包含 adj/mask/ts
    def _make_batch(graphs: dict) -> dict:
        """
        规范化一个 batch，假设 graphs 包含 'adj'、'mask'、'ts' 三个键。
        - 提前做归一化与设备迁移，减少后续分支。
        """
        return {
            "adj": scaler(graphs["adj"]).to(config.device),
            "mask": graphs["mask"].to(config.device),
            "ts": graphs["ts"].to(config.device),
        }

    # 初始化 SDE、模型与优化器
    sde = _build_sde(config)
    sampling_eps = 1e-3
    score_model = mutils.create_model(config).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, score_model.parameters())

    # 1) 解码器：训练或加载
    decoder = None
    ts_feat = config.data.ts_features
    if config.decoder.pretrained:
        decoder_ckpt = config.decoder.ckpt
        if decoder_ckpt and os.path.isfile(decoder_ckpt):
            # 需按保存格式加载
            tmp = torch.load(decoder_ckpt, map_location=device)
            decoder = TemporalDecoder(
                n_in_node=ts_feat,
                msg_hid=config.model.nf,
                msg_out=config.model.nf // 2,
                n_hid=config.model.nf,
                do_prob=0.1,
                sigma_init=0.1,
            ).to(device)
            state_dict = tmp["model_state_dict"] if "model_state_dict" in tmp else tmp
            decoder.load_state_dict(state_dict)
            logging.info(f"Loaded a pretrained decoder from {decoder_ckpt}, ts_feat={ts_feat}.")
        else:
            decoder = train_temporal_decoder(config, workdir, train_ds, eval_ds).to(device)
            logging.info(f"Trained a new decoder from scratch, ts_feat={ts_feat}.")
    else:
        decoder = TemporalDecoder(
            n_in_node=ts_feat,
            msg_hid=config.model.nf,
            msg_out=config.model.nf // 2,
            n_hid=config.model.nf,
            do_prob=0.1,
            sigma_init=0.1,
        ).to(device)
    decoder.eval()  # 训练 score 时固定 ψ

    # 构建 checkpoint 目录，与 evaluate 保持一致
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_meta = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(os.path.dirname(checkpoint_meta), exist_ok=True)

    # 恢复训练状态
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)
    state = restore_checkpoint(checkpoint_meta, state, config.device)
    initial_step = int(state["step"])

    # 2) 训练 score：基础 DSM/EDM + 可选路线 C 联合正则
    # - 与 sde_train 对齐：使用 train_iter + StopIteration，周期评估与抢占式快照
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    loss_fn = losses.get_sde_loss_fn(
        sde, 
        train=True, 
        reduce_mean=reduce_mean, 
        continuous=continuous, 
        likelihood_weighting=likelihood_weighting
    )
    eval_step_fn = losses.get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )
    max_steps = config.training.n_iters
    log_freq = config.training.log_freq
    snapshot_freq = config.training.snapshot_freq
    snapshot_freq_preempt = config.training.snapshot_freq_for_preemption
    eval_freq = config.training.eval_freq

    # 可选：与 sde_train 一致的快照期采样
    if config.training.snapshot_sampling:
        sampling_shape = (
            config.training.eval_batch_size,
            config.data.num_channels,
            config.data.max_node,
            config.data.max_node,
        )
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    train_iter = iter(train_loader)
    score_model.train()
    for step in range(initial_step, max_steps + 1):
        # 取一批数据（与 sde_train 对齐）：仅处理 dict 输入，去除历史兼容逻辑
        try:
            graphs = next(train_iter)
        except StopIteration:
            train_iter = train_loader.__iter__()
            graphs = next(train_iter)

        batch = _make_batch(graphs)

        optimizer.zero_grad(set_to_none=True)
        base_loss = loss_fn(score_model, batch)

        # 路线 C：联合正则（不回传穿过 decoder）
        joint_w = getattr(getattr(config, "training", object()), "posterior_guidance", object())
        joint_w = getattr(joint_w, "joint_weight", 0.0)
        if joint_w > 0.0 and isinstance(batch, dict):
            joint_loss = _compute_joint_regularizer(
                decoder=decoder,
                sde=sde,
                model=score_model,
                batch=batch,
                guidance_weight=joint_w,
                continuous=True,
            )
        else:
            joint_loss = torch.tensor(0.0, device=device)

        total_loss = base_loss + joint_loss
        total_loss.backward()
        if config.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), config.optim.grad_clip)
        optimizer.step()
        ema.update(score_model.parameters())

        # logging
        if step % log_freq == 0:
            writer.add_scalar("train/loss_base", base_loss.item(), step)
            writer.add_scalar("train/loss_joint", joint_loss.item(), step)
            writer.add_scalar("train/loss_total", total_loss.item(), step)
            logging.info("step: %d, loss_base: %.5e, loss_joint: %.5e, loss_total: %.5e" % (step, base_loss.item(), joint_loss.item(), total_loss.item()))

        # 抢占式快照（与 sde_train 对齐）
        if step != 0 and step % snapshot_freq_preempt == 0:
            state["step"] = step
            save_checkpoint(checkpoint_meta, state)

        # 周期评估（与 sde_train 对齐）：仅处理 dict 输入
        if step % eval_freq == 0:
            for eval_graphs in eval_loader:
                eval_batch = _make_batch(eval_graphs)
                eval_loss = eval_step_fn(state, eval_batch)
                logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
                writer.add_scalar("eval_loss", eval_loss.item(), step)
            for test_graphs in test_loader:
                test_batch = _make_batch(test_graphs)
                test_loss = eval_step_fn(state, test_batch)
                logging.info("step: %d, test_loss: %.5e" % (step, test_loss.item()))
                writer.add_scalar("test_loss", test_loss.item(), step)

        # 快照保存与可选采样（与 sde_train 对齐）
        if (step != 0 and step % snapshot_freq == 0) or step == max_steps:
            save_step = step // snapshot_freq if snapshot_freq > 0 else step
            save_checkpoint(
                os.path.join(checkpoint_dir, f"checkpoint_{save_step}.pth"),
                {
                    "optimizer": optimizer,
                    "model": score_model,
                    "ema": ema,
                    "step": step,
                },
            )
            save_checkpoint(
                checkpoint_meta,
                {
                    "optimizer": optimizer,
                    "model": score_model,
                    "ema": ema,
                    "step": step,
                },
            )

            if config.training.snapshot_sampling:
                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                # 采样条件 ts（优先 eval_ds）
                if config.data.temporal:
                    B = config.training.eval_batch_size
                    idxs = np.random.choice(len(eval_ds) if len(eval_ds) > 0 else len(train_ds), B, replace=True)
                    src_ds = eval_ds if len(eval_ds) > 0 else train_ds
                    ts_cond = torch.stack([src_ds[i]["ts"] for i in idxs]).to(config.device)
                else:
                    ts_cond = None
                _, _, _ = sampling_fn(score_model, n_node_pmf, ts=ts_cond)
                ema.restore(score_model.parameters())

    writer.close()
    logging.info("Training finished.")


run_train_dict = {"sde_with_decoder": sde_train_with_decoder, "sde": sde_train}

run_eval_dict = {"sde": sde_evaluate}


def train(config, workdir):
    run_train_dict[config.model_type](config, workdir)


def evaluate(config, workdir, eval_folder="eval"):
    run_eval_dict[config.model_type](config, workdir, eval_folder)
