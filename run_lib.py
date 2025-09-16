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
import networkx as nx

if not hasattr(nx, "from_numpy_matrix"):
    nx.from_numpy_matrix = nx.from_numpy_array

import losses
import sampling
from models import utils as mutils
from models.temporal_decoder import TemporalDecoder  # 新增：导入时序解码器
from models.RNNdecoder import RNNDecoder  # 新增：导入RNN解码器
from models.ema import ExponentialMovingAverage
import datasets
from run_lib_backup import sde_train, sde_evaluate
import sde_lib
from utils import *
from train_dec import train_temporal_decoder  # 新增：导入解码器训练函数
from evaluation.evaluations import compute_edge_metrics


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


def _build_decoder(config, device, workdir, train_ds, eval_ds, mode: str):
    decoder = None
    # 仅允许使用已训练好的 decoder
    ts_feat = config.data.ts_features
    if config.decoder.pretrained:
        decoder_ckpt = config.decoder.ckpt
        if decoder_ckpt and os.path.isfile(decoder_ckpt):
            tmp = torch.load(decoder_ckpt, map_location=device)
            decoder = RNNDecoder(
                n_in_node=ts_feat,
                msg_hid=config.decoder.msg_hidden,
                msg_out=config.model.nf // 2,
                n_hid=config.decoder.n_hidden,
                do_prob=0.1,
                sigma_init=0.1,
            ).to(device)
            state_dict = tmp.get("model_state_dict", tmp)
            decoder.load_state_dict(state_dict)
        else:
            if mode == "train":
                decoder = train_temporal_decoder(config, workdir, train_ds, eval_ds).to(device)
                logging.info(f"Trained a new decoder from scratch, ts_feat={ts_feat}.")
                return decoder
            elif mode == "eval":
                logging.error("Pretrained decoder checkpoint not found.")
            else:
                logging.error("Invalid mode, should be train or eval.")
            return
    else:
        logging.error("Evaluation requires a pretrained decoder, but none is provided.")
        return
    if decoder is None:
        raise RuntimeError(
            "Temporal decoder is required but not available. Please set "
            "config.decoder.pretrained=True and provide a valid "
            "config.decoder.ckpt for mode='%s'." % mode
        )
    return decoder


DEFAULT_SAMPLING_EPS: float = 1e-3


def _get_sampling_eps(config, default: float = DEFAULT_SAMPLING_EPS) -> float:
    """
    读取采样噪声步长 eps：
    - 优先从 config.sampling.eps 获取；
    - 否则退回默认值，不改变既有行为。
    """
    try:
        return float(getattr(getattr(config, "sampling", object()), "eps", default))
    except Exception:
        return default


def make_sampling_fn(
    config,
    sde,
    inverse_scaler,
    batch_size: int,
    eps: Optional[float] = None,
):
    """
    构造并返回固定 batch 形状的 sampling_fn。

    参数
    ----
    config : Any
        全局配置对象，需要 data.num_channels 与 data.max_node。
    sde : Any
        SDE 实例。
    inverse_scaler : Callable
        数据反归一化函数。
    batch_size : int
        采样时的 batch 大小。
    eps : float, optional
        采样 eps；若为 None 则从 config.sampling.eps 或默认值获取。

    返回
    ----
    Callable
        调用形式与原 sampling.get_sampling_fn 返回值一致。
    """
    eff_eps = _get_sampling_eps(config) if eps is None else eps
    sampling_shape = (
        batch_size,
        config.data.num_channels,
        config.data.max_node,
        config.data.max_node,
    )
    return sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, eff_eps)


def wait_and_restore(
    ckpt_filename: str,
    state: dict,
    device,
    *,
    warn_once: bool = True,
    retry_delays: tuple[int, ...] = (60, 60, 120),
) -> dict:
    """
    等待 checkpoint 文件出现并带退避重试恢复状态。
    - 等待阶段仅打印一次告警（可配置）。
    - 恢复阶段失败将按 retry_delays 顺序延时重试。

    返回
    ----
    dict
        成功恢复后的 state。
    """
    waited = False
    while not os.path.exists(ckpt_filename):
        if warn_once and not waited:
            logging.warning("Waiting for the arrival of %s", os.path.basename(ckpt_filename))
            waited = True
        time.sleep(retry_delays[0] if retry_delays else 60)

    # 恢复并按需重试
    try:
        return restore_checkpoint(ckpt_filename, state, device=device)
    except Exception:
        for delay in retry_delays:
            time.sleep(delay)
            try:
                return restore_checkpoint(ckpt_filename, state, device=device)
            except Exception:
                continue
        # 最后一试：若仍失败，抛出原始异常以便定位
        return restore_checkpoint(ckpt_filename, state, device=device)


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


def sde_train_with_decoder(config, workdir):
    """
    两阶段训练管线：
      1) 预训练时序解码器 ψ（可跳过并从 ckpt 加载）
      2) 训练先验 score θ（常规 DSM/EDM），可选加入路线 C 的联合正则
      3) 评估/采样期支持路线 B 的似然引导
    """
    device = config.device
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
    # 归一化器与其逆
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

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

    # 1) 解码器：训练或加载（早失败保证 decoder 有效）
    decoder = _build_decoder(config, device, workdir, train_ds, eval_ds, mode="train")
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
        sde, train=True, reduce_mean=reduce_mean, continuous=continuous, likelihood_weighting=likelihood_weighting
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

    # 可选：与 sde_train 一致的快照期采样（复用 make_sampling_fn）
    if config.training.snapshot_sampling:
        sampling_fn = make_sampling_fn(
            config,
            sde,
            inverse_scaler,
            batch_size=config.training.eval_batch_size,
        )

    train_iter = iter(train_loader)
    score_model.train()
    for step in range(initial_step, max_steps + 1):
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
            logging.info(
                "step: %d, loss_base: %.5e, loss_joint: %.5e, loss_total: %.5e"
                % (step, base_loss.item(), joint_loss.item(), total_loss.item())
            )

        # 抢占式快照（与 sde_train 对齐）
        if step != 0 and step % snapshot_freq_preempt == 0:
            state["step"] = step
            save_checkpoint(checkpoint_meta, state)

        # 周期评估：仅处理 dict 输入
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

        # 快照保存与可选采样
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
                if config.data.temporal:
                    B = config.training.eval_batch_size
                    # 随机索引一批测试样本，配对构造条件与真值
                    idxs = np.random.randint(len(eval_ds), size=B).tolist()
                    ts_batch = torch.stack([eval_ds[i]["ts"] for i in idxs]).to(config.device)
                    true_adj_batch = torch.stack([eval_ds[i]["adj"] for i in idxs]).to(config.device)
                    mask_batch = torch.stack([eval_ds[i]["mask"] for i in idxs]).to(config.device)
                else:
                    ts_batch = None
                # 采样
                pred_adj_batch, _, _ = sampling_fn(score_model, n_node_pmf, decoder, ts=ts_batch)
                batch_results = compute_edge_metrics(pred_adj_batch, true_adj_batch, mask_batch)
                logging.info(
                    "step: %d, sample_eval_auroc: %.6f, sample_eval_aupr: %.6f, "
                    "sample_eval_f1_best: %.6f, sample_eval_best_tau: %.6f",
                    step,
                    batch_results["auroc"],
                    batch_results["aupr"],
                    batch_results["f1_best"],
                    batch_results["best_threshold"],
                )
                ema.restore(score_model.parameters())

    writer.close()
    logging.info("Training finished.")


def sde_eval_with_decoder(config, workdir, eval_folder="eval"):
    """
    基于解码器的评估流程：
    - 加载 TemporalDecoder
    - 使用 guided_pc_sampler 进行配对时序条件采样
    - 按 num_sampling_rounds 生成样本，计算邻接矩阵 AUROC 并做宏平均
    """
    device = config.device
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # 数据集与归一化
    train_ds, eval_ds, test_ds, n_node_pmf = datasets.get_dataset(config)
    n_node_pmf = torch.from_numpy(n_node_pmf).to(device)
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # 模型与状态
    score_model = mutils.create_model(config).to(device)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")

    # SDE
    sde = _build_sde(config)

    # 仅允许使用已训练好的 decoder（早失败）
    decoder = _build_decoder(config, device, workdir, train_ds, eval_ds, mode="eval")
    decoder.eval()

    # 固定 batch 形状的采样函数（复用 make_sampling_fn，统一 eps）
    sampling_fn = make_sampling_fn(config, sde, inverse_scaler, batch_size=config.eval.batch_size)

    begin_ckpt = config.eval.begin_ckpt
    logging.info("begin checkpoint: %d" % (begin_ckpt,))

    # 遍历评估区间内的所有 checkpoint
    for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):
        ckpt_filename = os.path.join(checkpoint_dir, f"checkpoint_{ckpt}.pth")
        # 等待并恢复该 checkpoint（文件可能尚未写完或未到位，带退避重试）
        state = wait_and_restore(ckpt_filename, state, device=config.device)

        # 将 EMA 权重拷贝到当前模型参数，用于更稳定的推理
        ema.copy_to(score_model.parameters())

        # 采用“总样本数 num_samples，按 batch B 分多轮采样”的策略进行宏平均
        B = config.eval.batch_size
        num_samples = int(config.eval.num_samples)
        num_rounds = int(np.ceil(num_samples / B))

        # 累积各指标的加权和（按本轮实际使用样本数 use 加权），最后再除以总计数 counted 得到宏平均
        sum_auroc, sum_aupr, sum_f1, sum_tau = 0.0, 0.0, 0.0, 0.0
        counted = 0

        for r in range(num_rounds):
            # 随机索引一批测试样本，构造条件（ts）与真值（adj/mask）
            idxs = np.random.randint(len(test_ds), size=B).tolist()
            ts_batch = torch.stack([test_ds[i]["ts"] for i in idxs]).to(config.device) if config.data.temporal else None
            true_adj_batch = torch.stack([test_ds[i]["adj"] for i in idxs]).to(config.device)
            mask_batch = torch.stack([test_ds[i]["mask"] for i in idxs]).to(config.device)

            # 使用解码器条件（ts）进行采样，得到预测的邻接矩阵
            pred_adj_batch, _, _ = sampling_fn(score_model, n_node_pmf, decoder, ts=ts_batch)

            # 控制本轮参与评估的样本数量，避免超过总需求 num_samples
            remain = num_samples - counted
            use = min(remain, B)

            # 计算本轮的边级别指标（AUC-ROC、AUC-PR、F1@bestτ），先按 use 截断再评估
            results = compute_edge_metrics(
                preds=pred_adj_batch[:use],
                target=true_adj_batch[:use],
                mask=mask_batch[:use],
            )

            # 兼容 tensor/float，提取标量
            auroc = float(results["auroc"]) if hasattr(results["auroc"], "item") else results["auroc"]
            aupr = float(results["aupr"]) if hasattr(results["aupr"], "item") else results["aupr"]
            f1b = float(results["f1_best"]) if hasattr(results["f1_best"], "item") else results["f1_best"]
            tau = (
                float(results["best_threshold"])
                if hasattr(results["best_threshold"], "item")
                else results["best_threshold"]
            )

            # 加权累积（按 use 权重），为最终宏平均做准备
            sum_auroc += auroc * use
            sum_aupr += aupr * use
            sum_f1 += f1b * use
            sum_tau += tau * use
            counted += use

            # 若已满足目标样本量则提前结束
            if counted >= num_samples:
                break

        # 计算最终的宏平均指标（避免除零）
        final_auroc = sum_auroc / max(counted, 1)
        final_aupr = sum_aupr / max(counted, 1)
        final_f1 = sum_f1 / max(counted, 1)
        final_tau = sum_tau / max(counted, 1)

        # 记录当前 checkpoint 的评估结果
        logging.info(
            "ckpt: %d, mean_auroc: %.6f, mean_aupr: %.6f, "
            "mean_f1_best: %.6f, mean_best_tau: %.6f",
            ckpt,
            final_auroc,
            final_aupr,
            final_f1,
            final_tau,
        )


run_train_dict = {"sde_with_decoder": sde_train_with_decoder, "sde": sde_train}

run_eval_dict = {"sde": sde_evaluate, "sde_with_decoder": sde_eval_with_decoder}


def train(config, workdir):
    run_train_dict[config.model_type](config, workdir)


def evaluate(config, workdir, eval_folder="eval"):
    run_eval_dict[config.model_type](config, workdir, eval_folder)
