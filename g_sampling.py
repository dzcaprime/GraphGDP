"""Various guided sampling methods."""

import functools

import torch

from models import utils as mutils
from sampling import shared_predictor_update_fn, shared_corrector_update_fn
from sampling import ReverseDiffusionPredictor


def likelihood_guided_score(
    A_t, t, sde, model, decoder, mask: torch.Tensor, X_obs: torch.Tensor, guidance_weight: float = 2.0
):
    assert A_t.dim() in (3, 4), f"A_t dim must be 3 or 4, got {A_t.shape}"
    assert mask.dim() in (3, 4), f"mask dim must be 3 or 4, got {mask.shape}"
    assert A_t.shape[-2:] == mask.shape[-2:], f"A_t/mask shape mismatch: {A_t.shape} vs {mask.shape}"

    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    with torch.no_grad():
        score_orig = score_fn(A_t, t, mask=mask)  # [B,1,N,N]
    if guidance_weight <= 0.0 or decoder is None or X_obs is None:
        return score_orig

    # 2) 似然梯度在 3D 空间计算（A、mask 皆为 3D）
    A_3d = A_t.squeeze(1) if A_t.dim() == 4 else A_t  # [B,N,N]
    mask_3d = mask.squeeze(1) if mask.dim() == 4 else mask  # [B,N,N]

    with torch.enable_grad():
        A_3d = A_3d.clone().detach().requires_grad_(True)
        logp = decoder.compute_log_likelihood(A_3d, X_obs)  # X_obs 支持 [B,T,N,F]
        grad = torch.autograd.grad(logp.sum(), A_3d, allow_unused=False)[0]  # [B,N,N]

    # 3) 对称化 / 去对角 / 掩码（使用 3D 掩码）
    grad = 0.5 * (grad + grad.transpose(-1, -2))
    grad = grad - torch.diag_embed(torch.diagonal(grad, dim1=-2, dim2=-1))
    grad = grad * mask_3d  # [B,N,N]

    # 4) sigma_t 广播到 [B,1,1]，在 3D guidance 上施加，然后升到 4D 与 score 相加
    _, sigma_t = sde.marginal_prob(torch.zeros_like(A_3d), t)  # [B]
    sigma_t = sigma_t.view(-1, 1, 1)  # [B,1,1]
    guidance_3d = guidance_weight * sigma_t * grad  # [B,N,N]
    guided_score = score_orig + guidance_3d.unsqueeze(1)  # [B,1,N,N]

    return guided_score


def likelihood_guided_step(
    A_t: torch.Tensor,
    t,
    sde,
    model,
    decoder,
    X_obs: torch.Tensor,
    mask: torch.Tensor,
    guidance_weight: float = 2.0,  # 仅保留：引导强度 λ
    eps_stabilize: float = 1e-8,
    continuous: bool = True,
    guidance_power: float = 2.0,  # 仅保留：时间调度幂 p
    log_stats: bool = False,
):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    with torch.no_grad():
        original_score = score_fn(A_t, t, mask=mask)  # [B,1,N,N]

    if guidance_weight <= 0.0 or decoder is None or X_obs is None:
        return original_score

    # Tweedie 去噪
    mean, std = sde.marginal_prob(torch.zeros_like(A_t), t)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)
    A0_hat = (A_t + (std**2) * original_score) / (mean + eps_stabilize)
    c = mean  # ∂A0/∂At = 1/mean

    # 解码器似然梯度（对投影后的 A 求 ∇A loglik, 内部已固定一次投影）
    A0_in = A0_hat.squeeze(1)
    try:
        with torch.enable_grad():
            grad_A0 = decoder.compute_likelihood_gradient(A0_in, X_obs)  # [B,N,N]
    except Exception as e:
        print(f"guidance grad failed: {e}")
        grad_A0 = torch.zeros_like(A0_in)

    # 结构与掩码
    grad_A0 = 0.5 * (grad_A0 + grad_A0.transpose(-1, -2))
    grad_A0 = grad_A0 - torch.diag_embed(torch.diagonal(grad_A0, dim1=-2, dim2=-1))
    grad_A0 = grad_A0 * mask.squeeze(1)

    # 批次范数（掩码域）
    s_flat = (original_score * mask).reshape(original_score.size(0), -1)
    g_flat = (grad_A0.unsqueeze(1) * mask).reshape(original_score.size(0), -1)
    s_norm = s_flat.norm(dim=1, keepdim=True) + eps_stabilize
    g_norm = g_flat.norm(dim=1, keepdim=True) + eps_stabilize

    # 方向门控（平滑）：负相关步为 0，正相关按 cos 加权
    cos_batch = torch.nn.functional.cosine_similarity(s_flat, g_flat, dim=1, eps=eps_stabilize).view(-1, 1, 1, 1)
    # gate = torch.clamp(cos_batch, min=0.0)
    # gate = torch.sigmoid(10.0 * cos_batch)  # 更平滑
    gate = torch.nn.functional.softplus(10.0 * cos_batch) - 0.5  # 更平滑

    # 时间调度：早强晚弱
    # schedule = ((1.0 - t.view(-1,1,1,1) / sde.T+0.1).clamp(0.0,1.0)) ** guidance_power

    # 时间调度：early steps guidance 弱，late steps guidance 强
    # 方案1：线性递增
    # schedule = (t.view(-1,1,1,1) / sde.T).clamp(0.0,1.0) ** guidance_power

    # 方案2：sigmoid 递增（更平滑）
    # alpha, beta 可调，推荐 alpha=10, beta=0.5
    schedule = torch.sigmoid(10.0 * (t.view(-1, 1, 1, 1) / sde.T - 0.5))
    # 屏蔽 early steps guidance
    mask_late = (t.view(-1, 1, 1, 1) / sde.T < 0.5).float()
    schedule = schedule * mask_late

    # 归一化 + 拉回 At 空间
    K = 10.0
    scale = K * guidance_weight * (s_norm / g_norm).view(-1, 1, 1, 1)  # 自适应到 score 的量纲
    guidance_raw = grad_A0.unsqueeze(1)
    guidance_raw = guidance_raw * gate * schedule * scale
    guidance_raw = guidance_raw * mask

    # 相对范数上限：||guidance|| ≤ 1.0 · ||score||
    g_raw_norm = (guidance_raw * mask).reshape(guidance_raw.size(0), -1).norm(dim=1, keepdim=True) + eps_stabilize
    cap = (s_norm / g_raw_norm).view(-1, 1, 1, 1)
    cap = torch.clamp(cap, max=0.1)
    guidance_term = guidance_raw * cap

    guided_score = original_score + guidance_term
    guided_score = torch.nan_to_num(guided_score, nan=0.0, posinf=0.0, neginf=0.0)

    if log_stats and (t.view(-1, 1, 1, 1) / sde.T >= 0.5).any():
        print(f"grad_A0 mean: {grad_A0.mean().item()}, norm: {grad_A0.norm().item()}")
        print(f"mask mean: {mask.mean().item()}, mask shape: {mask.shape}")
        print(f"guidance_raw mean: {guidance_raw.mean().item()}, norm: {guidance_raw.norm().item()}")

    if log_stats:
        guidance_term_mean = guidance_term.mean().item()
        guidance_term_max = guidance_term.max().item()
        guidance_term_min = guidance_term.min().item()
        cos_mean = torch.nn.functional.cosine_similarity(s_flat, g_flat, dim=1, eps=eps_stabilize).mean().item()
        ratio = (g_raw_norm / s_norm).mean().item()
        print(
            f"[guide] cos={cos_mean:.3f}, ||g_raw||/||s||={ratio:.3f}, λ={guidance_weight}, p={guidance_power}, "
            f"guide_mean={guidance_term_mean:.3e}, max={guidance_term_max:.3e}, min={guidance_term_min:.3e}"
        )

    return guided_score


def get_guided_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
    guidance_weight=1.0,
):
    """Create a guided PC sampler with improved error handling and logging."""
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn, sde=sde, corrector=corrector, continuous=continuous, snr=snr, n_steps=n_steps
    )

    def guided_pc_sampler(model, n_nodes_pmf, decoder, ts=None):
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            probs = (
                n_nodes_pmf
                if isinstance(n_nodes_pmf, torch.Tensor)
                else torch.tensor(n_nodes_pmf, device=device, dtype=torch.float)
            )
            n_nodes = torch.multinomial(probs, shape[0], replacement=True).to(device)
            node_mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                node_mask[i, : n_nodes[i]] = 1.0
            edge_mask = (node_mask[:, None, :] * node_mask[:, :, None]).unsqueeze(1)
            edge_mask = torch.tril(edge_mask, -1)
            edge_mask = edge_mask + edge_mask.transpose(-1, -2)

            x = x * edge_mask
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=device) * t

                # corrector
                guided_score = None
                if decoder is not None and ts is not None:
                    guided_score = likelihood_guided_score(
                        A_t=x,
                        t=vec_t,
                        sde=sde,
                        model=model,
                        decoder=decoder,
                        X_obs=ts,
                        mask=edge_mask,
                        guidance_weight=guidance_weight,
                    )
                # corrector 支持 guided_score
                x, x_mean = corrector_update_fn(x, vec_t, model=model, mask=edge_mask, ts=ts, guided_score=guided_score)
                x = x * edge_mask

                # predictor with guidance
                if guided_score is not None:

                    def temp_score_fn(x_in, t_in, **kwargs):
                        return guided_score

                    if predictor is None:
                        x_mean = x
                    else:
                        predictor_obj = predictor(sde, temp_score_fn, probability_flow)
                        x, x_mean = predictor_obj.update_fn(x, vec_t, mask=edge_mask, ts=ts)
                else:
                    x, x_mean = predictor_update_fn(x, vec_t, model=model, mask=edge_mask, ts=ts)

                x = x * edge_mask

            # Denoising
            if denoise:
                score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
                predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
                vec_eps = torch.ones(shape[0], device=device) * eps
                _, x = predictor_obj.update_fn(x, vec_eps, mask=edge_mask, ts=ts)
                x = x * edge_mask

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return inverse_scaler(x) * edge_mask, sde.N * (n_steps + 1), n_nodes

    return guided_pc_sampler
