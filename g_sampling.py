"""Various guided sampling methods."""

import functools

import torch

from models import utils as mutils

############################################################
# Baseline (A) guidance function (restored original simple version)
############################################################

def likelihood_guided_score_baseline(
    A_t: torch.Tensor,
    t,
    sde,
    model,
    decoder,
    X_obs: torch.Tensor,
    mask: torch.Tensor,
    guidance_weight: float = 1.0,
    eps: float = 1e-8,
    continuous: bool = True,
    project: bool = True,
):
    """Baseline (A) guidance: simple adaptive scaling.

    Core formula (no SNR schedule / no conflict handling):
        A0_hat = (A_t + σ_t^2 * s) / m_t
        g = ∇_{A0} log p(X|A0)
        guidance = λ * (||s||/||g||) * g / m_t
        s_guided = s + guidance

    Parameters mirror the original design; kept minimal for stability.
    """
    if guidance_weight <= 0.0 or decoder is None or X_obs is None:
        score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
        with torch.no_grad():
            return score_fn(A_t, t, mask=mask)

    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    with torch.no_grad():
        score = score_fn(A_t, t, mask=mask)  # [B,1,N,N]

    mean, std = sde.marginal_prob(torch.zeros_like(A_t), t)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)
    A0_hat = (A_t + (std**2) * score) / (mean + eps)
    c = mean  # dA0/dAt = 1/mean

    A0_in = A0_hat.squeeze(1) * mask.squeeze(1)
    if project and hasattr(decoder, "project_adjacency"):
        with torch.no_grad():
            A0_in = decoder.project_adjacency(A0_in)

    # likelihood gradient
    with torch.enable_grad():
        A0_var = A0_in.detach().requires_grad_(True)
        try:
            if hasattr(decoder, "compute_log_likelihood"):
                loglik = decoder.compute_log_likelihood(A0_var, X_obs)
            else:
                # fallback: assume decoder returns (out, loglik)
                _, loglik = decoder(A0_var, X_obs, return_loglik=True)
            grad = torch.autograd.grad(loglik.sum() if loglik.ndim > 1 else loglik, A0_var)[0]
        except Exception:
            grad = torch.zeros_like(A0_var)

    grad = 0.5 * (grad + grad.transpose(-1, -2))
    grad = grad - torch.diag_embed(torch.diagonal(grad, dim1=-2, dim2=-1))
    grad = grad * mask.squeeze(1)

    s_flat = (score * mask).reshape(score.size(0), -1)
    g_flat = (grad.unsqueeze(1) * mask).reshape(score.size(0), -1)
    s_norm = s_flat.norm(dim=1, keepdim=True) + eps
    g_norm = g_flat.norm(dim=1, keepdim=True) + eps
    scale = guidance_weight * (s_norm / g_norm).view(-1, 1, 1, 1)
    guidance = scale * grad.unsqueeze(1) / (c + eps)
    guided = score + guidance * mask
    return torch.nan_to_num(guided, nan=0.0, posinf=0.0, neginf=0.0)


def likelihood_guided_score(
    A_t,
    t,
    sde,
    model,
    decoder,
    mask: torch.Tensor,
    X_obs: torch.Tensor,
    guidance_weight: float = 1.0,
    # advanced params
    strategy: str = "cagrad",  # ["pcgrad", "cagrad"]
    row_normalize: bool = False,
    snr_power: float = 0.5,
    w_cap: float = 1.0,
    eps: float = 1e-12,
    # Ablation toggles (B-F). Enable cumulatively to reproduce experiments:
    exp_B: bool = False,  # switch from baseline to advanced pipeline (conflict + SNR + no projection + sigma scaling)
    exp_C: bool = False,  # disable conflict handling (PCGrad/CAGrad)
    exp_D: bool = False,  # disable SNR schedule (schedule=1)
    exp_E: bool = False,  # restore projection before grad
    exp_F: bool = False,  # replace advanced scaling with baseline adaptive (||s||/||g||) & drop sigma scaling
):
    """Unified guidance with ablation switches A→B→C→D→E→F.

    Modes (cumulative):
      A: all exp_* False -> Calls baseline simple guidance.
      B: exp_B -> advanced (PC/CAGrad + SNR schedule + sigma scaling + no projection + advanced scaling).
      C: exp_B + exp_C -> as B but NO conflict handling (raw grad).
      D: exp_B + exp_C + exp_D -> as C but NO SNR schedule (schedule=1).
      E: + exp_E -> restore projection step before grad.
      F: + exp_F -> use baseline adaptive scaling & remove sigma scaling & schedule (even if earlier on).

    You should enable flags in order (asserted). Higher stage implies earlier ones.
    """
    # If no experiment flags => baseline A
    if not any([exp_B, exp_C, exp_D, exp_E, exp_F]):
        return likelihood_guided_score_baseline(
            A_t, t, sde, model, decoder, X_obs, mask, guidance_weight=guidance_weight
        )

    # Enforce cumulative ordering (soft)
    if exp_C and not exp_B:
        raise ValueError("exp_C requires exp_B=True")
    if exp_D and not exp_C:
        raise ValueError("exp_D requires exp_C=True (thus B as well)")
    if exp_E and not exp_D:
        raise ValueError("exp_E requires exp_D=True (thus B,C)")
    if exp_F and not exp_E:
        raise ValueError("exp_F requires exp_E=True (thus B,C,D)")

    # 记号：
    # - sθ(A_t,t) ≈ ∇_{A_t} log p_t(A_t)（模型的得分）
    # - 似然项 L(A0) = log p(X|A0)，我们先估计 A0（Tweedie 近似），再对 A0 求梯度
    # - 再用链式法则映回 A_t 空间
    assert A_t.dim() in (3, 4)
    assert mask.dim() in (3, 4)
    assert A_t.shape[-2:] == mask.shape[-2:]

    # 0) base score: sθ(A_t,t)
    # 目标是合成 sθ 与 ∇_{A_t} L 的方向，避免冲突并进行强度调度
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)
    with torch.no_grad():
        score_orig = score_fn(A_t, t, mask=mask)  # [B,1,N,N]
    if guidance_weight <= 0.0 or decoder is None or X_obs is None:
        return score_orig

    # 1) Tweedie 近似反演到 A0：
    # 在 VPSDE/VESDE 类模型中，marginal_prob(0,t) 给出 (m_t, σ_t)。
    # 经验式（Tweedie）：A0_hat ≈ (A_t + σ_t^2 sθ(A_t,t)) / m_t
    # 因此 dA0/dA_t = 1/m_t（常数 w.r.t. A_t），便于后续链式法则。
    A_4d = A_t if A_t.dim() == 4 else A_t.unsqueeze(1)  # [B,1,N,N]
    mean, std = sde.marginal_prob(torch.zeros_like(A_4d), t)
    if mean.dim() == 1:
        mean = mean.view(-1, 1, 1, 1)
    if std.dim() == 1:
        std = std.view(-1, 1, 1, 1)

    with torch.no_grad():
        A0_hat = (A_4d + (std ** 2) * score_orig) / (mean + eps)  # [B,1,N,N]
    A0_in = A0_hat.squeeze(1)  # [B,N,N]
    if exp_E and hasattr(decoder, "project_adjacency") and not exp_F:  # projection restored at stage E+ (except F scaling override path keeps it)
        with torch.no_grad():
            A0_in = decoder.project_adjacency(A0_in)

    # 计算 ∇_{A0} log p(X|A0)
    # 注意：对称化与去对角保证图结构约束（无自环）；掩码保证只在有效边上生效。
    with torch.enable_grad():
        A0_in_req = A0_in.clone().detach().requires_grad_(True)
        if hasattr(decoder, "compute_log_likelihood"):
            logp = decoder.compute_log_likelihood(A0_in_req, X_obs, row_normalize=row_normalize)
        else:
            # fallback to forward returning loglik
            _, logp = decoder(A0_in_req, X_obs, return_loglik=True)
        grad_A0 = torch.autograd.grad(logp.sum() if logp.ndim > 1 else logp, A0_in_req, allow_unused=False)[0]

    # 2) 结构规整：symmetrize + zero-diag + mask
    M3 = mask.squeeze(1) if mask.dim() == 4 else mask
    grad_A0 = 0.5 * (grad_A0 + grad_A0.transpose(-1, -2))
    grad_A0 = grad_A0 - torch.diag_embed(torch.diagonal(grad_A0, dim1=-2, dim2=-1))
    grad_A0 = grad_A0 * M3  # [B,N,N]

    # 3) 链式法则映回 A_t 空间：∇_{A_t}L = ∇_{A0}L · (dA0/dA_t) = ∇_{A0}L / m_t
    inv_mean = (1.0 / (mean + eps)).squeeze(1)  # [B,1,1]
    grad_At = grad_A0 * inv_mean  # broadcast over N,N

    # 4) 计算冲突度量与范数
    s3d = score_orig.squeeze(1) * M3
    g3d = grad_At * M3

    dot = (s3d * g3d).sum(dim=(-2, -1), keepdim=True)          # [B,1,1]
    s_norm2 = (s3d * s3d).sum(dim=(-2, -1), keepdim=True) + eps
    g_norm2 = (g3d * g3d).sum(dim=(-2, -1), keepdim=True) + eps

    if not exp_C:  # conflict handling active only in B stage
        strat = strategy.lower()
        if strat == "pcgrad":
            proj = (dot / s_norm2) * s3d
            conflict = (dot < 0.0)
            g3d = torch.where(conflict, g3d - proj, g3d)
        elif strat == "cagrad":
            w_opt = (-dot / g_norm2).clamp(min=0.0, max=w_cap)
            g3d = w_opt * g3d
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # 5) SNR 调度：snr = (m_t/σ_t)^2；schedule = ( snr / (1+snr) )^p ∈ [0,1)
    # 动机：噪声小（后期）时提高观测似然的权重，前期避免过强引导。
    snr = (mean / (std + eps)) ** 2
    if not exp_D and not exp_F:
        schedule = (snr / (1.0 + snr)).pow(snr_power).squeeze(1)
    else:
        schedule = torch.ones_like(snr).squeeze(1)

    # 6) 引导项：Δs = -λ · schedule · σ_t · g
    # - 负号让梯度朝提升 log p(X|A0) 的方向（最大化似然）
    # - 乘 σ_t 以与原有 score 的量纲对齐（经验做法）
    if exp_F:
        # F: baseline adaptive scaling in At space (no sigma scaling, schedule=1 already)
        g_flat = (g3d).reshape(g3d.size(0), -1)
        s_flat = s3d.reshape(s3d.size(0), -1)
        g_norm = g_flat.norm(dim=1, keepdim=True) + eps
        s_norm = s_flat.norm(dim=1, keepdim=True) + eps
        scale = guidance_weight * (s_norm / g_norm)
        guidance_3d = scale.view(-1, 1, 1) * g3d
    else:
        _, sigma_t = sde.marginal_prob(torch.zeros_like(A0_in), t)
        sigma_t = sigma_t.view(-1, 1, 1)
        guidance_3d = guidance_weight * schedule * sigma_t * g3d

    guided_score = score_orig + guidance_3d.unsqueeze(1)

    # 数值稳定：清理 NaN/Inf
    guided_score = torch.nan_to_num(guided_score, nan=0.0, posinf=0.0, neginf=0.0)
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
    import sampling
    from sampling import shared_predictor_update_fn, shared_corrector_update_fn
    from sampling import ReverseDiffusionPredictor

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
