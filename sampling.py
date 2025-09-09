"""Various sampling methods."""

import functools
from re import X

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn

from scipy import integrate
from torchdiffeq import odeint
import sde_lib
from models import utils as mutils

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered predictor with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered corrector with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def build_cfg_score_fn(sde, model, continuous: bool, cfg_scale: float | None):
    """
    构建支持 Classifier-Free Guidance 的 score 函数。
    当 cfg_scale<=0 或无 ts 时退化为普通 score。
    """
    base_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)

    def _fn(x, t, mask=None, ts=None, **kw):
        if cfg_scale is None or cfg_scale <= 0.0 or ts is None:
            return base_fn(x, t, mask=mask, ts=ts, **kw)
        # 双前向（无梯度），共享参数
        s_cond = base_fn(x, t, mask=mask, ts=ts, **kw)
        s_uncond = base_fn(x, t, mask=mask, ts=None, **kw)
        return s_uncond + cfg_scale * (s_cond - s_uncond)

    return _fn


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
        config: A `ml_collections.ConfigDict` object that contains all configuration information.
        sde: A `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers representing the expected shape of a single sample.
        inverse_scaler: The inverse data normalizer function.
        eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
        A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
            rtol=config.sampling.rtol,
            atol=config.sampling.atol,
            device=config.device,
        )
    elif sampler_name.lower() == "diffeq":
        sampling_fn = get_diffeq_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=eps,
            rtol=config.sampling.rtol,
            atol=config.sampling.atol,
            step_size=config.sampling.ode_step,
            method=config.sampling.ode_method,
            device=config.device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
            cfg_scale=getattr(config.sampling, "cfg_scale", None),
        )
    elif sampler_name.lower() == "guided_pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_guided_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=eps,
            device=config.device,
            guidance_weight=config.sampling.guidance_weight,
            cfg_scale=getattr(config.sampling, "cfg_scale", None),
        )
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        if isinstance(sde, tuple):
            self.rsde = (sde[0].reverse(score_fn, probability_flow), sde[1].reverse(score_fn, probability_flow))
        else:
            self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the predictor.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, *args, **kwargs):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state.
            t: A PyTorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        z = torch.tril(z, -1)
        z = z + z.transpose(-1, -2)
        drift, diffusion = self.rsde.sde(x, t, *args, **kwargs)
        drift = torch.tril(drift, -1)
        drift = drift + drift.transpose(-1, -2)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t, *args, **kwargs):
        f, G = self.rsde.discretize(x, t, *args, **kwargs)
        f = torch.tril(f, -1)
        f = f + f.transpose(-1, -2)
        z = torch.randn_like(x)
        z = torch.tril(z, -1)
        z = z + z.transpose(-1, -2)

        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t, *args, **kwargs):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            # Note: it seems that subVPSDE doesn't set alphas
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):

            grad = score_fn(x, t, *args, **kwargs)
            noise = torch.randn_like(x)

            noise = torch.tril(noise, -1)
            noise = noise + noise.transpose(-1, -2)

            mask = kwargs["mask"]

            # mask invalid elements and calculate norm
            mask_tmp = mask.reshape(mask.shape[0], -1)

            grad_norm = torch.norm(mask_tmp * grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(mask_tmp * noise.reshape(noise.shape[0], -1), dim=-1).mean()

            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t, *args, **kwargs):
        return x, x


def shared_predictor_update_fn(
    x, t, sde, model, predictor, probability_flow, continuous, cfg_scale=None, *args, **kwargs
):
    """支持 CFG 的 predictor 包装。"""
    score_fn = build_cfg_score_fn(sde, model, continuous=continuous, cfg_scale=cfg_scale)
    if predictor is None:
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, *args, **kwargs)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps, cfg_scale=None, *args, **kwargs):
    """支持 CFG 的 corrector 包装。"""
    score_fn = build_cfg_score_fn(sde, model, continuous=continuous, cfg_scale=cfg_scale)
    if corrector is None:
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t, *args, **kwargs)


def get_pc_sampler(
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
    cfg_scale: float | None = None,
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        sde: An `sde_lib.SDE` object representing the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
        corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
        inverse_scaler: The inverse data normalizer.
        snr: A `float` number. The signal-to-noise ratio for configuring correctors.
        n_steps: An integer. The number of corrector steps per predictor update.
        probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
        continuous: `True` indicates that the score model was continuously trained.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
        cfg_scale=cfg_scale,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
        cfg_scale=cfg_scale,
    )

    def pc_sampler(model, n_nodes_pmf):
        """The PC sampler function.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.

        Returns:
            Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][: n_nodes[i]] = 1.0
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)
            mask = torch.tril(mask, -1)
            mask = mask + mask.transpose(-1, -2)

            x = x * mask

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model, mask=mask)
                x = x * mask
                x, x_mean = predictor_update_fn(x, vec_t, model=model, mask=mask)
                x = x * mask

            return inverse_scaler(x_mean if denoise else x) * mask, sde.N * (n_steps + 1), n_nodes

    return pc_sampler


def get_ode_sampler(
    sde, shape, inverse_scaler, denoise=False, rtol=1e-5, atol=1e-5, method="RK45", eps=1e-3, device="cuda"
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, mask):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, mask=mask)
        return x

    def drift_fn(model, x, t, mask):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, mask=mask)[0]

    def ode_sampler(model, n_nodes_pmf, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][: n_nodes[i]] = 1.0
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t, mask)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x), rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, mask)

            x = inverse_scaler(x) * mask
            return x, nfe, n_nodes

    return ode_sampler


def get_diffeq_sampler(
    sde,
    shape,
    inverse_scaler,
    denoise=False,
    rtol=1e-5,
    atol=1e-5,
    step_size=0.01,
    method="dopri5",
    eps=1e-3,
    device="cuda",
):
    """
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        shape: A sequence of integers. The expected shape of a single sample.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver in torchdiffeq.
            See the documentation of `torchdiffeq`. eg: adaptive solver('dopri5', 'bosh3', 'fehlberg2')
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x, mask):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps, mask=mask)
        return x

    def drift_fn(model, x, t, mask):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t, mask=mask)[0]

    def diffeq_sampler(model, n_nodes_pmf, z=None):
        """The probability flow ODE sampler with ODE solver from torchdiffeq.

        Args:
            model: A score model.
            n_nodes_pmf: Probability mass function of graph nodes.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distribution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            # Sample the number of nodes
            n_nodes = torch.multinomial(n_nodes_pmf, shape[0], replacement=True)
            mask = torch.zeros((shape[0], shape[-1]), device=device)
            for i in range(shape[0]):
                mask[i][: n_nodes[i]] = 1.0
            mask = (mask[:, None, :] * mask[:, :, None]).unsqueeze(1)

            class ODEfunc(torch.nn.Module):
                def __init__(self):
                    super(ODEfunc, self).__init__()
                    self.nfe = 0

                def forward(self, t, x):
                    self.nfe += 1
                    x = x.reshape(shape)
                    vec_t = torch.ones(shape[0], device=x.device) * t
                    drift = drift_fn(model, x, vec_t, mask)
                    return drift.reshape((-1,))

            # Black-box ODE solver for the probability flow ODE
            ode_func = ODEfunc()
            if method in ["dopri5", "bosh3", "fehlberg2"]:
                solution = odeint(
                    ode_func,
                    x.reshape((-1,)),
                    torch.tensor([sde.T, eps], device=x.device),
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    options={"step_t": torch.tensor([1e-3], device=x.device)},
                )
            elif method in ["euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"]:
                solution = odeint(
                    ode_func,
                    x.reshape((-1,)),
                    torch.tensor([sde.T, eps], device=x.device),
                    rtol=rtol,
                    atol=atol,
                    method=method,
                    options={"step_size": step_size},
                )

            x = solution[-1, :].reshape(shape)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x, mask)

            x = inverse_scaler(x) * mask
            return x, ode_func.nfe, n_nodes

    return diffeq_sampler


def likelihood_guided_step(
    A_t: torch.Tensor,
    t,
    sde,
    model,
    decoder,
    X_obs: torch.Tensor,
    mask: torch.Tensor,
    guidance_weight: float = 1.0,
    eps_stabilize: float = 1e-8,
    continuous: bool = True,
    cfg_scale: float | None = None,
):
    """
    单步似然引导采样（在 no_grad 外围下，局部开启梯度用于 decoder）。
    """
    # check data shape
    if X_obs.shape[2] != A_t.size(-1) and X_obs.shape[1] != A_t.size(-1):
        X_obs = X_obs.permute(0, 2, 1, 3).contiguous()  # [B, N, T, C]-->[B, T, N, C]

    # 1) 原始 score（冻结模型梯度）
    score_fn_base = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    with torch.no_grad():
        if cfg_scale is not None and cfg_scale > 0.0 and X_obs is not None:
            s_cond = score_fn_base(A_t, t, mask=mask, ts=X_obs)
            s_uncond = score_fn_base(A_t, t, mask=mask, ts=None)
            original_score = s_uncond + cfg_scale * (s_cond - s_uncond)
        else:
            original_score = score_fn_base(A_t, t, mask=mask, ts=X_obs)  # [B, C, N, N]

    # 2) Tweedie 去噪：A0_hat = (A_t + std^2 * score) / mean
    if hasattr(sde, "marginal_prob"):
        mean, std = sde.marginal_prob(torch.zeros_like(A_t), t)  # [B] 或 [B,1,1,1]
        if mean.dim() == 1:
            mean = mean.view(-1, 1, 1, 1)
        if std.dim() == 1:
            std = std.view(-1, 1, 1, 1)
        A_0_hat = (A_t + (std**2) * original_score) / (mean + eps_stabilize)
        c = mean  # ∂A0/∂At = 1/mean
    else:
        A_0_hat = A_t
        c = torch.ones_like(A_t[:, :1, :1, :1])

    # 3) 投影（不回传投影），构造可导副本
    A_0_clean = A_0_hat.squeeze(1) * mask.squeeze(1)  # [B,N,N]
    decoder.eval()
    with torch.no_grad():
        A_0_clean = decoder.project_adjacency(A_0_clean)
    # 4) 似然梯度 g = ∇_{A0} log pψ(X|A0)，在局部开启梯度计算
    with torch.enable_grad():
        A_0_var = A_0_clean.clone().detach().requires_grad_(True)
        try:
            _, loglik = decoder(A_0_var, X_obs, return_loglik=True)
            if loglik is None or (hasattr(loglik, "requires_grad") and not loglik.requires_grad):
                grad_A0 = torch.zeros_like(A_0_var)
            else:
                grad_A0 = torch.autograd.grad(outputs=loglik, inputs=A_0_var, retain_graph=False, create_graph=False)[0]
        except Exception as e:
            print(f"Gradient computation failed: {e}")
            grad_A0 = torch.zeros_like(A_0_var)

    # 结构一致性：对称、清零对角、乘 mask
    grad_A0 = 0.5 * (grad_A0 + grad_A0.transpose(-1, -2))
    grad_A0 = grad_A0 - torch.diag_embed(torch.diagonal(grad_A0, dim1=-2, dim2=-1))
    grad_A0 = grad_A0 * mask.squeeze(1)

    # 5) 按掩码域自适应归一化，避免无效区域影响尺度
    grad_flat = grad_A0.reshape(grad_A0.size(0), -1)
    score_flat = (original_score * mask).reshape(original_score.size(0), -1)
    grad_norm = grad_flat.norm(dim=1, keepdim=True)  # [B,1]
    score_norm = score_flat.norm(dim=1, keepdim=True)  # [B,1]
    # 分子/分母都加入 eps，避免 0/0 或除零导致 NaN/Inf
    adaptive = (guidance_weight * (score_norm + eps_stabilize) / (grad_norm + eps_stabilize)).view(-1, 1, 1, 1)

    # 6) 映射回 score 空间：∇_{At} = (1/c) ∇_{A0}
    guidance_term = adaptive * grad_A0.unsqueeze(1) / (c + eps_stabilize)  # [B,1,N,N]
    guidance_term = guidance_term * mask  # 保持结构
    guided_score = original_score + guidance_term
    # 输出前进行数值清理，避免把 NaN/Inf 传回外层积分器
    guided_score = torch.nan_to_num(guided_score, nan=0.0, posinf=0.0, neginf=0.0)

    # 计算 ratio 供调试（不改变返回接口）
    with torch.no_grad():
        num = guidance_term.reshape(guidance_term.size(0), -1).norm(dim=1) + eps_stabilize
        den = original_score.reshape(original_score.size(0), -1).norm(dim=1) + eps_stabilize
        ratio = (num / den).mean()  # 可在外部按需打印

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
    cfg_scale: float | None = None,
):
    """Create a guided PC sampler with improved error handling."""
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
        cfg_scale=cfg_scale,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
        cfg_scale=cfg_scale,
    )

    def guided_pc_sampler(model, n_nodes_pmf, decoder, ts=None):
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)

            # 节点数采样
            probs = (
                n_nodes_pmf
                if isinstance(n_nodes_pmf, torch.Tensor)
                else torch.tensor(n_nodes_pmf, device=device, dtype=torch.float)
            )
            n_nodes = torch.multinomial(probs, shape[0], replacement=True).to(device)

            # 结构化 edge mask：节点外积 -> 下三角 -> 对称（清零对角）
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

                # 先 corrector，再 predictor（与基线对齐）
                x, x_mean = corrector_update_fn(x, vec_t, model=model, mask=edge_mask, ts=ts)
                x = x * edge_mask

                if decoder is not None and ts is not None:
                    guided_score = likelihood_guided_step(
                        A_t=x,
                        t=vec_t,
                        sde=sde,
                        model=model,
                        decoder=decoder,
                        X_obs=ts,
                        mask=edge_mask,
                        guidance_weight=guidance_weight,
                        continuous=continuous,
                        cfg_scale=cfg_scale,
                    )

                    def temp_score_fn(x_in, t_in, **_):
                        return guided_score

                    if predictor is None:
                        x_mean = x
                    else:
                        predictor_obj = predictor(sde, temp_score_fn, probability_flow)
                        x, x_mean = predictor_obj.update_fn(x, vec_t, mask=edge_mask, ts=ts)
                else:
                    x, x_mean = predictor_update_fn(x, vec_t, model=model, mask=edge_mask, ts=ts)
                x = x * edge_mask
            # ...existing code (denoise + 清理 + 返回)...
            if denoise:
                score_fn = get_score_fn(sde, model, train=False, continuous=True)
                predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
                vec_eps = torch.ones(shape[0], device=device) * eps
                _, x = predictor_obj.update_fn(x, vec_eps, mask=edge_mask, ts=ts)
                x = x * edge_mask

            # 返回前进行一次防御性清理，避免 NaN/Inf 传播（不改变范围/分布）
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return inverse_scaler(x) * edge_mask, sde.N * (n_steps + 1), n_nodes

    return guided_pc_sampler
