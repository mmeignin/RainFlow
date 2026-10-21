# Author: Alex Tong
#         Kilian Fatras
#         +++
# Copyright (c) 2023 Alexander Tong
# License: MIT License
# from https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/conditional_flow_matching.py


import math
from typing import Union

import torch



def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))


class ConditionalFlowMatcher:
    """Base class for conditional flow matching methods. This class implements the independent
    conditional flow matching methods from [1] and serves as a parent class for all other flow
    matching methods.

    It implements:
    - Drawing data from gaussian probability path N(t * x1 + (1 - t) * x0, sigma) function
    - conditional flow matching ut(x1|x0) = x1 - x0
    - score function $\nabla log p_t(x|x0, x1)$
    """

    def __init__(self, sigma: Union[float, int] = 0.0):
        r"""Initialize the ConditionalFlowMatcher class.

        It requires the hyper-parameter $\sigma$.
                Parameters
                ----------
                sigma : Union[float, int]
        """
        self.sigma = sigma

    def compute_mu_t(self, x0, x1, t):
        """
        Compute the mean of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)

        Returns
        -------
        mean mu_t: t * x1 + (1 - t) * x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0

    def compute_sigma_t(self, t):
        """
        Compute the standard deviation of the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        standard deviation sigma

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t
        return self.sigma

    def sample_xt(self, x0, x1, t, epsilon):
        """
        Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        epsilon : Tensor, shape (bs, *dim)
            noise sample from N(0, 1)

        Returns
        -------
        xt : Tensor, shape (bs, *dim)

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x0)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, x0, x1, t, xt):
        """
        Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt

        Returns
        -------
        ut : conditional vector field ut(x1|x0) = x1 - x0

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        del t, xt
        return x1 - x0

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        """
        Compute the sample xt (drawn from N(t * x1 + (1 - t) * x0, sigma))
        and the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        (optionally) t : Tensor, shape (bs)
            represents the time levels
            if None, drawn from uniform [0,1]
        return_noise : bool
            return the noise sample epsilon


        Returns
        -------
        t : FloatTensor, shape (bs)
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn from probability path pt
        ut : conditional vector field ut(x1|x0) = x1 - x0
        (optionally) eps: Tensor, shape (bs, *dim) such that xt = mu_t + sigma_t * epsilon

        References
        ----------
        [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
        """
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

    def compute_lambda(self, t):
        """Compute the lambda function, see Eq.(23) [3].

        Parameters
        ----------
        t : FloatTensor, shape (bs)

        Returns
        -------
        lambda : score weighting function

        References
        ----------
        [4] Simulation-free Schrodinger bridges via score and flow matching, Preprint, Tong et al.
        """
        sigma_t = self.compute_sigma_t(t)
        return 2 * sigma_t / (self.sigma**2 + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Source prior and training-timestep samplers (config-agnostic; callers pass args)
# ──────────────────────────────────────────────────────────────────────────────

def sample_prior(shape, prior="gaussian", nu=None, device=None):
    """Draw the flow-matching source x0.

    `gaussian`   : x0 ~ N(0, I). `nu` is unused.
    `student_t`  : unit-variance multivariate Student-t (Pandey et al., ICLR 2025,
                   t-Flow). One chi-square scalar per sample scales the whole field:
                       x0 = z * sqrt((nu - 2) / v),  z ~ N(0, I),  v ~ chi2(nu).
                   Heavier tails than the Gaussian, SAME variance (the (nu-2)
                   factor normalises Var to 1), so the path/objective are otherwise
                   identical. `v` is one scalar per sample -> a true multivariate-t.
                   `nu` defaults to 5.0 (the paper's choice) when unspecified.
    """
    z = torch.randn(shape, device=device)
    if prior == "gaussian":
        return z
    if prior == "student_t":
        nu = 5.0 if nu is None else nu
        if nu <= 2:
            raise ValueError(f"student_t nu must be > 2 for finite variance, got {nu}.")
        v = torch.distributions.Chi2(float(nu)).sample((shape[0],)).to(z.device)
        scale = torch.sqrt((nu - 2) / v).view(-1, *([1] * (z.dim() - 1)))
        return z * scale
    raise ValueError(f"unknown prior {prior!r}; expected 'gaussian' or 'student_t'.")


def sample_timesteps(n, kind="uniform", std=None, k=None, mix_prob=None, device=None):
    """Draw training timesteps in [0, 1]. Convention: t=0 is noise, t=1 is data.

    `uniform`      : t ~ U[0, 1]. `std`/`k`/`mix_prob` unused.
    `logit_normal` : t = sigmoid(std * z), z ~ N(0, 1). Concentrates samples on
                     mid-path timesteps (the SD3 / EDM weighting). Density is zero
                     at both t = 0 and t = 1. `std` defaults to 1.0.
    `power`        : t = u^k, u ~ U[0, 1], with density p(t) = (1/k) t^(1/k - 1).
                     k = 1 is uniform; k > 1 shifts mass toward t = 0, where xt
                     carries no signal and the velocity can only be predicted from
                     the conditioning. The density diverges at t = 0, which
                     logit_normal cannot express. `k` defaults to 1.5.
    `logit_normal_power_mix` : per-sample mixture of the two above: with
                     probability `mix_prob` (default 0.2) draw from `power(k)`,
                     otherwise from `logit_normal(std)`. Adds mass near t = 0 to
                     logit_normal while keeping its mid-path concentration (cf. Lin
                     et al. 2024, "Common Diffusion Noise Schedules and Sample
                     Steps are Flawed"). `k` defaults to 3.0 here.

    Inference is unaffected (it integrates the whole ODE); this only reweights
    which t the training loss sees.
    """
    if kind == "uniform":
        return torch.rand(n, device=device)
    if kind == "logit_normal":
        std = 1.0 if std is None else std
        return torch.sigmoid(std * torch.randn(n, device=device))
    if kind == "power":
        k = 1.5 if k is None else k
        if k <= 0:
            raise ValueError(f"time_sampling_k must be > 0, got {k}.")
        return torch.rand(n, device=device) ** k
    if kind == "logit_normal_power_mix":
        mix_prob = 0.2 if mix_prob is None else mix_prob
        if not 0.0 <= mix_prob <= 1.0:
            raise ValueError(f"time_sampling_mix_prob must be in [0, 1], got {mix_prob}.")
        mix_k = 3.0 if k is None else k
        from_power = torch.rand(n, device=device) < mix_prob
        t_power = sample_timesteps(n, kind="power", k=mix_k, device=device)
        t_logit_normal = sample_timesteps(n, kind="logit_normal", std=std, device=device)
        return torch.where(from_power, t_power, t_logit_normal)
    raise ValueError(
        f"unknown time_sampling {kind!r}; expected 'uniform', 'logit_normal', 'power' "
        f"or 'logit_normal_power_mix'."
    )


if __name__ == "__main__":
    cfm = ConditionalFlowMatcher(sigma=0.0)  # sigma must be float or int
    x0 = torch.randn(4, 3)  # batch of 4, dim 3
    x1 = torch.randn(4, 3)
    t, xt, ut = cfm.sample_location_and_conditional_flow(x0, x1)
    print("t:", t)
    print("xt:", xt)
    print("ut:", ut)
