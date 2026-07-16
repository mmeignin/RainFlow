"""
ODE samplers for the learned velocity field.
Training previews call `euler_sample` with the defaults (`null_cond=None`,
`guidance_scale=1.0`), i.e. unguided. Inference may guide.
"""

import torch


def cfg_forward(model, t, x, cond, null_cond=None, guidance_scale=1.0):
    """Velocity at (t, x, cond), with optional classifier-free guidance."""
    t = t.expand(x.shape[0]) if t.dim() == 0 else t
    if guidance_scale == 1.0 or null_cond is None:
        return model(t, x, cond)
    null = null_cond.expand(x.shape[0], -1, x.shape[2], x.shape[3])
    v_cond = model(t, x, cond)
    v_null = model(t, x, null)
    return v_null + guidance_scale * (v_cond - v_null)


@torch.no_grad()
def euler_sample(model, x0, cond, null_cond=None, guidance_scale=1.0, n_steps=50):
    x = x0.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * cfg_forward(model, t, x, cond, null_cond, guidance_scale)
    return x, n_steps


def dopri5_sample(model, x0, cond, null_cond=None, guidance_scale=1.0, rtol=1e-3, atol=1e-3):
    from torchdiffeq import odeint

    class _ODE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.nfe = 0

        def forward(self, t, x):
            self.nfe += 1
            return cfg_forward(model, t, x, cond, null_cond, guidance_scale)

    func = _ODE()
    t_span = torch.tensor([0.0, 1.0], device=x0.device)
    with torch.no_grad():
        traj = odeint(func, x0, t_span, method="dopri5", rtol=rtol, atol=atol)
    return traj[-1], func.nfe


def sample(model, x0, cond, null_cond, cfg):
    """Dispatch on cfg['solver']. Returns (x1, nfe)."""
    if cfg["solver"] == "euler":
        return euler_sample(model, x0, cond, null_cond,
                            cfg["guidance_scale"], cfg["n_steps"])
    if cfg["solver"] == "dopri5":
        return dopri5_sample(model, x0, cond, null_cond,
                             cfg["guidance_scale"], cfg["rtol"], cfg["atol"])
    raise ValueError(f"unknown solver {cfg['solver']!r}; expected 'euler' or 'dopri5'.")
