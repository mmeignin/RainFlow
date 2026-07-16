# RainCFM

Precipitation retrieval via Conditional Flow Matching: a UNet learns to morph noise
into a rain-rate field, conditioned on satellite infrared imagery.

## Idea

The model doesn't predict rain rate directly. Instead it learns a velocity field
that transports a random Gaussian sample `x0` to a real rain field `x1` along a
straight path, over a continuous time `t ∈ [0, 1]`. Training regresses that
velocity in closed form — no simulation, no multi-step denoising needed to get a
training signal.

At inference, the learned velocity is integrated with an ODE solver from `t=0`
(noise) to `t=1` (rain), optionally steered by classifier-free guidance.
Because the model is generative rather than a point estimator,
sampling the same input multiple times gives an ensemble that expresses the
retrieval's uncertainty, instead of one deterministic answer.

- `cfm/` — the flow-matching math (probability path, target velocity, loss).
- `models/` — the UNet that predicts velocity from `(t, noisy field, IR condition)`.
- `utils/` — data loading, normalization, ODE samplers, training/checkpoint plumbing.

## Data

Each training sample is a single `.npy` file holding a dict:

- `rain_rate` — the target field, `[H, W]`.
- one entry per conditioning channel (e.g. an IR/WV satellite band), also `[H, W]`.

A CSV manifest per split ([splits/](../splits/)) lists the absolute path to every
sample for that split. The training config's `channels` field picks which dict
keys are used as conditioning, and in what order.

Before training, both streams are transformed into model space using per-channel
statistics from [normalisation_data/](../normalisation_data/).

## Running it

```bash
conda run -n <venv> python train.py --config configs/FM.yaml
conda run -n <venv> python inference.py --config configs/infer.yaml
```

All settings — architecture, optimization, solver, guidance — live in the two YAML
configs, so a run is fully reproducible from its config file alone. Inference reads
most of its setup (channels, transforms, architecture) straight from the training
checkpoint, so its own config only needs to specify what changes at inference time:
which checkpoint, which ODE solver, how many ensemble members.

Each training run writes checkpoints, preview figures, and a loss log under
`results/<exp_name>/`; each inference run writes predictions and previews under
`results/<exp_name>/inference/`.

## Environment

Run everything through your own conda/virtualenv (Python >= 3.10) — never bare
`python`. Core dependencies:

- `torch` — model, training loop, checkpointing
- `torchdiffeq` — adaptive (`dopri5`) ODE solver used at inference
- `numpy`, `pandas` — data loading and normalization stats
- `pyyaml` — config files
- `matplotlib` — training/inference preview figures
- `tqdm` — progress bars

