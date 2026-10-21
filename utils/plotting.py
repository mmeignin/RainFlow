# TODO(MRMS port): the rain LogNorm limits and the IR colour limits below encode
# Météo-France / SEVIRI physical-unit assumptions and must be revisited for MRMS.
RAIN_VMIN, RAIN_VMAX = 0.12, 50.0   # mm/h, log-scaled
IR_VMIN, IR_VMAX = 200.0, 300.0     # K
PREVIEW_IR_INDEX = 4                # which cond channel to display


def save_preview(rain_gt, rain_pred, cond_raw, path, channel_list=None, max_rows=4, det_rain=None):
    """
    Render up to `max_rows` rows of (IR channel | ground truth rain | [x_det |] prediction).
    All tensors are in raw physical units — the output of `Pipeline.post`.
    `rain_gt` may contain NaN where the radar has no coverage.
    `det_rain` (optional): the deterministic baseline x_det, shown as an extra
    column between GT and prediction.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if cond_raw.shape[1] <= PREVIEW_IR_INDEX:
        raise ValueError(
            f"save_preview needs at least {PREVIEW_IR_INDEX + 1} conditioning channels to "
            f"display PREVIEW_IR_INDEX={PREVIEW_IR_INDEX}, got {cond_raw.shape[1]}."
        )

    n = min(rain_gt.shape[0], max_rows)
    ir_name = channel_list[PREVIEW_IR_INDEX] if channel_list else f"channel {PREVIEW_IR_INDEX}"
    n_cols = 4 if det_rain is not None else 3

    fig, axes = plt.subplots(n, n_cols, figsize=(5 * n_cols, 4 * n))
    if n == 1:
        axes = [axes]
    norm = LogNorm(vmin=RAIN_VMIN, vmax=RAIN_VMAX)

    def _no_rain_masked(img):
        """Below RAIN_VMIN -> NaN, so it renders blank/masked (like GT's real
        no-coverage NaN) instead of a solid near-zero colour block."""
        img = img.clamp(min=0)
        return img.masked_fill(img < RAIN_VMIN, float("nan"))

    for i in range(n):
        panels = [
            (cond_raw[i, PREVIEW_IR_INDEX], ir_name, dict(cmap="inferno_r", vmin=IR_VMIN, vmax=IR_VMAX)),
            (rain_gt[i, 0].clamp(min=0), "GT rain", dict(cmap="jet", norm=norm)),
        ]
        if det_rain is not None:
            panels.append((_no_rain_masked(det_rain[i, 0]), "x_det", dict(cmap="jet", norm=norm)))
        panels.append((_no_rain_masked(rain_pred[i, 0]), "CFM pred", dict(cmap="jet", norm=norm)))

        for ax, (img, title, kw) in zip(axes[i], panels):
            im = ax.pcolormesh(img.detach().cpu(), **kw)
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # pcolormesh's row 0 is at the bottom; imshow's is at the top
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


RES_LIM = 5.0   # symmetric colour limit for the normalized residual panels


def save_preview_residual(cond_raw, rain_gt, x_det_rain, rain_pred, res_pred, res_true,
                          path, channel_list=None, max_rows=4):
    """Residual-mode preview: two rows per sample.

    Row A (rain, shared LogNorm): IR | GT rain | x_det | generated rain.
    Row B (normalized residual, shared ±RES_LIM diverging): pred_r | true_r.

    Rain tensors are raw physical units (`Pipeline.post`); `res_pred`/`res_true` are
    the normalized target `r` the model works in. `rain_gt` may carry NaN where the
    radar has no coverage.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    if cond_raw.shape[1] <= PREVIEW_IR_INDEX:
        raise ValueError(
            f"save_preview_residual needs at least {PREVIEW_IR_INDEX + 1} conditioning "
            f"channels, got {cond_raw.shape[1]}."
        )

    n = min(rain_gt.shape[0], max_rows)
    ir_name = channel_list[PREVIEW_IR_INDEX] if channel_list else f"channel {PREVIEW_IR_INDEX}"
    rain_norm = LogNorm(vmin=RAIN_VMIN, vmax=RAIN_VMAX)
    res_kw = dict(cmap="RdBu_r", vmin=-RES_LIM, vmax=RES_LIM)

    fig, axes = plt.subplots(2 * n, 4, figsize=(20, 5 * n), squeeze=False)
    for i in range(n):
        top = (
            (cond_raw[i, PREVIEW_IR_INDEX], ir_name, dict(cmap="inferno_r", vmin=IR_VMIN, vmax=IR_VMAX)),
            (rain_gt[i, 0].clamp(min=0), "GT rain", dict(cmap="jet", norm=rain_norm)),
            (x_det_rain[i, 0].clamp(min=0), "x_det", dict(cmap="jet", norm=rain_norm)),
            (rain_pred[i, 0].clamp(min=0), "generated rain", dict(cmap="jet", norm=rain_norm)),
        )
        bottom = (
            (res_pred[i, 0], "pred residual", res_kw),
            (res_true[i, 0], "true residual", res_kw),
        )
        for ax, (img, title, kw) in zip(axes[2 * i], top):
            im = ax.pcolormesh(img.detach().cpu(), **kw)
            ax.set_title(title); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
        for j, (img, title, kw) in enumerate(bottom):
            ax = axes[2 * i + 1][j]
            im = ax.pcolormesh(img.detach().cpu(), **kw)
            ax.set_title(title); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
        for j in range(len(bottom), 4):       # blank the unused residual-row cells
            axes[2 * i + 1][j].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
