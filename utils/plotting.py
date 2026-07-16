# TODO(MRMS port): the rain LogNorm limits and the IR colour limits below encode
# Météo-France / SEVIRI physical-unit assumptions and must be revisited for MRMS.
RAIN_VMIN, RAIN_VMAX = 0.12, 50.0   # mm/h, log-scaled
IR_VMIN, IR_VMAX = 200.0, 300.0     # K
PREVIEW_IR_INDEX = 4                # which cond channel to display


def save_preview(rain_gt, rain_pred, cond_raw, path, channel_list=None, max_rows=4):
    """Render up to `max_rows` rows of (IR channel | ground truth rain | prediction).

    All three tensors are in raw physical units — the output of `Pipeline.post`.
    `rain_gt` may contain NaN where the radar has no coverage/
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

    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = [axes]
    norm = LogNorm(vmin=RAIN_VMIN, vmax=RAIN_VMAX)

    for i in range(n):
        panels = (
            (cond_raw[i, PREVIEW_IR_INDEX], ir_name, dict(cmap="inferno_r", vmin=IR_VMIN, vmax=IR_VMAX)),
            (rain_gt[i, 0].clamp(min=0), "GT rain", dict(cmap="jet", norm=norm)),
            (rain_pred[i, 0].clamp(min=0), "CFM pred", dict(cmap="jet", norm=norm)),
        )
        for ax, (img, title, kw) in zip(axes[i], panels):
            im = ax.imshow(img.detach().cpu(), **kw)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
