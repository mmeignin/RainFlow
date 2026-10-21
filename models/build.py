from models.unet import UNetModelWrapper


def build_unet(num_channel, n_cond, image_size, channel_mult, attention_resolutions,
               num_heads, num_head_channels, device,use_scale_shift_norm,dropout, residual=False,):
    """Rain rate (1 channel) conditioned on `n_cond` channels.

    In residual mode the model also takes the deterministic baseline `x_det` as an
    extra conditioning channel, so the input width grows by one.

    `use_scale_shift_norm` for ADALN/film behavior
    """
    in_ch = 1 + n_cond + int(residual)
    return UNetModelWrapper(
        dim=(in_ch, image_size, image_size), out_channels=1, num_res_blocks=2,
        num_channels=num_channel, channel_mult=channel_mult,
        num_heads=num_heads, num_head_channels=num_head_channels,
        attention_resolutions=attention_resolutions,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    ).to(device)
