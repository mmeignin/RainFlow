from models.unet import UNetModelWrapper


def build_unet(num_channel, n_cond, image_size, channel_mult, attention_resolutions, device):
    """Rain rate (1 channel) conditioned on `n_cond` channels, at `image_size`x`image_size`.
    """
    return UNetModelWrapper(
        dim=(1 + n_cond, image_size, image_size), out_channels=1, num_res_blocks=2,
        num_channels=num_channel, channel_mult=channel_mult,
        num_heads=1, num_head_channels=-1, attention_resolutions=attention_resolutions,
        dropout=0.1,
    ).to(device)
