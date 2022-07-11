from monai.networks.layers import Norm
from monai.networks.nets import UNet

from csnet.models.csnet import CSNet
from csnet.models.csnet_orig import CSNetOrig


def get_model(config):
    if config.model.lower() == 'unet':
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            num_res_units=config.num_res_units,
            norm=Norm.BATCH,
        )
    elif config.model.lower() == 'csnet':
        net = CSNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=config.n_channels,
            strides=(2,) * (len(config.n_channels) - 1),
            num_res_units=config.num_res_units,
            norm=Norm.BATCH,
        )
    elif config.model.lower() == 'csnet_orig':
        net = CSNetOrig(2, 1)
    else:
        raise NotImplementedError(
            rf'{config.model} is an invalid model; must be one of ["unet", "csnet", "csnet_orig"]')
    return net
