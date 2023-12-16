def get_network(opt_net):
    """Instantiate the network with configuration"""

    kind = opt_net.pop("type").lower()

    # generators
    if kind == "sr_resnet":
        from . import SRResNet

        net = SRResNet.SRResNet
    elif kind == "rrdb_net":  # ESRGAN
        from . import RRDBNet

        net = RRDBNet.RRDBNet
    elif kind == "mrrdb_net":  # Modified ESRGAN
        from . import RRDBNet

        net = RRDBNet.MRRDBNet
    elif kind == "ppon":
        from . import PPON

        net = PPON.PPON
    elif kind == "pan_net":
        from . import PAN

        net = PAN.PAN
    elif kind == "unet_net":
        from . import UNet

        net = UNet.UnetGenerator
    elif kind == "resnet_net":
        from . import ResNet

        net = ResNet.ResnetGenerator
    elif kind == "wbcunet_net":
        from . import WBCNet

        net = WBCNet.UnetGeneratorWBC
    else:
        raise NotImplementedError("Model [{:s}] not recognized".format(kind))

    net = net(**opt_net)

    return net
