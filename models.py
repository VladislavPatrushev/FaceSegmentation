from typing import Tuple
from argparse import Namespace

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def initialize_model(params: Namespace,
                     device: torch.device = torch.device("cpu")
                     ) -> Tuple[nn.Module, str]:
    if params.model_type == "deeplabv3":
        model = smp.DeepLabV3(params.backbone,
                              encoder_weights=params.weights,
                              classes=params.classes)
    elif params.model_type == "unet":
        model = smp.Unet(params.backbone,
                         encoder_weights=params.weights,
                         classes=params.classes)

    elif params.model_type == "unet_plusplus":
        model = smp.UnetPlusPlus(params.backbone,
                                 encoder_weights=params.weights,
                                 classes=params.classes)
    else:
        raise Exception('Incorrect model_type. Check config!')

    model_name = '_'.join([params.model_type,
                           params.weights,
                           params.backbone,
                           params.model_version])
    model.to(device)
    return model, model_name
