from typing import Optional

from torch import Tensor, nn
from torchvision import models

from src.utils.functional import freeze_weight


class ResnetBackbone(nn.Module):
    def __init__(
        self,
        depth: int,
        pretrained: bool = False,
        num_stages: int = 3,
        frozen_stages: Optional[int] = None,
    ):
        super().__init__()

        resnet_model = self._get_models(depth, pretrained)
        layers = (
            resnet_model.layer1,
            resnet_model.layer2,
            resnet_model.layer3,
            resnet_model.layer4,
        )
        self.nets = nn.ModuleList(
            (
                resnet_model.conv1,
                resnet_model.bn1,
                resnet_model.relu,
                resnet_model.maxpool,
            )
        )
        if frozen_stages is not None:
            freeze_weight(self.nets)

        for i in range(num_stages):
            if frozen_stages is not None and i < frozen_stages:
                freeze_weight(layers[i])
            self.nets.append(layers[i])

    def forward(self, x: Tensor):
        for net in self.nets:
            x = net(x)
        return x

    def _get_models(self, depth: int, pretrained: bool = False):
        if depth == 50:
            return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif depth == 101:
            return models.resnet101(pretrained=models.ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"Cannot find Resnet version: {depth}")
