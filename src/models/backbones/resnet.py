from torchvision import models
from torch import nn

from utils.functional import freeze_weight


class ResnetBackbone(nn.Module):
    def __init__(self, *modules):
        super().__init__()

        self._nets = nn.Sequential(*modules)

    def forward(self, x):
        return self._nets(x)

    def train(self, mode=True):
        self.training = mode
        self._nets.train(mode)

        def batchnorm_eval(m):
            classname = m.__class__.__name__

            if classname.find("BatchNorm") != -1 and m.training:
                m.eval()

        self._nets.apply(batchnorm_eval)
        assert not self._nets[1].training


def resnet_backbone(layer: int = 50, fixed_blocks=2):
    assert fixed_blocks >= 0 and fixed_blocks < 4

    if layer == 50:
        resnet = models.resnet50(pretrained=True)
    elif layer == 101:
        resnet = models.resnet101(pretrained=True)
    else:
        raise ValueError("Only support resnet 50 or 101 currently")

    freeze_weight(resnet.bn1)
    freeze_weight(resnet.conv1)

    if fixed_blocks >= 3:
        freeze_weight(resnet.layer3)
    if fixed_blocks >= 2:
        freeze_weight(resnet.layer2)
    if fixed_blocks >= 1:
        freeze_weight(resnet.layer1)

    def set_bn_fix(m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            freeze_weight(m)

    # Fixed batch_norm
    resnet.apply(set_bn_fix)

    backbone = ResnetBackbone(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
    )

    return backbone
