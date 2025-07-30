import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class MobileNetV3_adaption(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(MobileNetV3_adaption, self).__init__()
        # 1) Load a MobileNetV3 backbone with random initialization
        self.backbone = mobilenet_v3_small(weights=None)

        # 2) Adapt the first conv layer to accept `in_channels` input channels
        orig_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None
        )
        # Initialize the new conv weights with Kaiming (He) initialization
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)
        self.backbone.features[0][0] = new_conv

        # 3) Replace the classifier head with a single-output linear layer
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor):
        # x: Tensor[N, C, H, W] where N = number of frames
        logits = self.backbone(x)          # [N, 1]
        prob = torch.sigmoid(logits.mean(dim=0))    # [1]
        return prob
