import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetV3_adaption(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Backbone pretrained
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Adapt input conv to in_channels
        orig_conv = base.features[0][0]
        self.input_adapter = nn.Conv2d(
            in_channels,
            orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=(orig_conv.bias is not None),
        )
        nn.init.kaiming_normal_(self.input_adapter.weight, mode="fan_out", nonlinearity="relu")
        if self.input_adapter.bias is not None:
            nn.init.zeros_(self.input_adapter.bias)

        # Revove original conv from the model
        base.features[0][0] = nn.Identity()

        # Separate backbone and head
        self.backbone = base.features
        self.pool     = base.avgpool
        in_features   = base.classifier[0].in_features  # 576 per v3-small torchvision

        self.head = nn.Linear(in_features, 1)

        # Freeze flag
        self._backbone_frozen = False

    def forward(self, x: torch.Tensor):
        # x: [N, C, H, W] (N = #video frames)
        x = self.input_adapter(x)
        feats = self.backbone(x)
        feats = self.pool(feats).flatten(1)     # [N, in_features]
        logits = self.head(feats)               # [N, 1]
        video_logit = logits.mean(dim=0)        # [1] average pooling over frames
        return video_logit 

    # Freezing/unfreezing the backbone
    def freeze_backbone(self, bn_eval: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_frozen = True
        if bn_eval:
            self.backbone.eval()

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False
        self.backbone.train()

    def trainable_head_params(self):
        return list(self.input_adapter.parameters()) + list(self.head.parameters())
