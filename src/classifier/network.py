import torch
import torch.nn as nn
import classifier.mobileNetV3 as mn

def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

class MLP(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        # RGB branch
        self.rgb_mlp = mn.MobileNetV3_adaption(in_channels=in_channels)
        # MV branch
        self.mv_mlp = mn.MobileNetV3_adaption(in_channels=in_channels)
        # alpha param to weight the result of the two networks
        self.alpha_param = nn.Parameter(torch.tensor(0.0)) # Not in the paper, they just average

    def forward(self, x: torch.Tensor):
        imgs, mvs = x

        # frame-wise probability from each branch
        res_img = self.rgb_mlp(imgs)
        res_mv  = self.mv_mlp(mvs)

        # convert alpha_param in (0,1)
        alpha = torch.sigmoid(self.alpha_param)

        # Weighted sum
        out = alpha * res_img + (1 - alpha) * res_mv

        return out
    
    # freeze/unfreeze the backbones of the two branches
    def freeze_backbones(self, bn_eval: bool = True):
        self.rgb_mlp.freeze_backbone(bn_eval=bn_eval)
        self.mv_mlp.freeze_backbone(bn_eval=bn_eval)

    def unfreeze_backbones(self):
        self.rgb_mlp.unfreeze_backbone()
        self.mv_mlp.unfreeze_backbone()
