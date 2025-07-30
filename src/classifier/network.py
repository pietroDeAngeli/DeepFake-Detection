import torch
import torch.nn as nn
import classifier.mobileNetV3 as mn

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
