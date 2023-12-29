import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(MetaAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = F.interpolate(y, size=(self.kernel_size, self.kernel_size), mode='bilinear', align_corners=True)
        return x * y
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    kernel_size = input.shape[2]
    meta_attn = MetaAttentionBlock(channel=512, reduction=16, kernel_size=kernel_size)
    output = meta_attn(input)
    print(output.shape)
