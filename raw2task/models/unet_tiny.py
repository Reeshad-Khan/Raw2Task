import torch
import torch.nn as nn

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNetTiny(nn.Module):
    """Very small UNet-like model; input is 1-channel RAW mosaic; output logits K classes."""
    def __init__(self, in_channels=1, num_classes=19, base=32):
        super().__init__()
        self.enc1 = nn.Sequential(conv_bn_relu(in_channels, base),
                                  conv_bn_relu(base, base))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_bn_relu(base, base*2),
                                  conv_bn_relu(base*2, base*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(conv_bn_relu(base*2, base*4),
                                  conv_bn_relu(base*4, base*4))

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = nn.Sequential(conv_bn_relu(base*4, base*2),
                                  conv_bn_relu(base*2, base*2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = nn.Sequential(conv_bn_relu(base*2, base),
                                  conv_bn_relu(base, base))

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)          # B,base,H,W
        p1 = self.pool1(e1)        # H/2
        e2 = self.enc2(p1)         # B,2B,H/2,W/2
        p2 = self.pool2(e2)        # H/4
        e3 = self.enc3(p2)         # B,4B,H/4,W/4

        u2 = self.up2(e3)          # H/2
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)          # H
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)       # logits (B,K,H,W)
