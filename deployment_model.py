import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + growth_rate, growth_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)
        concat1 = torch.cat([x, out1], dim=1)
        out2 = self.conv2(concat1)
        concat2 = torch.cat([concat1, out2], dim=1)
        out3 = self.conv3(concat2)
        return out3 + x

class ResidualSEBlock(nn.Module):
    def __init__(self, channels=64, reduction=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.se = SEBlock(channels, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        out = self.relu(out)
        return out

class ASPP(nn.Module):
    def __init__(self, channels=64, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=8, dilation=8)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        concat = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.project(concat)

class MultiScaleFusion(nn.Module):
    def __init__(self, channels=64, out_channels=64):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv2d(channels, out_channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv2d(channels, out_channels, kernel_size=3, padding=2, dilation=2)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        concat = torch.cat([f1, f2, f3], dim=1)
        return self.fusion(concat)

class RDMSNet(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual Dense Block x3
        self.rdb1 = ResidualDenseBlock(hidden_channels)
        self.rdb2 = ResidualDenseBlock(hidden_channels)
        self.rdb3 = ResidualDenseBlock(hidden_channels)

        # Residual SE Block x5
        self.se1 = ResidualSEBlock(hidden_channels)
        self.se2 = ResidualSEBlock(hidden_channels)
        self.se3 = ResidualSEBlock(hidden_channels)
        self.se4 = ResidualSEBlock(hidden_channels)
        self.se5 = ResidualSEBlock(hidden_channels)

        # ASPP Module
        self.aspp = ASPP(hidden_channels, hidden_channels)

        # MultiScale Feature Fusion
        self.multiscale_fusion = MultiScaleFusion(hidden_channels, hidden_channels)

        # 1x1 Fusion Conv
        self.fusion_1x1 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Final Conv Layer
        self.final_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        initial_feat = self.initial_conv(x)

        # RDB block x3
        x = self.rdb1(initial_feat)
        x = self.rdb2(x)
        x = self.rdb3(x)

        # SE block x5
        x = self.se1(x)
        x = self.se2(x)
        x = self.se3(x)
        x = self.se4(x)
        x = self.se5(x)

        # ASPP
        x = self.aspp(x)

        # MultiScale Fusion
        x = self.multiscale_fusion(x)

        # 1x1 Fusion Conv
        x = self.fusion_1x1(x)

        # Global Residual Skip
        x = x + initial_feat

        # Final Conv
        out = self.final_conv(x)
        return out
