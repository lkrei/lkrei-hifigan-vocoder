import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

LRELU_SLOPE = 0.1


class PeriodSubDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(2, 0))),
                weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), (1, 1), padding=(2, 0))),
            ]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), padding=(1, 0)))

    def forward(self, x):
        features = []
        b, c, t = x.shape

        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), "reflect")
            t = t + pad_len

        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodSubDiscriminator(p) for p in periods]
        )

    def forward(self, x):
        outputs = []
        features = []
        for d in self.discriminators:
            out, feat = d(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features


class ScaleSubDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            features.append(x)

        x = self.conv_post(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)

        return x, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleSubDiscriminator(use_spectral_norm=True),
                ScaleSubDiscriminator(),
                ScaleSubDiscriminator(),
            ]
        )
        self.pooling = nn.ModuleList(
            [
                nn.AvgPool1d(4, 2, padding=2),
                nn.AvgPool1d(4, 2, padding=2),
            ]
        )

    def forward(self, x):
        outputs = []
        features = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        return outputs, features


class Discriminator(nn.Module):
    def __init__(self, mpd_periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods=mpd_periods)
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        mpd_outputs, mpd_features = self.mpd(x)
        msd_outputs, msd_features = self.msd(x)
        return mpd_outputs + msd_outputs, mpd_features + msd_features
