import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=((1, 1), (3, 1), (5, 1))):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d1, d2 in dilations:
            self.convs1.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=d1, padding=get_padding(kernel_size, d1),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    nn.Conv1d(
                        channels, channels, kernel_size,
                        dilation=d2, padding=get_padding(kernel_size, d2),
                    )
                )
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_parametrizations(c, "weight")
        for c in self.convs2:
            remove_parametrizations(c, "weight")


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=80,
        upsample_initial_channel=512,
        upsample_kernel_sizes=(16, 16, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=(((1, 1), (3, 1), (5, 1)),),
    ):
        super().__init__()

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_kernel_sizes)

        if len(resblock_dilations) == 1:
            resblock_dilations = resblock_dilations * self.num_kernels

        self.conv_pre = weight_norm(
            nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for k in upsample_kernel_sizes:
            stride = k // 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, k,
                        stride=stride, padding=(k - stride) // 2,
                    )
                )
            )
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch_i = upsample_initial_channel // (2 ** (i + 1))
            for j, k in enumerate(resblock_kernel_sizes):
                self.resblocks.append(ResBlock(ch_i, k, resblock_dilations[j]))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, mel, **kwargs):
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x)

            xs = torch.zeros_like(x)
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return {"audio_gen": x}

    def remove_weight_norm(self):
        remove_parametrizations(self.conv_pre, "weight")
        for up in self.ups:
            remove_parametrizations(up, "weight")
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_parametrizations(self.conv_post, "weight")

    def __str__(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        info = super().__str__()
        info += f"\nAll parameters: {total}"
        info += f"\nTrainable parameters: {trainable}"
        return info
