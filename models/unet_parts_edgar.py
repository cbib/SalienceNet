""" Parts of the U-Net model """
from typing import Any, List, Tuple

import torch  # type: ignore
import torch.nn as nn  # type: ignore


class DoubleConv(nn.Module):
    """
    A double convolutional block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop_r: float,
        conv: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the DoubleConv block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop_r: dropout rate
        :type drop_r: float
        """
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop_r),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down_Block(nn.Module):
    """Downscaling double conv then maxpool"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Down_Block
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Down_Block, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)
        self.down = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.conv(x)
        return c, self.down(c)


class Bridge(nn.Module):
    """
    Bridge block (conv -> bn -> relu -> conv -> bn -> relu)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Bridge block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Bridge, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Up_Block, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = DoubleConv(in_channels, out_channels, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolutional block (conv -> bn -> relu -> conv -> bn -> sigmoid)
    """

    def __init__(
        self, in_channels: int, out_channels: int, conv_l: nn.Module = nn.Conv2d
    ):
        """
        Initialize the OutConv block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        """
        super(OutConv, self).__init__()
        self.conv = conv_l(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


"""
################################################################################
## Residual Blocks
################################################################################
"""


class DoubleConvFullPreact(nn.Module):
    """
    Double convolutional block with full preactivation

    :param in_channels: number of input channels
    :type in_channels: int
    :param out_channels: number of output channels
    :type out_channels: int
    :param drop: dropout rate
    :type drop: float
    """

    def __init__(self, in_channels: int, out_channels: int, drop_r: float):
        super(DoubleConvFullPreact, self).__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(drop_r),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(drop_r),
        )
        self.id_map = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.double_conv(x)
        x = self.id_map(x)
        return skip + x


class Down_Block_res(nn.Module):
    """Downscaling with double conv full preactivation then maxpool"""

    def __init__(self, in_channels: int, out_channels: int, drop: float = 0.1):
        """
        Initialize the Down_Block_res block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Down_Block_res, self).__init__()
        self.conv = DoubleConvFullPreact(in_channels, out_channels, drop)
        self.down = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.conv(x)
        return c, self.down(c)


class Bridge_res(nn.Module):
    """
    Bridge block with double conv full preactivation
    """

    def __init__(self, in_channels: int, out_channels: int, drop: float = 0.1):
        """
        Initialize the Bridge_res block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Bridge_res, self).__init__()
        self.conv = DoubleConvFullPreact(in_channels, out_channels, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up_Block_res(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, drop: float = 0.1):
        """
        Initialize the Up_Block_res block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Up_Block_res, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
        )
        self.conv = DoubleConvFullPreact(in_channels, out_channels, drop)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)


"""
################################################################################
## Tweaks
################################################################################
"""


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Any = "same",
        bias: bool = False,
        stride: int = 1,
    ):
        super(SeparableConv2d, self).__init__()
        padding = 1 if stride > 1 else "same"
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias,
            padding=padding,
            stride=stride,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias, padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Normalize_Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super(Normalize_Down, self).__init__()
        self.down = nn.MaxPool2d(kernel)
        self.norm = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.down(x))


class Normalize_Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int):
        super(Normalize_Up, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=kernel)
        self.norm = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.up(x))


class Concat_Block(nn.Module):
    def __init__(
        self,
        kernels_down: List,
        filters_down: List,
        kernels_up: List,
        filters_up: List,
    ):
        super(Concat_Block, self).__init__()
        self.norm_down = nn.ModuleList(
            [
                Normalize_Down(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_down, filters_down)
            ]
        )
        self.norm_up = nn.ModuleList(
            [
                Normalize_Up(in_, 32, kernel)
                for (kernel, in_) in zip(kernels_up, filters_up)
            ]
        )

    def forward(self, down: List[torch.Tensor], up: List[torch.Tensor]) -> torch.Tensor:
        res = [l(d) for d, l in zip(down, self.norm_down)]
        res.extend(l(u) for u, l in zip(up, self.norm_up))
        return torch.cat(res, dim=1)


class Up_Block_3p(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Up_Block_3p, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, 32, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConv(160, 160, drop, conv=conv_l)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)


class Strided_Down_Block(nn.Module):
    """Downscaling double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        super(Strided_Down_Block, self).__init__()

        self.conv1 = nn.Sequential(
            conv_l(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop),
        )
        self.conv2 = nn.Sequential(
            conv_l(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout2d(drop),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.conv1(x)
        return c, self.conv2(c)


class RCL_Block(nn.Module):
    def __init__(
        self,
        filters_in: int,
        filters_out: int,
        kernel_size: int,
        dp: nn.Module = nn.Dropout2d,
        conv_l: nn.Module = nn.Conv2d,
        dp_rate: float = 0.1,
        depth: int = 3,
    ):
        super(RCL_Block, self).__init__()
        self.conv1 = conv_l(
            filters_in, filters_out, kernel_size=kernel_size, padding="same"
        )
        self.stack = nn.Sequential(
            nn.BatchNorm2d(filters_out, momentum=0.9997), nn.PReLU(), dp(dp_rate)
        )
        self.conv2 = conv_l(
            filters_out, filters_out, kernel_size=kernel_size, padding="same"
        )
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(x)
        s = self.stack(c1)
        c = self.conv2(s)
        c = torch.add(c1, c)
        for _ in range(self.depth - 1):
            s = self.stack(c)
            c = self.conv2(s)
            c = torch.add(c1, c)
        return self.stack(c)


class GatingSignal(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool = False):
        super(GatingSignal, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.activation(x)


class Attention_Gate(nn.Module):
    def __init__(self, in_channels: int):
        super(Attention_Gate, self).__init__()
        self.conv_theta_x = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(2, 2)
        )
        self.conv_phi_g = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.att = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, kernel_size=(1, 1)),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x: torch.Tensor, gat: torch.Tensor) -> torch.Tensor:
        theta_x = self.conv_theta_x(x)
        phi_g = self.conv_phi_g(gat)
        res = torch.add(phi_g, theta_x)
        res = self.att(res)
        return torch.mul(res, x)


class Attention_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_Block, self).__init__()
        self.gating = GatingSignal(in_channels, out_channels)
        self.att_gate = Attention_Gate(out_channels)

    def forward(self, x, conc):
        gat = self.gating(x)
        return self.att_gate(conc, gat)


class NestedBlock(nn.Module):
    def __init__(
        self,
        filters_in: int,
        filters_out: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        super(NestedBlock, self).__init__()
        self.conv = DoubleConv(filters_in, filters_out, drop, conv_l)
        self.up = nn.ConvTranspose2d(
            filters_out * 2, filters_out, kernel_size=2, stride=2
        )

    def forward(
        self, x: torch.Tensor, conc: List[torch.Tensor], up: torch.Tensor
    ) -> torch.Tensor:
        upped = self.up(up)
        conc.append(upped)
        conc.append(x)
        x = torch.cat(conc, dim=1)
        return self.conv(x)


class Up_Block_res_3p(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        drop: float = 0.1,
        conv_l: nn.Module = nn.Conv2d,
    ):
        """
        Initialize the Up_Block

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param drop: dropout rate
        :type drop: float
        """
        super(Up_Block_res_3p, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, 32, kernel_size=(2, 2), stride=(2, 2))
        self.conv = DoubleConvFullPreact(160, 160, drop)

    def forward(self, x: torch.Tensor, conc: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x)
        x = torch.cat([conc, x1], dim=1)
        return self.conv(x)
