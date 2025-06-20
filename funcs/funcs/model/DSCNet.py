import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import dropout
import einops
from torchvision.models.segmentation.deeplabv3 import ASPP
from torchvision.models.swin_transformer import SwinTransformerBlock
from funcs.model.SegFormer import MiT

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x

class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
        device: str | torch.device = "cuda",
    ):
        """
        A Dynamic Snake Convolution Implementation

        Based on:

            TODO

        Args:
            in_ch: number of input channels. Defaults to 1.
            out_ch: number of output channels. Defaults to 1.
            kernel_size: the size of kernel. Defaults to 9.
            extend_scope: the range to expand. Defaults to 1 for this method.
            morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
            if_offset: whether deformation is required,  if it is False, it is the standard convolution kernel. Defaults to True.

        """

        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        # self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

    def forward(self, input: torch.Tensor):
        # Predict offset map between [-1, 1]
        offset = self.offset_conv(input)
        # offset = self.bn(offset)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        # Run deformative conv
        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(
            offset=offset,
            morph=self.morph,
            extend_scope=self.extend_scope,
            device=self.device,
        )
        deformed_feature = get_interpolated_feature(
            input,
            y_coordinate_map,
            x_coordinate_map,
        )

        output = self.dsc_conv_y(deformed_feature) if self.morph else self.dsc_conv_x(deformed_feature)

        # Groupnorm & ReLU
        output = self.gn(output)
        output = self.relu(output)

        return output


def get_coordinate_map_2D(
    offset: torch.Tensor,
    morph: int,
    extend_scope: float = 1.0,
    device: str | torch.device = "cuda",
):
    """Computing 2D coordinate map of DSCNet based on: TODO

    Args:
        offset: offset predict by network with shape [B, 2*K, W, H]. Here K refers to kernel size.
        morph: the morphology of the convolution kernel is mainly divided into two types along the x-axis (0) and the y-axis (1) (see the paper for details).
        extend_scope: the range to expand. Defaults to 1 for this method.
        device: location of data. Defaults to 'cuda'.

    Return:
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
    """

    if morph not in (0, 1): raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        """
        Initialize the kernel and flatten the kernel
            y: only need 0
            x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
        """
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (
                y_offset_new_[center + index - 1] + y_offset_[center + index]
            )
            y_offset_new_[center - index] = (
                y_offset_new_[center - index + 1] + y_offset_[center - index]
            )

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        """
        Initialize the kernel and flatten the kernel
            y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
            x: only need 0
        """
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        # The center position remains unchanged and the rest of the positions begin to swing
        # This part is quite simple. The main idea is that "offset is an iterative process"

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (
                x_offset_new_[center + index - 1] + x_offset_[center + index]
            )
            x_offset_new_[center - index] = (
                x_offset_new_[center - index + 1] + x_offset_[center - index]
            )

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map


def get_interpolated_feature(
    input_feature: torch.Tensor,
    y_coordinate_map: torch.Tensor,
    x_coordinate_map: torch.Tensor,
    interpolate_mode: str = "bilinear",
):
    """From coordinate map interpolate feature of DSCNet based on: TODO

    Args:
        input_feature: feature that to be interpolated with shape [B, C, H, W]
        y_coordinate_map: coordinate map along y-axis with shape [B, K_H * H, K_W * W]
        x_coordinate_map: coordinate map along x-axis with shape [B, K_H * H, K_W * W]
        interpolate_mode: the arg 'mode' of nn.functional.grid_sample, can be 'bilinear' or 'bicubic' . Defaults to 'bilinear'.

    Return:
        interpolated_feature: interpolated feature with shape [B, C, K_H * H, K_W * W]
    """

    if interpolate_mode not in ("bilinear", "bicubic"):
        raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

    y_max = input_feature.shape[-2] - 1
    x_max = input_feature.shape[-1] - 1

    y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
    x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

    y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
    x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

    # Note here grid with shape [B, H, W, 2]
    # Where [:, :, :, 2] refers to [x ,y]
    grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

    interpolated_feature = nn.functional.grid_sample(
        input=input_feature,
        grid=grid,
        mode=interpolate_mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return interpolated_feature


def _coordinate_map_scaling(
    coordinate_map: torch.Tensor,
    origin: list,
    target: list = [-1, 1],
):
    """Map the value of coordinate_map from origin=[min, max] to target=[a,b] for DSCNet based on: TODO

    Args:
        coordinate_map: the coordinate map to be scaled
        origin: original value range of coordinate map, e.g. [coordinate_map.min(), coordinate_map.max()]
        target: target value range of coordinate map,Defaults to [-1, 1]

    Return:
        coordinate_map_scaled: the coordinate map after scaling
    """
    min, max = origin
    a, b = target

    coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

    scale_factor = (b - a) / (max - min)
    coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

    return coordinate_map_scaled
    
class MultiView_DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 9,
        extend_scope: float = 1.0,
        device_id: str | torch.device = "cuda",
    ):
        super().__init__()

        device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        self.dsconv_x = DSConv(in_channels, out_channels, kernel_size, extend_scope, 1, True, device_id).to(device)
        self.dsconv_y = DSConv(in_channels, out_channels, kernel_size, extend_scope, 0, True, device_id).to(device)
        self.conv = Conv(in_channels, out_channels)
        self.conv_fusion = Conv(out_channels * 3, out_channels)

    def forward(self, x):
        conv_x = self.conv(x)
        dsconvx_x = self.dsconv_x(x)
        dsconvy_x = self.dsconv_y(x)
        x = self.conv_fusion(torch.cat([conv_x, dsconvx_x, dsconvy_x], dim=1))
        return x
    
class SwinBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels

        self.swin1 = SwinTransformerBlock(
            dim=in_channels,            # 输入通道数
            num_heads=12,               # 注意力头数量
            window_size=[7, 7],         # 窗口大小
            shift_size=[0, 0]           # 偏移窗口大小
        )
        self.swin2 = SwinTransformerBlock(
            dim=in_channels,
            num_heads=12,
            window_size=[14, 14],
            shift_size=[0, 0]
        )
    
    def forward(self, x):
        x = x.permute((0,2,3,1))
        x1 = self.swin1(x)
        x2 = self.swin2(x)
        x = torch.cat([x1, x2], dim=-1)
        return x.permute((0,3,1,2))

class MiTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if in_channels == 3: out_channels = 12
        params = {
            "channels": in_channels, 
            "dims": [out_channels],
            "heads": [8],
            "ff_expansion": [4],
            "reduction_ratio": [4],
            "num_layers": [1]
        }

        self.mit = MiT(**params)
    
    def forward(self, x):
        return self.mit(x)


class DSCNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        kernel_size=9,
        extend_scope=3,
        layer_depth=5,
        rate=72,
        dim=1,
        ga="ASPP", # "" -> ok
        device_id="0"
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dim = dim 
        self.layer_depth = layer_depth

        device_id = "cuda:{}".format(device_id)

        self.ga = ga
        if ga == "ASPP":
            init_ga_module = lambda in_ch, out_ch : ASPP(in_channels=in_ch, out_channels=out_ch, atrous_rates=[24])
        elif ga == "SWinT":
            init_ga_module = lambda in_ch, out_ch : SwinBlock(in_channels=in_ch, out_channels=out_ch)
        elif ga == "MiT":
            init_ga_module = lambda in_ch, out_ch : MiTBlock(in_channels=in_ch, out_channels=out_ch)
        else:
            self.ga = ""

        bf = [2**x for x in range(layer_depth)] # bf: basic_feature
        bf += bf[:-1][::-1] # [1, 2, 4, 8, 4, 2, 1]

        ga_flag = int(bool(self.ga))
        in_channel_nums = [in_channels] + [rate * bf[0]] + [x*rate for x in bf[1:layer_depth-1]] # encoder
        in_channel_nums += [(3+ga_flag)*x*rate for x in bf[-layer_depth+1:-1]] + [3*rate*bf[-1]] # decoder

        out_channel_nums = [x*rate for x in bf]

        init_DSConvFusion = lambda in_ch, out_ch : MultiView_DSConv(in_ch, out_ch, kernel_size, extend_scope, device_id)

        self.dsconvs = nn.Sequential(*[init_DSConvFusion(in_ch, out_ch) for in_ch, out_ch in zip(in_channel_nums, out_channel_nums)])

        self.out_conv = nn.Conv2d(rate, out_channels, 1)

        if self.ga:
            self.ga_blocks = nn.Sequential(*[init_ga_module(ch, ch*2) for ch in in_channel_nums[:layer_depth]])

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        layer_depth = self.layer_depth

        edc_features = []
        for i in range(layer_depth):
            xi = self.dsconvs[i](x)
            if i < layer_depth - 1: # downsampling
                if self.ga and i > 0:
                    xg = self.ga_blocks[i](x)
                    edc_features.append(torch.cat([xi, xg], dim=1))
                else:
                    edc_features.append(xi)
                x = self.down(xi)
            else: x = xi
        # decoder:
        x = self.up(x)
        
        for i in range(1, layer_depth-1)[::-1]:
            x = torch.cat([x, edc_features[i]], dim=1)
            x = self.dsconvs[2 * (layer_depth-1) - i](x)
            x = self.up(x)
        
        x = torch.cat([x, edc_features[0]], dim=1)
        x = self.dsconvs[2 * (layer_depth-1)](x)
        out = self.out_conv(x)

        return self.sigmoid(out)