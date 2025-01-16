# wip
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nunif.models import I2IBaseModel, register_model, register_model_factory
from nunif.modules.attention import WindowMHA2d, WindowScoreBias
from nunif.modules.replication_pad2d import ReplicationPad2dNaive as ReplicationPad2dNaive, replication_pad2d_naive
from nunif.modules.init import icnr_init, basic_module_init
from nunif.modules.compile_wrapper import conditional_compile


class GLUConvMLP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mlp_ratio=2, padding=True):
        super().__init__()
        mid = int(out_channels * mlp_ratio)
        # assert kernel_size % 2 == 1
        if padding:
            self.pad = ReplicationPad2dNaive(((kernel_size - 1) // 2,) * 4, detach=True)
        else:
            self.pad = nn.Identity()
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid // 2, out_channels, kernel_size=kernel_size, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = self.w1(x)
        x = F.glu(x, dim=1)
        x = self.pad(x)
        x = self.w2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        return x


class MLPBlocks(nn.Module):
    def __init__(self, in_channels, mlp_ratio=2, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([MLP(in_channels, in_channels, mlp_ratio=mlp_ratio) for i in range(num_layers)])
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        for m in self.layers:
            x = x + m(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=2):
        super().__init__()
        const_mlp_ratio = 1.0
        mid = int(out_channels * mlp_ratio * const_mlp_ratio)
        self.w1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0)
        self.w2 = nn.Conv2d(mid, out_channels, kernel_size=1, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = self.w1(x)
        x = F.leaky_relu(x, 0.1, inplace=True)
        x = self.w2(x)
        return x


class WACBlock(nn.Module):
    """ Window MHA + Multi Layer Conv2d
    """
    def __init__(self, in_channels, num_heads=4, qkv_dim=None, window_size=8, mlp_ratio=2, padding=True, conv_mlp=True):
        super(WACBlock, self).__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.padding = padding
        self.mha = WindowMHA2d(in_channels, num_heads, qkv_dim=qkv_dim, window_size=window_size)
        self.relative_bias = WindowScoreBias(self.window_size)
        if conv_mlp:
            self.conv_mlp = GLUConvMLP(in_channels, in_channels, kernel_size=3, mlp_ratio=mlp_ratio, padding=padding)
        else:
            self.conv_mlp = MLP(in_channels, in_channels, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x1 = self.mha(x, attn_mask=self.relative_bias())
        x = x + x1
        if isinstance(self.conv_mlp, GLUConvMLP):
            if self.padding:
                x = x + self.conv_mlp(x)
            else:
                x = F.pad(x, (-1,) * 4) + self.conv_mlp(x)
        else:
            x = x + self.conv_mlp(x)
        return x


class WACBlocks(nn.Module):
    def __init__(self, in_channels, num_heads=4, qkv_dim=None,
                 window_size=8, mlp_ratio=2, num_layers=2, padding=True, conv_mlp=True):
        super(WACBlocks, self).__init__()
        if isinstance(window_size, int):
            window_size = [window_size] * num_layers
        if isinstance(padding, bool):
            padding = [padding] * num_layers
        if isinstance(conv_mlp, bool):
            conv_mlp = [conv_mlp] * num_layers

        self.blocks = nn.Sequential(
            *[WACBlock(in_channels, window_size=window_size[i],
                       num_heads=num_heads, qkv_dim=qkv_dim, mlp_ratio=mlp_ratio, padding=padding[i], conv_mlp=conv_mlp[i])
              for i in range(num_layers)])

    def forward(self, x):
        return self.blocks(x)


class Overscan(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        C = in_channels
        padding = 1 + 2 + 3 + 1
        self.pad = ReplicationPad2dNaive((padding,) * 4, detach=True)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, dilation=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(C, C // 2, kernel_size=3, dilation=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(C // 2, C // 2, kernel_size=3, dilation=3, stride=1, padding=0)
        self.fuse = nn.Sequential(
            nn.Conv2d(C + C // 2 + C // 2, C, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0))
        basic_module_init(self)

    def forward(self, x):
        x = self.pad(x)
        x1 = self.conv1(x)
        x1 = F.leaky_relu(x1, 0.2, inplace=True)
        x2 = self.conv2(x1)
        x2 = F.leaky_relu(x2, 0.2, inplace=True)
        x3 = self.conv3(x2)
        x3 = F.leaky_relu(x3, 0.2, inplace=True)
        x1 = F.pad(x1, (-(2 + 3),) * 4)
        x2 = F.pad(x2, (-3,) * 4)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x = self.fuse(x4)
        return x


class IR(nn.Module):
    def __init__(self, in_channels=3, out_channels=16):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.overscan = Overscan(out_channels)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=0)
        basic_module_init(self)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = replication_pad2d_naive(x, (1,) * 4)
        x = self.patch(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        ov = self.overscan(x)
        x = torch.cat([x, ov], dim=1)
        x = replication_pad2d_naive(x, (1,) * 4)
        x = self.fusion(x)
        return x


class PatchDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        basic_module_init(self.conv)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = F.leaky_relu(self.conv(x), 0.2, inplace=True)
        return x


class PatchUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=1, padding=0)
        icnr_init(self.proj, scale_factor=2)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = F.leaky_relu(self.proj(x), 0.2, inplace=True)
        x = F.pixel_shuffle(x, 2)
        return x


class ToImage(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=1, stride=1, padding=0)
        icnr_init(self.proj, scale_factor=scale_factor)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x):
        x = self.proj(x)
        if self.scale_factor > 1:
            x = F.pixel_shuffle(x, self.scale_factor)
        x = F.pad(x, (-self.scale_factor,) * 4)
        return x


class SourceResidual(nn.Module):
    def __init__(self, out_channels, scale_factor, source_channels=3):
        assert out_channels == 3
        super().__init__()
        self.scale_factor = scale_factor
        self.resampling = nn.Conv2d(source_channels, out_channels * scale_factor ** 2,
                                    kernel_size=3, stride=1, padding=0, bias=False)
        # weight for main net
        self.scale_bias = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.nearest_neighbor_init(self.resampling, scale_factor=scale_factor)

    @staticmethod
    def nearest_neighbor_init(m, scale_factor):
        with torch.no_grad():
            OUT, IN, H, W = m.weight.data.shape
            center = (H - 1) // 2
            assert OUT % (scale_factor ** 2) == 0
            weight = torch.zeros((OUT // (scale_factor ** 2), IN, H, W))
            weight[0, 0, center, center] = 1.0
            weight[1, 1, center, center] = 1.0
            weight[2, 2, center, center] = 1.0
            if scale_factor > 1:
                weight = weight.permute(1, 0, 2, 3)
                weight = F.interpolate(weight, scale_factor=scale_factor, mode="nearest")
                weight = F.pixel_unshuffle(weight, scale_factor)
                weight = weight.permute(1, 0, 2, 3)
            m.weight.data.copy_(weight)

    @conditional_compile(["NUNIF_TRAIN", "WAIFU2X_WEB"])
    def forward(self, x, src):
        # print(self.scale_bias)
        # torch.save(self.resampling.state_dict(), "tmp/nn_upsample/weight.pth")
        src = replication_pad2d_naive(src, (1,) * 4)
        src = self.resampling(src)
        if self.scale_factor > 1:
            src = F.pixel_shuffle(src, self.scale_factor)
        unpad = (x.shape[2] - src.shape[2]) // 2
        if unpad != 0:
            src = F.pad(src, (unpad,) * 4)
        x = src + x * self.scale_bias

        return x


class WincUNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, base_dim=96,
                 lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 first_layers=2, last_layers=3,
                 scale_factor=2):
        super(WincUNetBase, self).__init__()
        assert scale_factor in {1, 2, 4}
        self.scale_factor = scale_factor
        C = base_dim
        C2 = int(C * lv2_ratio)
        # assert C % 32 == 0 and C2 % 32 == 0  # slow when C % 32 != 0
        HEADS = max(C // 32, 2)
        HEADS2 = max(C2 // 32, 2)

        # shallow feature extractor
        self.ir = IR(3, 16)
        self.patch = nn.Conv2d(16, C, kernel_size=3, stride=1, padding=0)

        # encoder
        self.wac1 = WACBlocks(C, mlp_ratio=lv1_mlp_ratio,
                              window_size=[8, 6] * first_layers, num_heads=HEADS, num_layers=first_layers)
        self.down1 = PatchDown(C, C2)
        self.wac2 = WACBlocks(C2, mlp_ratio=lv2_mlp_ratio,
                              window_size=[8, 12, 8, 6], num_heads=HEADS2, num_layers=4)
        # decoder
        self.up1 = PatchUp(C2, C)
        self.wac1_proj = nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0)
        self.wac3 = WACBlocks(C, mlp_ratio=lv1_mlp_ratio,
                              window_size=[8, 6] * last_layers, num_heads=HEADS, num_layers=last_layers,
                              conv_mlp=[True] * (last_layers - 1) + [False])
        self.res_mlp = MLPBlocks(C, mlp_ratio=lv1_mlp_ratio, num_layers=1)
        self.to_residual_image = ToImage(C, out_channels, scale_factor=scale_factor)
        self.to_image = SourceResidual(out_channels, scale_factor=scale_factor)

        basic_module_init(self.patch)
        basic_module_init(self.wac1_proj)

        self.tile_mode = False

    def set_tile_mode(self):
        self.tile_mode = True

    def _forward(self, x, src):
        x = self.patch(x)
        x = F.pad(x, (-7,) * 4)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x1 = self.wac1(x)
        x = self.down1(x1)
        x = self.wac2(x)
        x = self.up1(x)
        x = x + self.wac1_proj(x1)
        x = self.wac3(x)
        x = self.res_mlp(x)
        x = self.to_residual_image(x)
        z = self.to_image(x, src)

        return z

    def _forward_tile2x2(self, x, src):
        tl = x[:, :, :64, :64]
        tr = x[:, :, :64, -64:]
        bl = x[:, :, -64:, :64]
        br = x[:, :, -64:, -64:]
        x = torch.cat([tl, tr, bl, br], dim=0).contiguous()

        tl = src[:, :, :64, :64]
        tr = src[:, :, :64, -64:]
        bl = src[:, :, -64:, :64]
        br = src[:, :, -64:, -64:]
        src = torch.cat([tl, tr, bl, br], dim=0).contiguous()

        x = self._forward(x, src)
        tl, tr, bl, br = x.split(x.shape[0] // 4, dim=0)
        top = torch.cat([tl, tr], dim=3)
        bottom = torch.cat([bl, br], dim=3)
        x = torch.cat([top, bottom], dim=2).contiguous()
        return x

    def forward(self, x, src=None):
        if src is None:
            src = x
            x = self.ir(x)
        else:
            assert x.shape[1] == 16

        if self.tile_mode:
            B, C, H, W = x.shape
            if self.scale_factor == 4:
                assert H == 112 and W == H
                if self.training:
                    # randomly use 64x64 2x2 tile and 112x112 1x1 tile
                    if torch.rand((1,)).item() > 0.5:
                        # 112 -> 110
                        x = F.pad(x, (-1,) * 4)
                        src = F.pad(src, (-1,) * 4)
                        return self._forward_tile2x2(x, src)
                    else:
                        return self._forward(x, src)
                else:
                    x = F.pad(x, (-1,) * 4)
                    src = F.pad(src, (-1,) * 4)
                    return self._forward_tile2x2(x, src)
            else:
                raise NotImplementedError()
        else:
            return self._forward(x, src)


def tile_size_validator(size):
    return (size > 16 and
            (size - 16) % 12 == 0 and
            (size - 16) % 16 == 0)


class IRMixIn():
    def has_callback(self):
        # Not used
        # The basic idea of this is,
        # ```
        # intermediate_representation = ir(overall_image)
        # result_image = tiled_infer(intermediate_representation)
        # ```
        # Instead of tiling the image pixels directly,
        # tiling from intermediate feature that use larger receptive field
        return False

    def config_callback(self, x):
        C, H, W = x.shape
        return 16 + 3, H, W, C

    def preprocess_callback(self, rgb, padding):
        batch = True
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
            batch = False
        rgb = replication_pad2d_naive(rgb, padding)
        x = self.unet.ir(rgb)
        x = torch.cat([rgb, x], dim=1)
        if not batch:
            x = x.squeeze(0)
        return x.contiguous()


@register_model
class WincUNet1x(I2IBaseModel):
    name = "waifu2x.winc_unet_1x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=64, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 first_layers=2, last_layers=3,
                 **kwargs):
        super(WincUNet1x, self).__init__(locals(), scale=1, offset=9, in_channels=in_channels, blend_size=4)
        self.register_tile_size_validator(tile_size_validator)
        self.unet = WincUNetBase(in_channels, out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 first_layers=first_layers, last_layers=last_layers,
                                 scale_factor=1)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


@register_model
class WincUNet2x(I2IBaseModel):
    name = "waifu2x.winc_unet_2x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=96, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=4,
                 **kwargs):
        super(WincUNet2x, self).__init__(locals(), scale=2, offset=18, in_channels=in_channels, blend_size=8)
        self.register_tile_size_validator(tile_size_validator)
        self.unet = WincUNetBase(in_channels, out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 scale_factor=2)

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)


@register_model
class WincUNet4x(I2IBaseModel):
    name = "waifu2x.winc_unet_4x"

    def __init__(self, in_channels=3, out_channels=3,
                 base_dim=128, lv1_mlp_ratio=2, lv2_mlp_ratio=1, lv2_ratio=3,
                 **kwargs):
        super(WincUNet4x, self).__init__(locals(), scale=4, offset=36, in_channels=in_channels, blend_size=16)
        self.register_tile_size_validator(tile_size_validator)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet = WincUNetBase(in_channels, out_channels=out_channels,
                                 base_dim=base_dim,
                                 lv1_mlp_ratio=lv1_mlp_ratio, lv2_mlp_ratio=lv2_mlp_ratio, lv2_ratio=lv2_ratio,
                                 scale_factor=4, last_layers=3)

    def forward(self, x):
        if x.shape[1] == 16 + 3:
            src, x = x.split([3, 16], dim=1)
            z = self.unet(x, src)
        else:
            z = self.unet(x)

        if self.training:
            return z
        else:
            return torch.clamp(z, 0., 1.)

    def set_tile_mode(self):
        self.unet.set_tile_mode()

    def to_2x(self, shared=True):
        unet = self.unet if shared else copy.deepcopy(self.unet)
        return WincUNetDownscaled(unet, downscale_factor=2,
                                  in_channels=self.i2i_in_channels, out_channels=self.out_channels)

    def to_1x(self, shared=True):
        unet = self.unet if shared else copy.deepcopy(self.unet)
        return WincUNetDownscaled(unet=unet, downscale_factor=4,
                                  in_channels=self.i2i_in_channels, out_channels=self.out_channels)


def box_resize(x, size):
    H, W = x.shape[2:]
    assert H % size[0] == 0 or W % size[1] == 0 and H > size[0] and W > size[1]
    kernel_h = H // size[0]
    kernel_w = W // size[1]

    # NOTE: Need static kernel_size for export
    assert (kernel_h == kernel_w) and (kernel_h == 2 or kernel_h == 4)
    if kernel_h == 2:
        return F.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
    else:
        return F.avg_pool2d(x, kernel_size=(4, 4), stride=(4, 4))


def resize(x, size, mode, align_corners, antialias):
    if mode == "box":
        return box_resize(x, size=size)
    else:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners, antialias=antialias)


# TODO: Not tested
@register_model
class WincUNetDownscaled(I2IBaseModel):
    name = "waifu2x.winc_unet_downscaled"

    def __init__(self, unet, downscale_factor, in_channels=3, out_channels=3):
        assert downscale_factor in {2, 4}
        offset = {1: 9, 2: 18, 4: 36}[downscale_factor]
        scale = 4 // downscale_factor
        blend_size = 4 * downscale_factor
        self.antialias = True
        super().__init__(dict(in_channels=in_channels, out_channels=out_channels,
                              downscale_factor=downscale_factor),
                         scale=scale, offset=offset, in_channels=in_channels, blend_size=blend_size)
        self.register_tile_size_validator(tile_size_validator)
        self.unet = unet
        self.mode = "bicubic"
        self.antialias = True
        self.downscale_factor = downscale_factor

    def forward(self, x):
        z = self.unet(x)
        if self.training:
            z = resize(z, size=(z.shape[2] // self.downscale_factor, z.shape[3] // self.downscale_factor),
                       mode=self.mode, align_corners=False, antialias=self.antialias)
            return z
        else:
            z = torch.clamp(z, 0., 1.)
            z = resize(z, size=(z.shape[2] // self.downscale_factor, z.shape[3] // self.downscale_factor),
                       mode=self.mode, align_corners=False, antialias=self.antialias)
            z = torch.clamp(z, 0., 1.)
            return z

    @staticmethod
    def from_4x(unet_4x, downscale_factor):
        net = WincUNetDownscaled(unet=copy.deepcopy(unet_4x.unet),
                                 downscale_factor=downscale_factor,
                                 in_channels=unet_4x.unet.in_channels,
                                 out_channels=unet_4x.unet.out_channels)
        return net


register_model_factory(
    "waifu2x.winc_unet_1xs",
    lambda **kwargs: WincUNet1x(base_dim=32, first_layers=1, last_layers=1, lv1_mlp_ratio=1, lv2_mlp_ratio=1, **kwargs))


def _bench(name, compile):
    from nunif.models import create_model
    import time
    device = "cuda:0"
    model = create_model(name, in_channels=3, out_channels=3).to(device).eval()
    if compile:
        model = torch.compile(model)
    x = torch.zeros((4, 3, 256, 256)).to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z, *_ = model(x)
        print(z.shape)
        param = sum([p.numel() for p in model.parameters()])
        print(model.name, model.i2i_offset, model.i2i_scale, f"{param:,}", f"compile={compile}")

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(100):
            z = model(x)
    torch.cuda.synchronize()
    print(time.time() - t)


if __name__ == "__main__":
    enable_full_compile = False

    _bench("waifu2x.winc_unet_1x", enable_full_compile)
    _bench("waifu2x.winc_unet_2x", enable_full_compile)
    _bench("waifu2x.winc_unet_4x", enable_full_compile)
    _bench("waifu2x.winc_unet_1xs", enable_full_compile)
