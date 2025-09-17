import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import deform_conv2d
from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers

# 仅采用deconv的unet

'''

# 5层
        up_channels = [
            [1024, 1024, 512],   # 输入1024 → 跳跃512 → 输出512
            [512, 512, 256],    # 输入512 → 跳跃256 → 输出256
            [256, 256, 128],    # 输入256 → 跳跃128 → 输出128
            [128, 128, 64],      # 输入128 → 跳跃64 → 输出64
            [64, 64, 64]        # 输入64 → 跳跃64 → 输出64
        ]
        down_channels = [
            [64, 128],    # 下采样1
            [128, 256],   # 下采样2
            [256, 512],   # 下采样3
            [512, 1024],  # 下采样4
            [1024, 1024]
        ]
        
# 4层下采样（最后一层保持512→512）
        down_channels = [
            [64, 128, True],    # 下采样1: 64 → 128
            [128, 256, True],   # 下采样2: 128 → 256
            [256, 512, True]    # 下采样3: 256 → 512
            [512, 512, True]    # 下采样4: 512 → 512
        ]
        up_channels = [
            [512, 512, 256, True],
            [256, 256, 128, True],   # 输入512 → 跳跃128 → 输出128
            [128, 128, 64, True],    # 输入128 → 跳跃64 → 输出64
            [64, 64, 64, True]      # 输入64 → 跳跃64 → 输出64
        ]

# 3层下采样（最后一层256→512）
down_channels = [
    [64, 128],    # 下采样1: 64 → 128
    [128, 256],   # 下采样2: 128 → 256
    [256, 256]    # 下采样3: 256 → 512
]

# 3层上采样（最后一层输出64 → 64）
up_channels = [
    [256, 256, 128],   # 输入512 → 跳跃128 → 输出128
    [128, 128, 64],    # 输入128 → 跳跃64 → 输出64
    [64, 64, 64]      # 输入64 → 跳跃64 → 输出64
]


'''


class DeformableConv2D(nn.Layer):
    """可变形卷积模块（包含BN+ReLU）"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # 偏移量生成层
        self.offset_conv = nn.Conv2D(
            in_channels,
            2 * kernel_size * kernel_size,  # 每个采样点学习x/y偏移
            kernel_size=kernel_size,
            padding=(kernel_size-1)//2
        )
        # 主卷积参数（不直接使用，仅用于获取weight/bias）
        self.conv = nn.Conv2D(
            in_channels, out_channels, 
            kernel_size, 
            padding=(kernel_size-1)//2
        )
        # BN+ReLU层
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 生成偏移量 [B, 2*K*K, H, W]
        offset = self.offset_conv(x)
        
        # 应用可变形卷积
        x = deform_conv2d(
            x, 
            offset, 
            self.conv.weight,  # 直接访问卷积权重
            self.conv.bias,    # 直接访问卷积偏置
            padding=(self.conv._kernel_size[0]-1)//2
        )
        # BN+ReLU
        return self.relu(self.bn(x))


@manager.MODELS.add_component
class UNet(nn.Layer):
    def __init__(self, num_classes, align_corners=False, use_deconv=False, in_channels=3, pretrained=None):
        super().__init__()
        self.encode = Encoder(in_channels)
        self.decode = Decoder(align_corners, use_deconv)
        self.cls = nn.Conv2D(64, num_classes, kernel_size=1)
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        return [logit]

    def init_weight(self):
        if self.pretrained:
            utils.load_entire_model(self, self.pretrained)

class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, 64, 3),
            layers.ConvBNReLU(64, 64, 3)
        )
        # 修改下采样配置（仅最后两层使用可变形卷积）
        down_channels = [
            [64, 128, True],    # 下采样1: 64 → 128
            [128, 256, True],   # 下采样2: 128 → 256
            [256, 512, False],    # 下采样3: 256 → 512
            [512, 512, False]    # 下采样4: 512 → 512
        ]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(*ch) for ch in down_channels
        ])

    def down_sampling(self, in_ch, out_ch, use_deform=False):
        layers1 = [
            nn.MaxPool2D(kernel_size=2, stride=2),
            DeformableConv2D(in_ch, out_ch) if use_deform else layers.ConvBNReLU(in_ch, out_ch, 3),
            layers.ConvBNReLU(out_ch, out_ch, 3)
        ]
        return nn.Sequential(*layers1)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for block in self.down_sample_list:
            short_cuts.append(x)  # 保存下采样前的特征
            x = block(x)
        return x, short_cuts

class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()
        self.align_corners = align_corners  # 显式保存参数
        self.use_deconv = use_deconv
        
        up_channels = [
            [512, 512, 256, True],
            [256, 256, 128, True],   # 输入512 → 跳跃128 → 输出128
            [128, 128, 64, False],    # 输入128 → 跳跃64 → 输出64
            [64, 64, 64, False]      # 输入64 → 跳跃64 → 输出64
        ]
        self.up_sample_list = nn.LayerList([
            UpSampling(*cfg, align_corners, use_deconv) for cfg in up_channels
        ])

    def forward(self, x, skips):
        for i, block in enumerate(self.up_sample_list):
            x = block(x, skips[-(i+1)])  # 逆序使用跳跃连接
        return x

class UpSampling(nn.Layer):
    def __init__(self, in_channels, shortcut_channels, out_channels, 
                 use_deform, align_corners, use_deconv=False):
        super().__init__()
        self.align_corners = align_corners
        
        # 上采样部分保持原样
        if use_deconv:
            self.deconv = nn.Conv2DTranspose(in_channels, out_channels//2, 2, stride=2)
            in_ch_after = out_channels//2 + shortcut_channels
        else:
            self.deconv = None
            in_ch_after = in_channels + shortcut_channels
        
        # 双卷积层第一层使用可变形卷积
        conv1 = DeformableConv2D(in_ch_after, out_channels) if use_deform \
                else layers.ConvBNReLU(in_ch_after, out_channels, 3)
                
        self.double_conv = nn.Sequential(
            conv1,
            layers.ConvBNReLU(out_channels, out_channels, 3)
        )

    def forward(self, x, short_cut):
        # 保持原有前向传播逻辑
        if self.deconv:
            x = self.deconv(x)
        else:
            x = F.interpolate(
                x, short_cut.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
        x = paddle.concat([x, short_cut], axis=1)
        return self.double_conv(x)