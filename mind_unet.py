# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import deform_conv2d
from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers

# @manager.MODELS.add_component
# class MindUNet(nn.Layer):
#     """
#     多通道独立UNet + Transformer融合
    
#     Args:
#         num_classes (int): 输出类别数
#         align_corners (bool): 插值对齐标志
#         use_deconv (bool): 是否使用反卷积
#         in_channels (int): 输入总通道数（此处固定为3）
#         pretrained (str): 预训练路径
#     """
#     def __init__(self, 
#                  num_classes, 
#                  align_corners=False,
#                  use_deconv=False,
#                  in_channels=3,
#                  pretrained=None):
#         super().__init__()
        
#         assert in_channels == 3, "当前实现仅支持3通道输入"
        
#         # 为每个通道构建独立UNet
#         self.unet_c1 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
#         self.unet_c2 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
#         self.unet_c3 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
        
#         # Transformer融合模块 (新增部分)
#         self.trans_fusion = nn.TransformerEncoder(
#             encoder_layer=nn.TransformerEncoderLayer(
#                 d_model=num_classes,
#                 nhead=4,  # 4个注意力头
#                 dim_feedforward=256,
#                 activation='gelu'
#             ),
#             num_layers=2
#         )
        
#         # 位置编码 (新增)
#         self.pos_encoder = nn.Embedding(1000, num_classes)  # 假设最大序列长度1000
        
#         self.pretrained = pretrained
#         self.init_weight()

#     def forward(self, x):
#         # 拆分输入到不同通道 [B,3,H,W] -> 3x [B,1,H,W]
#         c1 = x[:, 0:1, :, :]   # 红色通道
#         c2 = x[:, 1:2, :, :]  # 绿色通道
#         c3 = x[:, 2:3, :, :]  # 蓝色通道
        
#         # 各通道独立前向
#         out_red = self.unet_c1(c1)[0]    # → [B,C,H,W]
#         out_green = self.unet_c2(c2)[0]  # → [B,C,H,W]
#         out_blue = self.unet_c3(c3)[0]   # → [B,C,H,W]
        
#         # 将空间位置转换为序列 (B, C, H, W) → (B, H*W, C)
#         B, C, H, W = out_red.shape
#         red_seq = out_red.reshape(B, C, H*W).transpose([0, 2, 1])  # [B, L, C]
#         green_seq = out_green.reshape(B, C, H*W).transpose([0, 2, 1])
#         blue_seq = out_blue.reshape(B, C, H*W).transpose([0, 2, 1])
        
#         # 拼接三个通道的序列 → [B, 3L, C]
#         fused_seq = paddle.concat([red_seq, green_seq, blue_seq], axis=1)
        
#         # 添加位置编码
#         positions = paddle.arange(fused_seq.shape[1]).unsqueeze(0)  # [1, 3L]
#         fused_seq = fused_seq + self.pos_encoder(positions)
        
#         # Transformer编码
#         fused_out = self.trans_fusion(fused_seq)  # [B, 3L, C]
        
#         # 取红通道对应位置输出 (可改为加权平均)
#         logit_seq = fused_out[:, :H*W, :]  # [B, L, C]
#         logit = logit_seq.transpose([0, 2, 1]).reshape(B, C, H, W)  # [B, C, H, W]
        
#         return [logit]

#     def init_weight(self):
#         if self.pretrained:
#             utils.load_entire_model(self, self.pretrained)
#         else:
#             # 初始化Transformer权重
#             for p in self.trans_fusion.parameters():
#                 if p.dim() > 1:
#                     nn.initializer.XavierNormal(p)
#             nn.initializer.Normal(self.pos_encoder.weight)
            

@manager.MODELS.add_component
class MindUNet(nn.Layer):
    """
    多通道独立UNet，合并输出的实现
    
    Args:
        num_classes (int): 输出类别数
        align_corners (bool): 插值对齐标志
        use_deconv (bool): 是否使用反卷积
        in_channels (int): 输入总通道数（此处固定为3）
        pretrained (str): 预训练路径
    """
    def __init__(self, 
                 num_classes, 
                 align_corners=False,
                 use_deconv=False,
                 in_channels=3,
                 pretrained=None):
        super().__init__()
        
        assert in_channels == 3, "当前实现仅支持3通道输入"
        
        # 为每个通道构建独立UNet
        self.unet_c1 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
        self.unet_c2 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
        self.unet_c3 = UNet(num_classes, align_corners, use_deconv, in_channels=1)
        
        # 融合层：将3*num_classes通道线性合并到num_classes
        self.fusion = nn.Conv2D(
            in_channels=3*num_classes,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        # 拆分输入到不同通道 [B,3,H,W] -> 3x [B,1,H,W]
        c1 = x[:, 0:1, :, :]   # 红色通道
        c2 = x[:, 1:2, :, :] # 绿色通道
        c3 = x[:, 2:3, :, :]  # 蓝色通道
        
        # 各通道独立前向
        out_red = self.unet_c1(c1)[0]    # 取logit_list[0]
        out_green = self.unet_c2(c2)[0]
        out_blue = self.unet_c3(c3)[0]
        
        # # 最大值融合
        # logit = paddle.maximum ( paddle . maximum ( out_red , out_green ) , out_blue )
        
        # 均值/投票
        # logit = ( out_red + out_green + out_blue ) / 3.0
        
        # 通道维度拼接 [B, C, H, W] -> [B, 3*C, H, W]
        fused = paddle.concat([out_red, out_green, out_blue], axis=1)
        
        # 线性融合
        logit = self.fusion(fused)
        return [logit]  # 保持与原结构输出格式兼容

    def init_weight(self):
        if self.pretrained:
            utils.load_entire_model(self, self.pretrained)

'''

# 5层
        up_channels = [
            [1024, 1024, 512],   # 输入1024 → 跳跃512 → 输出512
            [512, 512, 256],    # 输入512 → 跳跃256 → 输出256
            [256, 256, 128],    # 输入256 → 跳跃128 → 输出128
            [128, 128, 64],      # 输入128 → 跳跃64 → 输出64
            [64, 64, 64]        # 输入64 → 跳跃64 → 输出64
        ]
        up_channels = [
            [1024, 1024, 512, True],   # 输入1024 → 跳跃512 → 输出512
            [512, 512, 256, True],    # 输入512 → 跳跃256 → 输出256
            [256, 256, 128, True],    # 输入256 → 跳跃128 → 输出128
            [128, 128, 64, True],      # 输入128 → 跳跃64 → 输出64
            [64, 64, 64, True]        # 输入64 → 跳跃64 → 输出64
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
            [64, 128],    # 下采样1: 64 → 128
            [128, 256],   # 下采样2: 128 → 256
            [256, 512],   # 下采样3: 256 → 512
            [512, 1024]    # 下采样4: 512 → 512
        ]
        up_channels = [
            [1024, 512, 512],   # 输入512 → 跳跃256 → 输出256
            [512, 256, 256],   # 输入256 → 跳跃128 → 输出128
            [256, 128, 128],    # 输入128 → 跳跃64 → 输出64
            [128, 64, 64]      # 输入64 → 跳跃64 → 输出64
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
class MultiScaleBlock(nn.Layer):
    def __init__(self, in_channels, level):
        super().__init__()
        self.level = level
        
        # 为每个level独立定义膨胀率
        dilation_config = {
            0: (24, 16),    # 最浅层：极大感受野
            1: (12, 8),     # 次浅层：中等感受野
            2: (9, 6),     # 次深层：局部增强
            3: (3, 2)      # 最深层：标准卷积
        }
        
        d1, d2 = dilation_config[level]
        padding1 = d1 * (3-1)//2  # 保持特征图尺寸不变
        padding2 = d2 * (3-1)//2
        
        self.branch1 = nn.Conv2D(in_channels, in_channels, 3,
                               dilation=d1, padding=padding1)
        self.branch2 = nn.Conv2D(in_channels, in_channels, 3,
                               dilation=d2, padding=padding2)
        
        # 动态计算融合层输入通道数
        self.fusion = nn.Conv2D(
            in_channels * (3 if level < 2 else 2),  # 前两层保留原始输入
            in_channels, 1
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        if self.level < 2:
            fused = paddle.concat([x, b1, b2], axis=1)
        else:
            fused = paddle.concat([b1, b2], axis=1)
            
        return self.fusion(fused)

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
        
        # 下采样配置（同原代码）
        down_channels = [
            [64, 128, True],    # 第1层
            [128, 256, True],   # 第2层
            [256, 512, True],   # 第3层
            [512, 1024, True]   # 第4层
        ]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(*ch) for ch in down_channels
        ])
        
        # 新增：为每个跳跃连接层级创建独立的多尺度模块
        skip_channels = [64, 128, 256, 512]  # 各层级跳跃连接的通道数
        self.multiscale_blocks = nn.LayerList([
            MultiScaleBlock(ch, level=i) for i, ch in enumerate(skip_channels)
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
        
        for i, block in enumerate(self.down_sample_list):
            # 新增：通过对应层级的MultiScaleBlock处理跳跃连接
            processed_skip = self.multiscale_blocks[i](x)
            short_cuts.append(processed_skip)  # 保存处理后的特征
            x = block(x)
            
        return x, short_cuts

class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False):
        super().__init__()
        self.align_corners = align_corners  # 显式保存参数
        self.use_deconv = use_deconv
        
        # 仅前两层使用可变形卷积
        up_configs = [
            [1024, 512, 512, True],   # 可变形
            [512, 256, 256, True],    # 可变形
            [256, 128, 128, True],    # 普通
            [128, 64, 64, True]       # 普通
        ]
        self.up_sample_list = nn.LayerList([
            UpSampling(*cfg, align_corners, use_deconv) for cfg in up_configs
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