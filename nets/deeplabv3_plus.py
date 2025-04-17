import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 


#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# #-----------------------------------------#
# class ASPP(nn.Module):
# 	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
# 		super(ASPP, self).__init__()
# 		self.branch1 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch2 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch3 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch4 = nn.Sequential(
# 				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
# 		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
# 		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
# 		self.branch5_relu = nn.ReLU(inplace=True)
#
# 		self.conv_cat = nn.Sequential(
# 				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
# 				nn.BatchNorm2d(dim_out, momentum=bn_mom),
# 				nn.ReLU(inplace=True),
# 		)
#
# 	def forward(self, x):
# 		[b, c, row, col] = x.size()
#         #-----------------------------------------#
#         #   一共五个分支
#         #-----------------------------------------#
# 		conv1x1 = self.branch1(x)
# 		conv3x3_1 = self.branch2(x)
# 		conv3x3_2 = self.branch3(x)
# 		conv3x3_3 = self.branch4(x)
#         #-----------------------------------------#
#         #   第五个分支，全局平均池化+卷积
#         #-----------------------------------------#
# 		global_feature = torch.mean(x,2,True)
# 		global_feature = torch.mean(global_feature,3,True)
# 		global_feature = self.branch5_conv(global_feature)
# 		global_feature = self.branch5_bn(global_feature)
# 		global_feature = self.branch5_relu(global_feature)
# 		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
#
#         #-----------------------------------------#
#         #   将五个分支的内容堆叠起来
#         #   然后1x1卷积整合特征。
#         #-----------------------------------------#
# 		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
# 		result = self.conv_cat(feature_cat)
# 		return result


# # 深度可分离+空洞卷积
# class ASPP(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super(ASPP, self).__init__()
#
#         # ------------------------------------------
#         #   分支1：1x1普通卷积保持不变
#         # ------------------------------------------
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#         # ------------------------------------------
#         #   分支2-4：深度可分离卷积+空洞卷积
#         #   分解为：深度卷积(depthwise) + 逐点卷积(pointwise)
#         # ------------------------------------------
#         def make_aspp_branch(in_channels, out_channels, dilation):
#             return nn.Sequential(
#                 # 深度卷积（分组数=输入通道数）
#                 nn.Conv2d(in_channels, in_channels, 3,
#                           padding=dilation, dilation=dilation,
#                           groups=in_channels, bias=True),  # 关键修改点
#                 nn.BatchNorm2d(in_channels, momentum=bn_mom),
#                 nn.ReLU(inplace=True),
#                 # 逐点卷积调整通道数
#                 nn.Conv2d(in_channels, out_channels, 1, bias=True),
#                 nn.BatchNorm2d(out_channels, momentum=bn_mom),
#                 nn.ReLU(inplace=True),
#             )
#
#         # 创建不同膨胀率的三个分支
#         self.branch2 = make_aspp_branch(dim_in, dim_out, 6 * rate)  # 膨胀率6
#         self.branch3 = make_aspp_branch(dim_in, dim_out, 12 * rate)  # 膨胀率12
#         self.branch4 = make_aspp_branch(dim_in, dim_out, 18 * rate)  # 膨胀率18
#
#         # ------------------------------------------
#         #   分支5：全局特征分支保持不变
#         # ------------------------------------------
#         self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, bias=True)
#         self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
#         self.branch5_relu = nn.ReLU(inplace=True)
#
#         # ------------------------------------------
#         #   最终融合卷积（保持原结构）
#         # ------------------------------------------
#         self.conv_cat = nn.Sequential(
#             nn.Conv2d(dim_out * 5, dim_out, 1, bias=True),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         [b, c, row, col] = x.size()
#
#         # 各分支前向计算
#         conv1x1 = self.branch1(x)
#         conv3x3_1 = self.branch2(x)  # 深度可分离版本
#         conv3x3_2 = self.branch3(x)  # 深度可分离版本
#         conv3x3_3 = self.branch4(x)  # 深度可分离版本
#
#         # 全局特征分支
#         global_feature = torch.mean(x, dim=[2, 3], keepdim=True)
#         global_feature = self.branch5_conv(global_feature)
#         global_feature = self.branch5_bn(global_feature)
#         global_feature = self.branch5_relu(global_feature)
#         global_feature = F.interpolate(global_feature, (row, col), mode='bilinear', align_corners=True)
#
#         # 特征拼接与融合
#         feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
#         result = self.conv_cat(feature_cat)
#         return result



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision.ops import DeformConv2d
#
# # --------------------------------
# # 可变形卷积模块（含Offset生成）
# # --------------------------------
# class DeformableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1):
#         super().__init__()
#         # Offset生成网络
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * 3 * 3,  # 2*K*K (K=3)
#                       kernel_size=3, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(2 * 3 * 3),
#             nn.ReLU(inplace=True)
#         )
#         # 可变形卷积
#         self.deform_conv = DeformConv2d(
#             in_channels, out_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation
#         )
#
#     def forward(self, x):
#         offset = self.offset_conv(x)  # [B, 18, H, W]
#         return self.deform_conv(x, offset)



#
# # --------------------------------
# # 可变形深度可分离卷积模块（修复分支4）
# # --------------------------------
# class DeformableDepthwiseConv(nn.Module):
#     def __init__(self, in_channels, out_channels, dilation=1):
#         super().__init__()
#         # Offset生成网络（适配深度卷积）
#         self.offset_conv = nn.Sequential(
#             nn.Conv2d(in_channels, 2 * 3 * 3*in_channels,  # 2*K*K*C_in
#                       kernel_size=3, padding=dilation, dilation=dilation),
#             nn.BatchNorm2d(2 * 3 * 3*in_channels),
#             nn.ReLU(inplace=True)
#         )
#         # 可变形深度卷积（分组=输入通道）
#         self.deform_conv = DeformConv2d(
#             in_channels, in_channels,
#             kernel_size=3,
#             padding=dilation,
#             dilation=dilation,
#             groups=in_channels  # 深度卷积
#         )
#         # 逐点卷积
#         self.pointwise = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         offset = self.offset_conv(x)
#         x = self.deform_conv(x, offset)
#         return self.pointwise(x)
#
# # 修复后的ASPP模块可变形卷积
# # --------------------------------
# class ASPP_Enhanced(nn.Module):
#     def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
#         super().__init__()
#         # 分支1：普通1x1卷积
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支2：可变形卷积（带Offset）
#         self.branch2 = nn.Sequential(
#             DeformableConv(dim_in, dim_out, dilation=6*rate),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支3：深度可分离+空洞
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, 3, padding=12*rate,
#                      dilation=12*rate, groups=dim_in, bias=False),
#             nn.BatchNorm2d(dim_in, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 分支4：可变形深度可分离
#         self.branch4 = DeformableDepthwiseConv(dim_in, dim_out, dilation=18*rate)
#         # 分支5：全局上下文
#         self.branch5 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(dim_in, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True)
#         )
#         # 特征融合
#         self.fusion = nn.Sequential(
#             nn.Conv2d(5*dim_out, dim_out, 1, bias=False),
#             nn.BatchNorm2d(dim_out, momentum=bn_mom),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5)
#         )
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#         # 各分支前向
#         conv1x1 = self.branch1(x)
#         deform_out = self.branch2(x)
#         depthwise_out = self.branch3(x)
#         deform_depth_out = self.branch4(x)
#         # 全局特征
#         global_feat = self.branch5(x)
#         global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)
#         # 特征融合
#         concat = torch.cat([conv1x1, deform_out, depthwise_out, deform_depth_out, global_feat], dim=1)
#         return self.fusion(concat)
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()

        # --------------------------------
        # 分支1：保持原始1x1卷积
        # --------------------------------
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支2：深度可分离+空洞卷积
        # --------------------------------
        self.branch2 = nn.Sequential(
            # 深度卷积 (padding=6*rate保证输出尺寸不变)
            nn.Conv2d(dim_in, dim_in, 3, padding=6 * rate,
                      dilation=6 * rate, groups=dim_in, bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支3：空间可分离+空洞卷积
        # --------------------------------
        self.branch3 = nn.Sequential(
            # 垂直卷积 (仅高度方向填充)
            nn.Conv2d(dim_in, dim_in, (3, 1),
                      padding=(12 * rate, 0),  # 高度方向填充12*rate
                      dilation=(12 * rate, 1),
                      bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 水平卷积 (仅宽度方向填充)
            nn.Conv2d(dim_in, dim_out, (1, 3),
                      padding=(0, 12 * rate),  # 宽度方向填充12*rate
                      dilation=(1, 12 * rate),
                      bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支4：深度+空间可分离卷积（修复关键）
        # --------------------------------
        self.branch4 = nn.Sequential(
            # 垂直深度卷积 (仅高度方向填充)
            nn.Conv2d(dim_in, dim_in, (3, 1),
                      padding=(18 * rate, 0),  # 高度方向填充18*rate
                      dilation=18 * rate,
                      groups=dim_in,
                      bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            # 水平深度卷积 (仅宽度方向填充)
            nn.Conv2d(dim_in, dim_in, (1, 3),
                      padding=(0, 18 * rate),  # 宽度方向填充18*rate
                      dilation=18 * rate,
                      groups=dim_in,
                      bias=False),
            nn.BatchNorm2d(dim_in, momentum=bn_mom),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 分支5：全局特征分支（保持不变）
        # --------------------------------
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_in, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # --------------------------------
        # 特征融合层
        # --------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 各分支前向传播
        conv1x1 = self.branch1(x)  # [B,C,H,W]
        depthwise_out = self.branch2(x)  # [B,C,H,W]
        spatial_sep_out = self.branch3(x)  # [B,C,H,W]
        depth_spatial_out = self.branch4(x)  # [B,C,H,W]

        # 全局特征上采样
        global_feat = self.branch5(x)  # [B,C,1,1]
        global_feat = F.interpolate(global_feat, (h, w), mode='bilinear', align_corners=True)  # [B,C,H,W]

        # 特征拼接（确保所有分支输出尺寸为 [B,C,H,W]）
        concat_feat = torch.cat([
            conv1x1,
            depthwise_out,
            spatial_sep_out,
            depth_spatial_out,
            global_feat
        ], dim=1)  # 在通道维度拼接 → [B,5C,H,W]

        return self.fusion(concat_feat)  # [B,C,H,W]


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=8):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )		
        # 普通3*3卷积
        # self.cat_conv = nn.Sequential(
        #     nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #
        #     nn.Conv2d(256, 256, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Dropout(0.1),
        # )
        # self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        # 深度空间可分离卷积
        self.cat_conv = nn.Sequential(
            # 第一个融合卷积：空间可分离+深度可分离
            nn.Conv2d(304, 304, (3, 1), padding=(1, 0), groups=304, bias=False),  # 空间可分离-垂直
            nn.BatchNorm2d(304),
            nn.ReLU(inplace=True),
            nn.Conv2d(304, 304, (1, 3), padding=(0, 1), groups=304, bias=False),  # 空间可分离-水平
            nn.BatchNorm2d(304),
            nn.ReLU(inplace=True),
            nn.Conv2d(304, 256, 1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 第二个融合卷积：深度可分离

            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), groups=256, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

