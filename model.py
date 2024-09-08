import torch
import torch.nn as nn
from typing import Union, TypeVar, Tuple, Optional, Callable
from torch.nn.utils.parametrizations import weight_norm
import config

T = TypeVar('T')

num_class = 180 // config.acc

def convnxn(in_planes: int, out_planes: int, kernel_size: Union[T, Tuple[T]], stride: int = 1,
            groups: int = 1, dilation=1) -> nn.Conv1d:
    """nxn convolution and input size equals output size
    O = (I-K+2*P) / S + 1
    """
    if stride == 1:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 1) / 2)  # s = 1, to meet output size equals input size
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    elif stride == 2:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 2) / 2)  # s = 2, to meet output size equals input size // 2
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    else:
        raise Exception('No such stride, please select only 1 or 2 for stride value.')


# ===================== I2CBottleNeck ==========================
class I2CBottleNeck(nn.Module):
    def __init__(
            self,
            in_planes: int,
            stride: int = 1,
            groups: int = 11,
            expansion_rate: int = 3,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        input shape: (B, C, N)
        output shape: (B, C*reduction*expansion, N) = (B, C, N) in out paper
        """
        super(I2CBottleNeck, self).__init__()
        self.in_planes = in_planes
        self.reduction_rate = 1 / 3
        self.groups = groups
        self.expansion_rate = expansion_rate
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.I2CBlock1x1_1 = I2CBlockv2(
            in_planes=in_planes,
            rate=self.reduction_rate,
            intra_kernel_size=1,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        self.bn1 = norm_layer(int(self.in_planes * self.reduction_rate))

        self.I2CBlock3x3 = I2CBlockv2(
            in_planes=int(self.in_planes * self.reduction_rate),
            rate=1,
            intra_kernel_size=3,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        self.bn2 = norm_layer(int(self.in_planes * self.reduction_rate))

        self.I2CBlock1x1_2 = I2CBlockv2(
            in_planes=int(self.in_planes * self.reduction_rate),
            rate=self.expansion_rate,
            intra_kernel_size=1,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        self.bn3 = norm_layer(int(self.in_planes * self.reduction_rate) * self.expansion_rate)

        self.act = nn.SELU(inplace=True)
        self.dropout = nn.Dropout(0.05)

        if stride != 1 or in_planes != int(self.in_planes * self.reduction_rate) * self.expansion_rate:
            self.downsample = nn.Sequential(
                convnxn(in_planes, int(self.in_planes * self.reduction_rate) * self.expansion_rate, kernel_size=1, stride=stride, groups=groups),
                norm_layer(int(self.in_planes * self.reduction_rate) * self.expansion_rate)
            )
        else:
            self.downsample = None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.I2CBlock1x1_1(x)
        out = self.bn1(out)
        out = self.I2CBlock3x3(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.I2CBlock1x1_2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv1d):
                        nn.init.kaiming_normal_(n.weight.data)
                        if n.bias is not None:
                            n.bias.data.zero_()
                    elif isinstance(n, nn.BatchNorm1d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()


class BottleNeck(nn.Module):
    def __init__(
            self,
            in_planes: int,
            stride: int = 1,
            groups: int = 11,
            expansion_rate: int = 3,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(BottleNeck, self).__init__()
        self.reduction_rate = 1 / 3
        self.groups = groups
        self.stride = stride
        self.expansion = expansion_rate
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.conv1x1_1 = convnxn(in_planes, int(in_planes * self.reduction_rate), kernel_size=1, stride=1, groups=groups)
        self.bn1 = norm_layer(int(in_planes * self.reduction_rate))
        self.conv3x3 = convnxn(int(in_planes * self.reduction_rate), int(in_planes * self.reduction_rate), kernel_size=3, stride=stride, groups=groups)
        self.bn2 = norm_layer(int(in_planes * self.reduction_rate))
        self.conv1x1_2 = convnxn(int(in_planes * self.reduction_rate), int(in_planes * self.reduction_rate) * self.expansion, kernel_size=1, stride=1, groups=groups)
        self.bn3 = norm_layer(int(in_planes * self.reduction_rate) * self.expansion)
        self.act = nn.SELU(inplace=True)
        self.dropout = nn.Dropout(0.05)
        if stride != 1 or in_planes != int(in_planes * self.reduction_rate) * self.expansion:
            self.downsample = nn.Sequential(
                convnxn(in_planes, int(in_planes * self.reduction_rate) * self.expansion, kernel_size=1, stride=stride, groups=groups),
                norm_layer(int(in_planes * self.reduction_rate) * self.expansion)
            )
        else:
            self.downsample = None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1x1_1(x)
        out = self.bn1(out)

        out = self.conv3x3(out)
        out = self.bn2(out)

        out = self.dropout(out)

        out = self.conv1x1_2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv1d):
                        nn.init.kaiming_normal_(n.weight.data)
                        if n.bias is not None:
                            n.bias.data.zero_()
                    elif isinstance(n, nn.BatchNorm1d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()


# ===================== I2CBlock ===========================
class I2CBlockv1(nn.Module):

    def __init__(
            self,
            in_planes: int,
            expansion_rate: int = 1,
            intra_kernel_size: int = 3,
            inter_kernel_size: int = 1,
            stride: int = 1,
            groups: int = 10,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ac_flag: bool = True,
    ):
        """
        input size: [B, C, N]
        output size: [B, e*(C+1), N]
        """
        super(I2CBlockv1, self).__init__()
        self.C = in_planes
        self.intra_kernel_size = intra_kernel_size
        self.inter_kernel_size = inter_kernel_size
        self.groups = groups
        self.e = expansion_rate
        self.flag = ac_flag
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(self.C / self.groups)

        self.inter_conv = convnxn(self.C, self.group_width * self.e, kernel_size=self.inter_kernel_size, stride=stride,
                                  groups=1)
        self.intra_conv = convnxn(self.C, self.C * self.e, kernel_size=self.intra_kernel_size, stride=stride,
                                  groups=groups)
        if self.flag:
            self.bn = norm_layer((self.group_width + self.C) * self.e)
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward(self, x):

        inter_output = self.inter_conv(x)
        intra_output = self.intra_conv(x)
        # print(inter_output.shape)
        # print(inter_output.shape)
        out = torch.cat((intra_output, inter_output), 1)
        if self.flag:
            out = self.bn(out)
            out = self.act(out)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class I2CBlockv2(nn.Module):

    def __init__(
            self,
            in_planes: int,
            expansion_rate: Union[int, float] = 1,
            stride: int = 1,
            intra_kernel_size: int = 3,
            inter_kernel_size: int = 1,
            groups: int = 10 + 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ac_flag: bool = True,
    ):
        """
        input size: [B, C, N]
        output size: [B, e*C, N]
        """
        super(I2CBlockv2, self).__init__()
        self.groups = groups
        self.e = expansion_rate
        self.intra_kernel_size = intra_kernel_size
        self.inter_kernel_size = inter_kernel_size
        self.ac_flag = ac_flag

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(in_planes / self.groups)

        self.inter_channel1 = convnxn(self.group_width * (self.groups - 1), int(self.group_width * self.e),
                                      kernel_size=self.inter_kernel_size, stride=stride, groups=1)
        self.inter_channel2 = convnxn(int(self.group_width * (self.e + 1)), int(self.group_width * self.e),
                                      kernel_size=self.inter_kernel_size, stride=stride, groups=1)
        self.intra_channel = convnxn(int(self.group_width * (self.groups - 1)),
                                     int((self.group_width * (self.groups - 1)) * self.e),
                                     kernel_size=self.intra_kernel_size, stride=stride, groups=self.groups - 1)
        if self.ac_flag:
            self.bn = norm_layer(int((self.group_width * (self.groups - 1)) * self.e + self.group_width * self.e))
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward(self, x):
        intra_data_prev = x[:, :(self.groups - 1) * self.group_width, :]
        inter_data_prev = x[:, (self.groups - 1) * self.group_width:, :]

        inter_data_current1 = self.inter_channel1(intra_data_prev) # 注意，这里输入居然是intra prev，这意味着后续intra数据间也被卷积了
        inter_data_current2 = torch.cat((inter_data_prev, inter_data_current1), 1)
        inter_data_current = self.inter_channel2(inter_data_current2)

        intra_data_current = self.intra_channel(intra_data_prev)
        # print(inter_data_current.shape)
        # print(intra_data_current.shape)
        output = torch.cat((intra_data_current, inter_data_current), 1)

        if self.ac_flag:
            output = self.bn(output)
            output = self.act(output)

        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ===================== I2CMSE ==========================
class I2CMSE(nn.Module):

    def __init__(
            self,
            in_planes: int,
            groups: int = 11,
            b1_size: int = 5,
            b2_size: int = 11,
            b3_size: int = 21,
            expansion_rate: int = 2,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(I2CMSE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.in_planes = in_planes
        self.groups = groups
        self.group_width = int(in_planes / groups)
        self.expansion = expansion_rate
        self.b1_size = b1_size
        self.b2_size = b2_size
        self.b3_size = b3_size

        self.branch1_1 = I2CBlockv2(
            in_planes=in_planes,
            expansion_rate=self.expansion,
            intra_kernel_size=self.b1_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        # 这个branch结束后，为34个通道，前32个通道属于intra，后2通道属于inter
        self.branch1_2 = I2CBlockv2(
            in_planes=in_planes * self.expansion,
            expansion_rate=1,
            intra_kernel_size=self.b1_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=True
        )

        self.branch2_1 = I2CBlockv2(
            in_planes=in_planes,
            expansion_rate=self.expansion,
            intra_kernel_size=self.b2_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        self.branch2_2 = I2CBlockv2(
            in_planes=in_planes * self.expansion,
            expansion_rate=1,
            intra_kernel_size=self.b2_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=True
        )

        self.branch3_1 = I2CBlockv2(
            in_planes=in_planes,
            expansion_rate=self.expansion,
            intra_kernel_size=self.b3_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=False
        )
        self.branch3_2 = I2CBlockv2(
            in_planes=in_planes * self.expansion,
            expansion_rate=1,
            intra_kernel_size=self.b3_size,
            inter_kernel_size=1,
            stride=1,
            groups=self.groups,
            ac_flag=True
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branch1 = self.branch1_1(x)
        branch1_out = self.branch1_2(branch1)
        # print('branch1_out', branch1_out.shape)
        branch2 = self.branch2_1(x)
        branch2_out = self.branch2_2(branch2)
        # print('branch2_out', branch1_out.shape)

        branch3 = self.branch3_1(x)
        branch3_out = self.branch3_2(branch3)
        # print('branch3_out', branch3_out.shape)

        outputs = [
            torch.cat([branch1_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch2_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch3_out[:,
                       int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :]],
                      1)
            for i in range(self.groups)]

        out = torch.cat(outputs, 1)
        # print('out', out.shape)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# ===================== I2CAttention ==========================
class I2CSA(nn.Module):

    def __init__(
            self,
            channels: int,
            reduction: int = 4,
            groups: int = 11,
    ) -> None:
        super(I2CSA, self).__init__()
        self.group_width = int(channels // groups)
        self.groups = groups
        self.reduction = reduction
        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(self.channels, self.channels // self.reduction * self.groups, kernel_size=1, groups=self.groups, bias=False)
        self.ac = nn.ReLU()
        self.fc2 = nn.Conv1d(self.channels // self.reduction * self.groups, self.channels, kernel_size=1, groups=self.groups, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)

        avg_out = self.fc2(self.ac(self.fc1(y1)))
        max_out = self.fc2(self.ac(self.fc1(y2)))
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y.expand_as(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class I2CTA(nn.Module):

    def __init__(
            self,
            channels: int,
            groups: int = 11,
    ) -> None:
        super(I2CTA, self).__init__()
        self.group_width = int(channels // groups)
        self.groups = groups
        self.spatialAttConv = convnxn(2 * self.groups, self.groups, kernel_size=3, groups=self.groups)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()
        out = []
        for id in range(self.groups):
            data = x[:, id * self.group_width:(id+1) * self.group_width, :]
            avg_out = torch.mean(data, dim=1, keepdim=True)
            max_out, _ = torch.max(data, dim=1, keepdim=True)
            out_ = torch.cat([avg_out, max_out], dim=1)
            out.append(out_)

        out = torch.cat(out, 1)
        out = self.spatialAttConv(out)
        out = self.sigmoid(out)
        out = torch.cat([out[:, i].unsqueeze(1).expand((b, self.group_width, l)) for i in range(self.groups)], 1)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class I2CAttention(nn.Module):

    def __init__(
            self,
            in_planes: int,
            reduction: int,
            groups: int,
    ) -> None:
        super(I2CAttention, self).__init__()

        self.spatial_attention = I2CSA(in_planes, reduction, groups)
        #self.temporal_attention = I2CTA(in_planes)

    def forward(self, x):
        out = self.spatial_attention(x)
        #out = self.temporal_attention(out)
        return out



class IntraInterBlock(nn.Module):
    def __init__(self, in_planes, groups, decay):
        super(IntraInterBlock, self).__init__()
        self.groups = groups
        self.group_width = int(in_planes // (groups))
        self.decay = decay
        self.intra_conv = nn.Conv1d(self.group_width * (groups - 1), int(self.group_width * (groups - 1) * self.decay), kernel_size=3, padding=1, groups=(groups - 1))
        self.inter_conv1 = nn.Conv1d(self.group_width * (groups - 1), self.group_width, kernel_size=1)
        self.inter_conv2 = nn.Conv1d(self.group_width * 2, int(self.group_width * self.decay), kernel_size=1)
        self.intra_attention = nn.MultiheadAttention(embed_dim=int(in_planes * decay * 16 // 17),num_heads=18) 
        self.inter_attention = nn.MultiheadAttention(embed_dim=int(in_planes  // 17), num_heads=18) 
        self.selu = nn.SELU(inplace=True)



    def forward(self, x):
        batch_size, num_channels, seq_len = x.size()

        # Split into intra and inter groups
        intra_data_prev = x[:, :(self.groups - 1) * self.group_width, :]
        inter_data_prev = x[:, (self.groups - 1) * self.group_width:, :]
        inter_data_current1 = self.inter_conv1(intra_data_prev)
        inter_data_current1 = inter_data_current1.permute(2, 0, 1)
        inter_data_current1, _ = self.inter_attention(inter_data_current1, inter_data_current1, inter_data_current1)
        inter_data_current1 = inter_data_current1.permute(1, 2, 0)

        inter_data_current2 = torch.cat((inter_data_prev, inter_data_current1), 1)
        inter_data_current = self.inter_conv2(inter_data_current2)

        intra_data_current = self.intra_conv(intra_data_prev)
        intra_data_current = intra_data_current.permute(2, 0, 1)
        intra_data_current, _ = self.intra_attention(intra_data_current, intra_data_current, intra_data_current)
        intra_data_current = intra_data_current.permute(1, 2, 0)

        output = torch.cat((intra_data_current, inter_data_current), 1)
        output = self.selu(output)

        return output
    
class I2CNet(nn.Module):
    def __init__(self, in_planes=11, num_classes=52, mse_b1=5, mse_b2=11, mse_b3=21,
                 expansion_rate=2, decay=0.5, reduction_rate=4, block1_num=1, block2_num=1,
                 norm_layer=None, window_size=config.window_size, num_outputs=5):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.groups = in_planes
        self.mse_b1 = mse_b1
        self.mse_b2 = mse_b2
        self.mse_b3 = mse_b3
        self.mse_expansion = expansion_rate
        self.decay = decay
        self.attention_reduction = reduction_rate
        self.window_size = window_size
        self.num_outputs = num_outputs

        self.conv1 = I2CBlockv1(
            in_planes=in_planes,
            expansion_rate=1,
            intra_kernel_size=3,
            inter_kernel_size=1,
            groups=self.groups
        )
        self.in_planes = in_planes + 1
        self.groups += 1

        self.layer1 = self._make_blocks(nums=block1_num)
        self.layer2 = self._make_blocks(nums=block2_num)
        self.i2cblock1 = self._head_blocks(nums=1)
        i1_channels = self.in_planes
        self.i2cblock2 = self._head_blocks(nums=1)
        i2_channels = self.in_planes
        self.combine_and_classify = nn.Sequential(
            nn.Conv1d(in_channels=i1_channels + i2_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SELU(),
            nn.Conv1d(in_channels=128, out_channels= num_class * num_outputs, kernel_size=1, stride=1, padding=0)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(window_size // config.width) 

    def _make_blocks(self, nums):
        layers = []
        for i in range(nums):
            layers.append(I2CMSE(
                in_planes=self.in_planes,
                groups=self.groups,
                b1_size=self.mse_b1,
                b2_size=self.mse_b2,
                b3_size=self.mse_b3,
                expansion_rate=self.mse_expansion
            ))
            self.in_planes = self.in_planes * self.mse_expansion * 3
            layers.append(BottleNeck(
                in_planes=self.in_planes,
                groups=self.groups
            ))

        return nn.Sequential(*layers)
    
    def _head_blocks(self, nums):
        layers = []
        for i in range(nums):
            layers.append(IntraInterBlock(in_planes=self.in_planes,groups=self.groups,decay=self.decay))
            self.in_planes = int(self.in_planes * self.decay)

        return nn.Sequential(*layers)
    

    def _forward_imp(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out_i1 = self.i2cblock1(out)
        out_i2 = self.i2cblock2(out_i1)
        out = torch.cat([out_i1, out_i2], dim=1)
        out = self.combine_and_classify(out)
        if config.width != 1:
            out = self.avg_pool(out)
        out = out.view(out.size(0), self.num_outputs, num_class, self.window_size // config.width)
        return out

    def forward(self, x):
        return self._forward_imp(x)
    
    
    # 调用示例代码
batch_size = 4
input_tensor = torch.randn(batch_size, 16, 256)

# 创建模型实例
model = I2CNet(in_planes=16)

# 运行模型并检查输出形状
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # 应该输出 (batch_size, num_outputs, window_size)

# 下一步，根据两个group的数据设计block，block应该同时考虑通道内和通道间数据