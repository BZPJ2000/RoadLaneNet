import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义h_sigmoid激活函数
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

# 定义h_swish激活函数
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# 定义CoordAttention模块
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

# 定义SAFM模块
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super(SAFM, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for _ in range(n_levels)])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = torch.cat(out, dim=1)
        out = self.aggr(out)
        out = self.act(out) * x
        return out


class PPM(nn.Module):
    def __init__(self, in_channels, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, down_dim, 3, padding=1),
            nn.GroupNorm(1, down_dim),  # 使用 GroupNorm
            nn.PReLU()
        )

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.GroupNorm(1, down_dim),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.GroupNorm(1, down_dim),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(3, 3)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.GroupNorm(1, down_dim),
            nn.PReLU()
        )
        self.conv4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),
            nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.GroupNorm(1, down_dim),
            nn.PReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(4 * down_dim, down_dim, kernel_size=1),
            nn.GroupNorm(1, down_dim),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.interpolate(conv1, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv2_up = F.interpolate(conv2, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv3_up = F.interpolate(conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        conv4_up = F.interpolate(conv4, size=x.size()[2:], mode='bilinear', align_corners=True)
        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))


class ContinusParalleConv(nn.Module):
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(1024 + 512, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(512 + 256 + 256, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(512 + 256, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(256 + 128, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(256 + 128 + 128, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(256 + 128 + 128 + 128, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(128 + 64, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(128 + 64 + 64, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(128 + 64 + 64 + 64, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(128 + 64 + 64 + 64 + 64, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.coord_att_0 = CoordAtt(64, 64)
        self.coord_att_1 = CoordAtt(128, 128)
        self.coord_att_2 = CoordAtt(256, 256)
        self.coord_att_3 = CoordAtt(512, 512)
        self.coord_att_4 = CoordAtt(1024, 1024)

        self.safm_0 = SAFM(64)
        self.safm_1 = SAFM(128)
        self.safm_2 = SAFM(256)
        self.safm_3 = SAFM(512)
        self.safm_4 = SAFM(1024)

        self.ppm_0 = PPM(64, 64)
        self.ppm_1 = PPM(128, 128)
        self.ppm_2 = PPM(256, 256)
        self.ppm_3 = PPM(512, 512)
        self.ppm_4 = PPM(1024, 1024)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)



        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_0_0 = self.coord_att_0(x_0_0)
        x_0_0 = self.safm_0(x_0_0)
        x_0_0 = self.ppm_0(x_0_0)  # 应用 PPM

        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_1_0 = self.coord_att_1(x_1_0)
        x_1_0 = self.safm_1(x_1_0)
        x_1_0 = self.ppm_1(x_1_0)  # 应用 PPM

        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_2_0 = self.coord_att_2(x_2_0)
        x_2_0 = self.safm_2(x_2_0)
        x_2_0 = self.ppm_2(x_2_0)  # 应用 PPM

        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_3_0 = self.coord_att_3(x_3_0)
        x_3_0 = self.safm_3(x_3_0)
        x_3_0 = self.ppm_3(x_3_0)  # 应用 PPM

        x_4_0 = self.stage_4(self.pool(x_3_0))
        x_4_0 = self.coord_att_4(x_4_0)
        x_4_0 = self.safm_4(x_4_0)
        x_4_0 = self.ppm_4(x_4_0)  # 应用 PPM

        x_0_1 = torch.cat([F.interpolate(x_1_0, size=x_0_0.shape[2:], mode='bilinear', align_corners=True), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([F.interpolate(x_2_0, size=x_1_0.shape[2:], mode='bilinear', align_corners=True), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([F.interpolate(x_3_0, size=x_2_0.shape[2:], mode='bilinear', align_corners=True), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([F.interpolate(x_4_0, size=x_3_0.shape[2:], mode='bilinear', align_corners=True), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([F.interpolate(x_3_1, size=x_2_0.shape[2:], mode='bilinear', align_corners=True), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([F.interpolate(x_2_1, size=x_1_0.shape[2:], mode='bilinear', align_corners=True), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([F.interpolate(x_2_2, size=x_1_0.shape[2:], mode='bilinear', align_corners=True), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([F.interpolate(x_1_1, size=x_0_0.shape[2:], mode='bilinear', align_corners=True), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([F.interpolate(x_1_2, size=x_0_0.shape[2:], mode='bilinear', align_corners=True), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([F.interpolate(x_1_3, size=x_0_0.shape[2:], mode='bilinear', align_corners=True), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)
# 这里好像没有使用，你要在forward里面应用上去

if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cpu')
    inputs = torch.randn((1, 3, 224, 224)).to(device)
    model = UnetPlusPlus(num_classes=8, deep_supervision=deep_supervision).to(device)
    model.eval()  # 将模型设置为评估模式
    print(model)
    outputs = model(inputs)
    print(outputs.shape)

    # print("deep_supervision: True")
    # deep_supervision = True
    # model = UnetPlusPlus(num_classes=2, deep_supervision=deep_supervision).to(device)
    # outputs = model(inputs)
    # for out in outputs:
    #     print(out.shape)
