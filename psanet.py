import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

import lib.psa.functional as PF
# import model.resnet as models
# from model.vit import vit_large_patch16


class PSA(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=512, psa_type=2, compact=False, shrink_factor=2, mask_h=59,
                 mask_w=59, normalization_factor=1.0, psa_softmax=True):
        super(PSA, self).__init__()
        assert psa_type in [0, 1, 2]
        self.psa_type = psa_type
        self.compact = compact
        self.shrink_factor = shrink_factor
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.psa_softmax = psa_softmax
        if normalization_factor is None:
            normalization_factor = mask_h * mask_w
        self.normalization_factor = normalization_factor

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mask_h*mask_w, kernel_size=1, bias=False),
        )
        if psa_type == 2:
            self.reduce_p = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True)
            )
            self.attention_p = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mask_h*mask_w, kernel_size=1, bias=False),
            )
        self.proj = nn.Sequential(
            nn.Conv2d(mid_channels * (2 if psa_type == 2 else 1), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x):
        out = x
        if self.psa_type in [0, 1]:
            x = self.reduce(x)
            n, c, h, w = x.size()
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            y = self.attention(x)
            if self.compact:
                if self.psa_type == 1:
                    y = y.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y = PF.psa_mask(y, self.psa_type, self.mask_h, self.mask_w)
            if self.psa_softmax:
                y = F.softmax(y, dim=1)
            x = torch.bmm(x.view(n, c, h * w), y.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.normalization_factor)
        elif self.psa_type == 2:
            x_col = self.reduce(x)
            x_dis = self.reduce_p(x)
            n, c, h, w = x_col.size()
            assert h == self.mask_h and h == 51
            if self.shrink_factor != 1:
                h = (h - 1) // self.shrink_factor + 1
                w = (w - 1) // self.shrink_factor + 1
                x_col = F.interpolate(x_col, size=(h, w), mode='bilinear', align_corners=True)
                x_dis = F.interpolate(x_dis, size=(h, w), mode='bilinear', align_corners=True)
            y_col = self.attention(x_col)
            y_dis = self.attention_p(x_dis)
            if self.compact:
                y_dis = y_dis.view(n, h * w, h * w).transpose(1, 2).view(n, h * w, h, w)
            else:
                y_col = PF.psa_mask(y_col, 0, self.mask_h, self.mask_w)
                y_dis = PF.psa_mask(y_dis, 1, self.mask_h, self.mask_w)
            if self.psa_softmax:
                y_col = F.softmax(y_col, dim=1)
                y_dis = F.softmax(y_dis, dim=1)
            x_col = torch.bmm(x_col.view(n, c, h * w), y_col.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.normalization_factor)
            x_dis = torch.bmm(x_dis.view(n, c, h * w), y_dis.view(n, h * w, h * w)).view(n, c, h, w) * (1.0 / self.normalization_factor)
            x = torch.cat([x_col, x_dis], 1)
        x = self.proj(x)
        if self.shrink_factor != 1:
            h = (h - 1) * self.shrink_factor + 1
            w = (w - 1) * self.shrink_factor + 1
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat((out, x), 1)


class PSANet(nn.Module):
    def __init__(self, encoder, patch_size,
                 layers=50, dropout=0.1, classes=2, zoom_factor=8, use_psa=True, psa_type=2, compact=False,
                 shrink_factor=2, mask_h=59, mask_w=59, normalization_factor=1.0, psa_softmax=True,
                 criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=False):
        super(PSANet, self).__init__()
        assert layers in [50, 101, 152]
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        assert psa_type in [0, 1, 2]
        self.zoom_factor = zoom_factor
        self.use_psa = use_psa
        self.criterion = criterion

        self.use_vit = False

        self.patch_size = patch_size
        self.encoder = encoder

        # if layers == 50:
        #     resnet = models.resnet50(pretrained=pretrained)
        # elif layers == 101:
        #     resnet = models.resnet101(pretrained=pretrained)
        # elif layers == 152:
        #     resnet = models.resnet152(pretrained=pretrained)
        # else:
        #     self.use_vit = True

        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        # self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        # self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        # for n, m in self.layer3.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)
        # for n, m in self.layer4.named_modules():
        #     if 'conv2' in n:
        #         m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
        #     elif 'downsample.0' in n:
        #         m.stride = (1, 1)

        fea_dim = 1024
        if use_psa:
            self.psa = PSA(fea_dim, 512, psa_type, compact, shrink_factor, mask_h, mask_w, normalization_factor, psa_softmax)
            fea_dim *= 2

        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        # if self.training:
        self.aux = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(256, classes, kernel_size=1)
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("psa.", self.psa)
        )
        return nwd_params

    def forward(self, im, ts, is_train=True):
        # print(x.shape)
        # x = F.interpolate(x, size=(448, 448), mode='bilinear', align_corners=True)
        # x1 = x[:, :, :224, :224]
        # x2 = x[:, :, 224:, :224]
        # x3 = x[:, :, :224, 224:]
        # x4 = x[:, :, 224:, 224:]
        # x = self.vit(torch.cat([x1, x2, x3, x4], dim=0))
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=0)
        # x = torch.cat(
        #     [torch.cat([x1, x2], dim=-2),
        #     torch.cat([x3, x4], dim=-2)], dim=-1
        # )
        # x = F.interpolate(x, size=(51, 51), mode='bilinear', align_corners=True)
        h, w = im.size(3), im.size(4)

        x = self.encoder(im, ts, return_features=True)
        # remove CLS/DIST tokens for decoding
        x = x[:, 1:]  # (N, L, D)

        GS = h // self.patch_size
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        x_tmp = x

        x = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=True)
        # h = 401
        # w = 401
        # assert False

        if self.use_psa:
            x = self.psa(x)
        # print(x.shape)
        
        x = self.cls(x)
        
        # print(x.shape)
        # assert False
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        # print(x.shape)
        if is_train:
            aux = self.aux(x_tmp)
            # print(aux.shape)
            # if self.zoom_factor != 1:
            #     aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)

            return x, aux
            # print(aux.shape)
            # main_loss = self.criterion(x, y)
            # aux_loss = self.criterion(aux, y)
            # return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    crop_h = crop_w = 465
    input = torch.rand(4, 3, crop_h, crop_w).cuda()
    compact = False
    mask_h, mask_w = None, None
    shrink_factor = 2
    if compact:
        mask_h = (crop_h - 1) // (8 * shrink_factor) + 1
        mask_w = (crop_w - 1) // (8 * shrink_factor) + 1
    else:
        assert (mask_h is None and mask_w is None) or (mask_h is not None and mask_w is not None)
        if mask_h is None and mask_w is None:
            mask_h = 2 * ((crop_h - 1) // (8 * shrink_factor) + 1) - 1
            mask_w = 2 * ((crop_w - 1) // (8 * shrink_factor) + 1) - 1
        else:
            assert (mask_h % 2 == 1) and (mask_h >= 3) and (mask_h <= 2 * ((crop_h - 1) // (8 * shrink_factor) + 1) - 1)
            assert (mask_w % 2 == 1) and (mask_w >= 3) and (mask_w <= 2 * ((crop_h - 1) // (8 * shrink_factor) + 1) - 1)

    model = PSANet(layers=50, dropout=0.1, classes=21, zoom_factor=8, use_psa=True, psa_type=2, compact=compact,
                   shrink_factor=shrink_factor, mask_h=mask_h, mask_w=mask_w, psa_softmax=True, pretrained=True).cuda()
    print(model)
    model.eval()
    output = model(input)
    print('PSANet', output.size())
