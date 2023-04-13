import torch
import torch.nn as nn
import torchvision.ops as ops
from collections import OrderedDict


class MAIR(nn.Module):

    def __init__(self):
        super(MAIR, self).__init__()

        self.PDFA()
        self.RE()

    def forward(self, features): #B, D, C, H, W
        B, D, C, H, W = features.size()
        center = D // 2
        #DownSample
        features = features.view(B * D, C, H, W)
        feat_l0 = features
        feat_l1 = self.Conv2D_de1(feat_l0)
        feat_l2 = self.Conv2D_de2(feat_l1)

        #Deforrmable Feature Alignment
        #Level 2
        BD, C, H, W = feat_l2.size()  # 1/4 H
        feat_l2 = feat_l2.view(B, D, -1, H, W)
        keyFeat_l2 = feat_l2[:, center:center + 1]  # B, 1, C, H, W
        keyFeat_rep_l2 = keyFeat_l2.repeat(1, D - 1, 1, 1, 1)#B, 1, C, H, W
        refFeat_l2 = torch.cat([feat_l2[:, :center], feat_l2[:, center + 1:]], dim=1)#B, D-1, C, H, W
        ref_key_feat_l2 = torch.cat([refFeat_l2, keyFeat_rep_l2], dim = 2)#B, D-1, 2 * C, H, W
        ref_key_feat_l2 = ref_key_feat_l2.view(B * (D - 1), -1, H, W)

        offsetFeat_l2 = self.Conv2D_l2_offset1(ref_key_feat_l2)
        offset_l2 = self.Conv2D_l2_offset2(offsetFeat_l2)
        x_l2, y_l2, mask_l2 = torch.chunk(offset_l2, 3, dim=1)
        offset_l2 = torch.cat((x_l2, y_l2), dim=1)
        mask_l2 = torch.sigmoid(mask_l2)
        refFeat_l2 = refFeat_l2.view(B * (D - 1), -1, H, W) #B * (D-1), C, H, W
        alignFeat_l2 = self.DeformConv2D_l2(refFeat_l2, offset_l2, mask = mask_l2)  #B * (D-1), C, H, W
        alignFeat_l2 = self.relu(alignFeat_l2)
        up_offsetFeat_l2 = self.upSample(offsetFeat_l2) * 2.0  #B * (D-1), C, H, W
        up_alignFeat_l2 = self.upSample(alignFeat_l2)  #B * (D-1), C, H, W

        #Level 1
        BD, C, H, W = feat_l1.size() #1/4 H
        feat_l1 = feat_l1.view(B, D, -1, H, W)
        keyFeat_l1 = feat_l1[:, center:center + 1]  # B, 1, C, H, W
        keyFeat_rep_l1 = keyFeat_l1.repeat(1, D - 1, 1, 1, 1)  # B, 1, C, H, W
        refFeat_l1 = torch.cat([feat_l1[:, :center], feat_l1[:, center + 1:]], dim=1)  # B, D-1, C, H, W
        ref_key_feat_l1 = torch.cat([refFeat_l1, keyFeat_rep_l1], dim = 2)  # B, D-1, 2 * C, H, W
        ref_key_feat_l1 = ref_key_feat_l1.view(B * (D - 1), -1, H, W)

        offsetFeat_l1 = self.Conv2D_l1_offset1(ref_key_feat_l1)
        offsetFeat_l1 = torch.cat([offsetFeat_l1, up_offsetFeat_l2], dim = 1)
        offsetFeat_l1 = self.Conv2D_l1_offset2(offsetFeat_l1)
        offset_l1 = self.Conv2D_l1_offset3(offsetFeat_l1)
        x_l1, y_l1, mask_l1 = torch.chunk(offset_l1, 3, dim = 1)
        offset_l1 = torch.cat((x_l1, y_l1), dim = 1)
        mask_l1 = torch.sigmoid(mask_l1)
        refFeat_l1 = refFeat_l1.view(B * (D - 1), -1, H, W)  # B * (D-1), C, H, W
        alignFeat_l1 = self.DeformConv2D_l1(refFeat_l1, offset_l1, mask_l1)  # B * (D-1), C, H, W
        alignFeat_l1 = self.relu(alignFeat_l1)
        align_up_Feat_l1 = torch.cat([alignFeat_l1, up_alignFeat_l2], dim = 1)
        fusionAlignFeat_l1 = self.Conv2D_l1_fusion(align_up_Feat_l1)
        up_offsetFeat_l1 = self.upSample(offsetFeat_l1) * 2.0  # B * (D-1), C, H, W
        up_alignFeat_l1 = self.upSample(fusionAlignFeat_l1)  # B * (D-1), C, H, W

        # Level 0
        BD, C, H, W = feat_l0.size()
        feat_l0 = feat_l0.view(B, D, -1, H, W)
        keyFeat_l0 = feat_l0[:, center:center + 1]  # B, 1, C, H, W
        keyFeat_rep_l0 = keyFeat_l0.repeat(1, D - 1, 1, 1, 1)  # B, 1, C, H, W
        refFeat_l0 = torch.cat([feat_l0[:, :center], feat_l0[:, center + 1:]], dim=1)  # B, D-1, C, H, W
        ref_key_feat_l0 = torch.cat([refFeat_l0, keyFeat_rep_l0], dim=2)  # B, D-1, 2 * C, H, W
        ref_key_feat_l0 = ref_key_feat_l0.view(B * (D - 1), -1, H, W)

        offsetFeat_l0 = self.Conv2D_l1_offset1(ref_key_feat_l0)
        offsetFeat_l0 = torch.cat([offsetFeat_l0, up_offsetFeat_l1], dim=1)
        offsetFeat_l0 = self.Conv2D_l1_offset2(offsetFeat_l0)
        offset_l0 = self.Conv2D_l1_offset3(offsetFeat_l0)
        x_l0, y_l0, mask_l0 = torch.chunk(offset_l0, 3, dim=1)
        offset_l0 = torch.cat((x_l0, y_l0), dim=1)
        mask_l0 = torch.sigmoid(mask_l0)
        refFeat_l0 = refFeat_l0.view(B * (D - 1), -1, H, W)  # B * (D-1), C, H, W
        alignFeat_l0 = self.DeformConv2D_l0(refFeat_l0, offset_l0, mask_l0)  # B * (D-1), C, H, W
        alignFeat_l0 = self.relu(alignFeat_l0)
        align_up_Feat_l0 = torch.cat([alignFeat_l0, up_alignFeat_l1], dim=1) # B * (D-1), C, H, W
        fusionAlignFeat_l0 = self.Conv2D_l0_fusion(align_up_Feat_l0) # B * (D-1), C, H, W

        #Reliability Based Feature Fusion
        _, _, C, H, W = keyFeat_l0.size()
        keyFeature = keyFeat_l0  # B, 1, C, H, W
        refFeatures = fusionAlignFeat_l0.view(B, D - 1, -1, H, W)  # B, D-1, C, H, W
        keyFeatures_rep = keyFeature.repeat(1, D-1, 1, 1, 1)
        # print(refFeatures.size(), keyFeatures_rep.size())
        ref_key_cat = torch.cat([refFeatures, keyFeatures_rep], dim=2) #B, (D - 1), 2 * C, H, W
        # print(ref_key_cat.size())
        ref_key_cat = ref_key_cat.view(B * (D - 1), 2 * C, H, W) # B * (D - 1), 2 * C, H, W
        # print(ref_key_cat.size())
        attWeights = self.Conv2D_att(ref_key_cat) # B * (D - 1), 1, H, W
        attWeights = attWeights.view(B, D - 1, 1, H, W).repeat(1, 1, C, 1, 1) #B, (D - 1), C, H, W
        refFeatures = torch.multiply(refFeatures, attWeights)

        finalFeatures = torch.cat([keyFeature, refFeatures], dim=1) #B, D, C, H, W
        finalFeatures = finalFeatures.view(B, D * C, H, W)
        preImg = self.Conv2D_prediction(finalFeatures).squeeze(dim=1) #B, H, W

        return preImg

    def PDFA(self):
        self.PyramidBlock()

    def RE(self):

        self.AttentionBlock()
        self.PreDictionBlock()

    def PyramidBlock(self):
        #levle 0
        self.Conv2D_l0_offset1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l0_offset2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l0_offset3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 27, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))
        # self.DeformConv2D_l0 = nn.Sequential(OrderedDict([
        #     ('deformConv1', ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        # ]))
        self.DeformConv2D_l0 = ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.Conv2D_l0_fusion = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        self.Conv2D_de1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2,2), padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l1_offset1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l1_offset2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l1_offset3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 27, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))
        # self.DeformConv2D_l1 = nn.Sequential(OrderedDict([
        #     ('deformConv1', ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        # ]))
        self.DeformConv2D_l1 = ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.Conv2D_l1_fusion = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))

        self.Conv2D_de2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l2_offset1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
        ]))
        self.Conv2D_l2_offset2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64, 27, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))
        # self.DeformConv2D_l2 = nn.Sequential(OrderedDict([
        #     ('deformConv1', ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        # ]))
        self.DeformConv2D_l2 = ops.DeformConv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upSample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)

    def AttentionBlock(self):

        self.Conv2D_att = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def PreDictionBlock(self):
        self.Conv2D_prediction = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(320, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 1, kernel_size=(3, 3), stride=1, padding=1, bias=False)),
        ]))


if __name__ == "__main__":
    pass