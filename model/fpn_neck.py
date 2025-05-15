import torch.nn as nn
import torch.nn.functional as F
from model.fftd_layer import AdaptiveFourierConv2d,TransformerFeatureExtractor
from model.asmf_module import DUC,HDCblock


class FPN(nn.Module):
    '''只针对 resnet50, 101, 152'''

    def __init__(self, features=256, use_p5=True, rank=16, num_heads=8, num_layers=6):
        super(FPN, self).__init__()
        self.rank = rank
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.prj_2 = nn.Conv2d(256, features, kernel_size=1)

        self.conv_5 = AdaptiveFourierConv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = AdaptiveFourierConv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = AdaptiveFourierConv2d(features, features, kernel_size=3, padding=1)
        self.conv_2 = AdaptiveFourierConv2d(features, features, kernel_size=3, padding=1)

        self.transformer_5 = TransformerFeatureExtractor(features, rank, num_heads, num_layers)
        self.transformer_4 = TransformerFeatureExtractor(features, rank, num_heads, num_layers)
        self.transformer_3 = TransformerFeatureExtractor(features, rank, num_heads, num_layers)
        self.transformer_2 = TransformerFeatureExtractor(features, rank, num_heads, num_layers)

        self.cduc1 = DUC(features, features, factor=2)
        self.cduc2 = DUC(features, features, factor=2)
        self.shdc1 = HDCblock(features, features)
        self.shdc2 = HDCblock(features, features)
        self.shdc3 = HDCblock(features, features)

        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def forward(self, x):
        C2, C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        P2 = self.prj_2(C2)

        # FFTD
        P5_low_rank = self.conv_5(P5)
        P4_low_rank = self.conv_4(P4)
        P3_low_rank = self.conv_3(P3)
        P2_low_rank = self.conv_2(P2)

        P5_transformed = self.transformer_5(P5_low_rank)
        P4_transformed = self.transformer_4(P4_low_rank)
        #P3_transformed = self.transformer_3(P3_low_rank)


        P5_fused = P5 * P5_transformed
        P4_fused = P4 * P4_transformed
        P3_fused = P3 * P3_low_rank
        P2_fused = P2 * P2_low_rank

        # ASMF
        UP4 = P4_fused + self.cduc1(P5_fused)
        UP3 = P3_fused + self.cduc2(UP4)

        DP3 = P3_fused + self.shdc1(P2_fused)
        DP4 = P4_fused + self.shdc2(DP3)
        DP5 = P5_fused + self.shdc3(DP4)

        P3 = UP3 + DP3
        P4 = UP4 + DP4
        P5 = DP5 + P5_fused

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))

        return [P3, P4, P5, P6, P7]
