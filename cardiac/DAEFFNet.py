import torch
import torch.nn as nn
from torchvision import transforms,models
import torch.nn.functional as F
import numpy as np
class SUP(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SUP, self).__init__()
        #self.feature=feature
        self.conv1 = nn.Conv2d(inchannel, 1, kernel_size=(1, 1), stride=1, padding=0)
        self.softmax=nn.Softmax(dim=2)
        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=(1, 1), stride=1, padding=0)
        self.ln=nn.LayerNorm([outchannel,1,1])
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outchannel, outchannel, kernel_size=(1, 1), stride=1, padding=0)

    def spatial_pool(self,x):
        batch,channel,height,width=x.size()

        input_x=x
        #print(input_x.shape)
        input_x=input_x.view(batch,channel,height*width)
        #print(input_x.shape)
        input_x=input_x.unsqueeze(1)
        #print(x.shape)
        context_mask=self.conv1(x)
        #print(context_mask.shape)
        context_mask1=context_mask.view(batch,1,height*width)
        context_mask2=self.softmax(context_mask1)
        context_mask3=context_mask2.unsqueeze(-1)
        context=torch.matmul(input_x,context_mask3)
        context=context.view(batch,channel,1,1)
        return context
    def forward(self, high_feature):
        #high_feature=nn.functional.interpolate(high_feature,(self.feature.shape[2],self.feature.shape[3]),mode='bilinear',align_corners=False)
        context_1=self.spatial_pool(high_feature)
        context_2=self.conv2(context_1)
        context_3=self.ln(context_2)
        context_4=self.relu(context_3)
        context_5=self.conv3(context_4)
        return context_5

class ECA(nn.Module):
    def __init__(self, gamma=2,k_size=1):
        super(ECA, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=k_size,padding=(k_size-1)//2,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        N,C,H,W=x.size()
        y=self.avg_pool(x)
        y=self.conv(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        y=self.sigmoid(y)
        return x*y.expand_as(x)


class SS(nn.Module):
    def __init__(self, inchannel,outchannel):
        super(SS, self).__init__()
        self.k_size=3
        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=(7,7),stride=2,padding=3)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(outchannel,outchannel,kernel_size=(3,3),stride=1,padding=1)
        self.eca=ECA(outchannel,k_size=3)
    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        s=self.eca(x)
        return s

def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


class mynet(nn.Module):
    def __init__(self, num_category):
        super(mynet, self).__init__()

        model_ft = models.vgg16(pretrained=True)
        features = list(model_ft.features.children())
        conv1 = nn.Conv2d(3, 64, 3, 1, 100)
        conv1.weight.data = features[0].weight.data
        conv1.bias.data = features[0].bias.data
        features[0] = conv1
        features[4] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[9] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[16] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[23] = nn.MaxPool2d(2, 2, ceil_mode=True)
        features[30] = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv=nn.Sequential(*features[:2])
        self.stage1 = nn.Sequential(*features[2:5])  # ???
        self.ss1=SS(64,64)
        self.stage2=nn.Sequential(*features[5:10])
        self.ss2 = SS(64, 128)

        self.stage3=nn.Sequential(*features[10:17])
        self.ss3 = SS(128, 256)
        self.stage4 = nn.Sequential(*features[17:24])  # ???
        self.ss4 = SS(256, 512)
        self.stage5= nn.Sequential(*features[24:])  # ???
        self.ss5 = SS(512, 512)

        # fc6, fc7a
        fc = list(model_ft.classifier.children())
        fc6 = nn.Conv2d(512, 1024, 7)
        fc7 = nn.Conv2d(1024, 1024, 1)
        fc[0] = fc6
        fc[3] = fc7
        self.fc = nn.Sequential(*fc[:6])

        self.scores1 = nn.Conv2d(1024, num_category*4, 1)  #
        self.scores2 = nn.Conv2d(512, num_category, 1)
        self.scores3 = nn.Conv2d(256, num_category, 1)
        self.scores4 = nn.Conv2d(128, num_category, 1)
        self.scores5 = nn.Conv2d(64, num_category, 1)

        self.sup1=SUP(1024,512)
        self.sup2=SUP(512,256)
        self.sup3=SUP(256,128)
        self.sup4=SUP(128,64)


        for layer in [self.scores1, self.scores2, self.scores3,self.scores4,self.scores5]:
            nn.init.kaiming_normal_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
        self.upsample_32x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_32x.weight.data = bilinear_kernel(num_category, num_category, 4)

        self.upsample_16x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_16x.weight.data = bilinear_kernel(num_category, num_category, 4)

        self.upsample_8x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(num_category, num_category, 4)  # ????? kernel

        self.upsample_4x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(num_category, num_category, 4)  # ????? kernel

        self.upsample_2x = nn.ConvTranspose2d(num_category, num_category, 4, 2, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(num_category, num_category, 4)  # ????? kernel



    def forward(self, x):
        h0=self.conv(x)
        h1 = self.stage1(h0)
        a1=self.ss1(h0)


        g1 = h1+a1  # 1/2

        h2 = self.stage2(g1)
        a2 = self.ss2(g1)

        g2 = h2+a2  # 1/4

        h3 = self.stage3(g2)
        a3 = self.ss3(g2)


        g3 = h3+a3  # 1/8

        h4 = self.stage4(g3)
        a4 = self.ss4(g3)

        g4=h4+a4 #1/16

        h5= self.stage5(g4)
        a5 = self.ss5(g4)
        g5=h5+a5




        h = self.fc(g5)

        s5 = h# 1/32




        s5 = self.scores1(s5)
        up = nn.PixelShuffle(2)
        s5 = up(s5)

        high_feature1 = nn.functional.interpolate(h, (g4.shape[2], g4.shape[3]), mode='bilinear', align_corners=False)
        high_feature1 = self.sup1(high_feature1)
        s4=high_feature1+g4
        s4 = self.scores2(s4 * 1e-2)
        s4 = s4[:, :, 5:5 + s5.size()[2], 5:5 + s5.size()[3]].contiguous()
        s4 = s4 + s5


        s4 = self.upsample_4x(s4)
        high_feature2 = nn.functional.interpolate(g4, (g3.shape[2], g3.shape[3]), mode='bilinear', align_corners=False)
        high_feature2 = self.sup2(high_feature2)
        s3 = high_feature2 + g3
        s3 = self.scores3(s3 * 1e-3)
        s3 = s3[:, :, 9:9 + s4.size()[2], 9:9 + s4.size()[3]].contiguous()
        s3 = s3 + s4

        s3 = self.upsample_8x(s3)
        high_feature3 = nn.functional.interpolate(g3, (g2.shape[2], g2.shape[3]), mode='bilinear', align_corners=False)
        high_feature3 = self.sup3(high_feature3)
        s2 = high_feature3 + g2
        s2 = self.scores4(s2 * 1e-3)
        s2 = s2[:, :, 17:17 + s3.size()[2], 17:17 + s3.size()[3]].contiguous()
        s2=s2+s3

        s2= self.upsample_16x(s2)
        high_feature4 = nn.functional.interpolate(g2, (g1.shape[2], g1.shape[3]), mode='bilinear', align_corners=False)
        high_feature4 = self.sup4(high_feature4)
        s1 = high_feature4 + g1
        s1 = self.scores5(s1 * 1e-4)
        s1 = s1[:, :, 24:24 + s2.size()[2], 24:24 + s2.size()[3]].contiguous()
        s1 = s1+ s2

        s = self.upsample_32x(s1)
        s = s[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return s



if __name__=='__main__':
    print(mynet(3))