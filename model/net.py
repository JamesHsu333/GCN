import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from model.GCN import GCN
from model.BR import BR

resnet = torchvision.models.resnet152(pretrained=True)

class GCN_Resnet(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_Resnet, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256,self.num_classes) #gcn_i after layer-1
        self.gcn2 = GCN(512,self.num_classes)
        self.gcn3 = GCN(1024,self.num_classes)
        self.gcn4 = GCN(2048,self.num_classes)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm2 = self.br2(self.gcn2(fm2))
        gc_fm3 = self.br3(self.gcn3(fm3))
        gc_fm4 = self.br4(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.interpolate(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br8(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_Resnet_512(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_Resnet_512, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048
        
        self.gcn1 = GCN(256,self.num_classes, k=15) #gcn_i after layer-1
        self.gcn2 = GCN(512,self.num_classes, k=15)
        self.gcn3 = GCN(1024,self.num_classes, k=15)
        self.gcn4 = GCN(2048,self.num_classes, k=15)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)
        self.br7 = BR(num_classes)
        self.br8 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        #print(gc_fm1.size())
        gc_fm2 = self.br2(self.gcn2(fm2))
        #print(gc_fm2.size())
        gc_fm3 = self.br3(self.gcn3(fm3))
        #print(gc_fm3.size())
        gc_fm4 = self.br4(self.gcn4(fm4))
        #print(gc_fm4.size())

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br5(gc_fm3 + gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br6(gc_fm2 + gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm1 = F.interpolate(self.br7(gc_fm1 + gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br8(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)
        #print(out.size())
        return out

class GCN_1(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_1, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn1 = GCN(256, self.num_classes,k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        #fm2 = self.layer2(fm1)
        #fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm1 = F.interpolate(gc_fm1, pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br2(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out


class GCN_2(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_2, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn2 = GCN(512, self.num_classes,k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        #fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm2 = self.br1(self.gcn2(fm2))
        gc_fm2 = F.interpolate(gc_fm2, fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br2(gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br3(gc_fm2), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm3 = F.interpolate(gc_fm3, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br2(gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br3(gc_fm3), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br4(gc_fm3), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_1_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_1_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn1 = GCN(256, self.num_classes,k=55)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        #fm2 = self.layer2(fm1)
        #fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm1 = self.br1(self.gcn1(fm1))
        gc_fm1 = F.interpolate(gc_fm1, pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br2(gc_fm1), input.size()[2:], mode='bilinear', align_corners=True)

        return out


class GCN_2_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_2_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn2 = GCN(512, self.num_classes,k=27)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        #fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm2 = self.br1(self.gcn2(fm2))
        gc_fm2 = F.interpolate(gc_fm2, fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm2 = F.interpolate(self.br2(gc_fm2), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br3(gc_fm2), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        #fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm3 = F.interpolate(gc_fm3, fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br2(gc_fm3), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm3 = F.interpolate(self.br3(gc_fm3), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br4(gc_fm3), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_4(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_4, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)


    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm4 = self.br1(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br2(gc_fm4), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br5(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_4(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=7)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4 + gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_4_C(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4_C, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=7)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(2 * num_classes)
        self.br4 = BR(2 * num_classes)
        self.br5 = BR(2 * num_classes)
        self.br6 = BR(2 * num_classes)

        self.classifier = nn.Conv2d(2 * num_classes, num_classes, 1)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(torch.cat([gc_fm4, gc_fm3], 1)), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)
        out = self.classifier(out)
        return out

class GCN_3_times_4(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_times_4, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=7)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(self.sigmoid(gc_fm4 * gc_fm3)), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_times_4_NoSigmoid(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_times_4_NoSigmoid, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=7)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4 * gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_4_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4 + gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_times_4_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_times_4_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(self.sigmoid(gc_fm4 * gc_fm3)), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_times_4_NoSigmoid_L(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_times_4_NoSigmoid_L, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))
        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4 * gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_4_Linear(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4_Linear, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br3(gc_fm4 * gc_fm3 + gc_fm4 + gc_fm3), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out

class GCN_3_4_Linear_alpha(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4_Linear_alpha, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.alpha = nn.Parameter(torch.FloatTensor(3))
        self.softmax = nn.Softmax()

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        alpha = self.softmax(self.alpha)
        weight_output = (alpha[0] * gc_fm4 * gc_fm3) + (alpha[1] * gc_fm4) + (alpha[2] * gc_fm3)
        gc_fm4 = F.interpolate(self.br3(weight_output), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out


class GCN_3_4_alpha(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes #21 in paper
        super(GCN_3_4_alpha, self).__init__()
        
        self.conv1 = resnet.conv1 # 7x7,64, stride=2 (output_size=112x112)
        self.bn0 = resnet.bn1 #BatchNorm2d(64)
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # maxpool /2 (kernel_size=3, stride=2, padding=1)
        
        self.layer1 = resnet.layer1 #res-2 o/p = 56x56,256
        self.layer2 = resnet.layer2 #res-3 o/p = 28x28,512
        self.layer3 = resnet.layer3 #res-4 o/p = 14x14,1024
        self.layer4 = resnet.layer4 #res-5 o/p = 7x7,2048

        self.gcn3 = GCN(1024,self.num_classes,k=13)
        self.gcn4 = GCN(2048,self.num_classes, k=7)

        self.alpha = nn.Parameter(torch.FloatTensor(1))

        self.br1 = BR(num_classes)
        self.br2 = BR(num_classes)
        self.br3 = BR(num_classes)
        self.br4 = BR(num_classes)
        self.br5 = BR(num_classes)
        self.br6 = BR(num_classes)

        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        pooled_x = x
        x = self.maxpool(x)

        fm1 = self.layer1(x) 
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gc_fm3 = self.br1(self.gcn3(fm3))
        gc_fm4 = self.br2(self.gcn4(fm4))

        gc_fm4 = F.interpolate(gc_fm4, fm3.size()[2:], mode='bilinear', align_corners=True)
        weight_output = (self.sigmoid(self.alpha) * gc_fm4) + ((1-self.sigmoid(self.alpha)) * gc_fm3)
        gc_fm4 = F.interpolate(self.br3(weight_output), fm2.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br4(gc_fm4), fm1.size()[2:], mode='bilinear', align_corners=True)
        gc_fm4 = F.interpolate(self.br5(gc_fm4), pooled_x.size()[2:], mode='bilinear', align_corners=True)
        out = F.interpolate(self.br6(gc_fm4), input.size()[2:], mode='bilinear', align_corners=True)

        return out


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class SoftInvDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftInvDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = torch.sigmoid(logits)
        iflat = 1 - logits.view(-1)
        tflat = 1 - targets.view(-1)
        intersection = (iflat * tflat).sum()
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

loss_fns = {
    'CrossEntropy': loss_fn,
}

class ConfusionMatrix:
    def __init__(self, outputs, labels, class_nums):
        self.outputs = outputs
        self.labels = labels
        self.class_nums = class_nums
    def construct(self):
        self.outputs = self.outputs.flatten()
        self.outputs_count = np.bincount(self.outputs, minlength=self.class_nums)
        self.labels = self.labels.flatten()
        self.labels_count = np.bincount(self.labels, minlength=self.class_nums)

        tmp = self.labels * self.class_nums + self.outputs

        self.cm = np.bincount(tmp, minlength=self.class_nums*self.class_nums)
        self.cm = self.cm.reshape((self.class_nums, self.class_nums))

        self.Nr = np.diag(self.cm)
        self.Dr = self.outputs_count + self.labels_count - self.Nr
        self.Sr = self.outputs_count + self.labels_count
    def mIOU(self):
        iou = self.Nr / self.Dr
        miou = np.nanmean(iou)
        return miou
    def dice(self):
        dice = (2 * self.Nr) / (self.Sr)
        mdice = np.nanmean(dice)
        return mdice

def mIOU(outputs, labels, class_nums):
    m_IOU = 0
    for _, (output, label) in enumerate(zip(outputs, labels)):
        output = output.transpose(1,2,0)
        output = np.argmax(output, axis=2)
        cm = ConfusionMatrix(output, label, class_nums)
        cm.construct()
        m_IOU+=cm.mIOU()
    return m_IOU/len(outputs)

def dice(outputs, labels, class_nums):
    m_dice = 0
    for _, (output, label) in enumerate(zip(outputs, labels)):
        output = output.transpose(1,2,0)
        output = np.argmax(output, axis=2)
        cm = ConfusionMatrix(output, label, class_nums)
        cm.construct()
        m_dice +=cm.dice()
    return m_dice/len(outputs)

metrics = {
    'mIOU': mIOU,
    'dice': dice,
}

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        for gt, pre in zip(gt_image, pre_image):
            pre = np.argmax(pre, axis=0)
            assert gt.shape == pre.shape
            self.confusion_matrix += self._generate_matrix(gt.flatten(), pre.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()