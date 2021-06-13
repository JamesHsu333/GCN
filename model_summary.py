import model.net as net

import torch
import torch.nn as nn
from torchsummary import summary
"""
model = net.GCN_Resnet(20+1).cuda()
summary(model, (3, 224, 224))

model_3_4 = net.GCN_3_4(20+1).cuda()
summary(model_3_4, (3, 224, 224))

model_3_times_4 = net.GCN_3_4(20+1).cuda()
summary(model_3_times_4, (3, 224, 224))

model_3_4_L = net.GCN_3_4_L(20+1).cuda()
summary(model_3_4_L, (3, 224, 224))

model_3_times_4_L = net.GCN_3_4_L(20+1).cuda()
summary(model_3_times_4_L, (3, 224, 224))

model_3_times_4_NoSigmoid = net.GCN_3_times_4_NoSigmoid(20+1).cuda()
summary(model_3_times_4_NoSigmoid, (3, 224, 224))

model_3_times_4_NoSigmoid_L = net.GCN_3_times_4_NoSigmoid_L(20+1).cuda()
summary(model_3_times_4_NoSigmoid_L, (3, 224, 224))

model_3_4_Linear = net. GCN_3_4_Linear(20+1).cuda()
summary(model_3_4_Linear, (3, 224, 224))

model_3_4_Linear_alpha = net.GCN_3_4_Linear_alpha(20+1).cuda()
summary(model_3_4_Linear_alpha, (3, 224, 224))

model_3_4_alpha = net.GCN_3_4_alpha(20+1).cuda()
summary(model_3_4_alpha, (3, 224, 224))

model_3_4_C = net.GCN_3_4_C(20+1).cuda()
summary(model_3_4_C, (3, 224, 224))
"""

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

model = net.GCN_Resnet_512(20+1).cuda()
model.apply(deactivate_batchnorm)
summary(model, (3, 512, 512))