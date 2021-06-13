import torch
import torch.nn as nn

def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


model = nn.Sequential(
    nn.Conv2d(3, 6, 3, 1, 1),
    nn.BatchNorm2d(6)
)

x = torch.randn(10, 3, 24, 24)
output1 = model[0](x)
output2 = model(x)
print(torch.allclose(output1, output2))

model.apply(deactivate_batchnorm)
output1 = model[0](x)
output2 = model(x)
print(torch.allclose(output1, output2))
