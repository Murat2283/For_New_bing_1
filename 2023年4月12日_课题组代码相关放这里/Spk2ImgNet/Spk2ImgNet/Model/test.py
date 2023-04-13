import torch.nn as nn
from Model.DeformConv2d import DeformConv2d
import torch
from torchvision.ops.deform_conv import deform_conv2d
from collections import OrderedDict

DeformConv2D_l2 = DeformConv2d(3, 5, 3, stride=1, padding=1, bias=False)
# DeformConv2D_l2 = nn.Sequential(OrderedDict([
#             ('deformConv1', DeformConv2d(3, 5, 3, stride=1, padding=1, bias=False)),
#         ]))
input = torch.rand(4, 3, 10, 10)
kh, kw = 3, 3
weight = torch.rand(5, 3, kh, kw)
# offset and mask should have the same spatial size as the output
# of the convolution. In this case, for an input of 10, stride of 1
# and kernel size of 3, without padding, the output size is 8
offset = torch.rand(4, 2 * kh * kw, 10, 10)
mask = torch.rand(4, kh * kw, 10, 10)
# out = deform_conv2d(input, offset, weight, mask=mask)
out = DeformConv2D_l2(input, offset, mask=mask)
print(out.shape)


