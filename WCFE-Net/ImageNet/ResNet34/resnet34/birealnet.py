import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['birealnet18', 'birealnet34']


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        a = input
        w = self.weight

        # * binarize
        bw = BinaryQuantize().apply(w)
        ba = BinaryQuantize_a().apply(a)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)

        return output


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(input),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        # binary_weights = scaling_factor * torch.sign(input)
        binary_weights = torch.sign(input)

        return binary_weights

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = grad_output.clone()
        grad_input[input.ge(1) | input.le(-1)] = 0

        return grad_input


class BinaryQuantize_a(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        grad_input = (2 - torch.abs(2 * input))
        grad_input = grad_input.clamp(min=0) * grad_output.clone()
        return grad_input


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

#
# class HardBinaryConv(nn.Module):
#     def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
#         super(HardBinaryConv, self).__init__()
#         self.stride = stride
#         self.padding = padding
#         self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
#         self.shape = (out_chn, in_chn, kernel_size, kernel_size)
#         self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
#
#     def forward(self, x):
#         real_weights = self.weights.view(self.shape)
#         scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
#         #print(scaling_factor, flush=True)
#         scaling_factor = scaling_factor.detach()
#         binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
#         cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
#         binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
#         #print(binary_weights, flush=True)
#         y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)
#
#         return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.binary_conv = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.downsample = downsample
        self.stride = stride

        self.binary_conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.bn4 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU(planes)
        self.downsample2 = downsample



    def forward(self, x):
        residual = x
        # x = self.bn2(x)

        out = self.binary_conv(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.bn2(residual)
       # out = self.bn2(out)
        out = self.prelu(out)

        residual = out
        # out = self.bn3(out)

        out = self.binary_conv2(out)
        out = self.bn4(out)

        # if out.size()!=residual.size():
        #     residual = self.downsample2(out)
        out += self.bn3(residual)
        out = self.prelu2(out)

        return out

class BiRealNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(BiRealNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.avgpool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def birealnet18(pretrained=False, **kwargs):
    """Constructs a BiRealNet-18 model. """
    model = BiRealNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def birealnet34(pretrained=False, **kwargs):
    """Constructs a BiRealNet-34 model. """
    model = BiRealNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

