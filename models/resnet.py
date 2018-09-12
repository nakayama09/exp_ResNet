'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from se_module import SELayer

class Basic_Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Basic_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_Block(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(ResNet, self).__init__()
        depth = args.depth

        self.num_planes = args.baseWidth

        if args.block ==  'Basic_Block':
            block = Basic_Block
            print('Deploying basic block')
        elif args.block == 'Bottleneck_Block':
            block = Bottleneck_Block
            print('Deploying bottleneck block')
        elif args.block == 'SE_Bottleneck_Block':
            block = SE_Bottleneck_Block
            print('Deploying se bottleneck block')

        num_stages = 3

        if not block == Basic_Block:
            print('bottleneck')
            num_blocks = (depth-2)/(num_stages*3)
        else:
            print('basic block')
            num_blocks = (depth-2)/(num_stages*2)


        self.conv1 = nn.Conv2d(3, args.baseWidth, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(args.baseWidth)
        self.layer1 = self._make_layer(block, args.baseWidth, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, args.baseWidth*2, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, args.baseWidth*4, num_blocks, stride=2)
        self.linear = nn.Linear(args.baseWidth*4*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.num_planes, planes, stride))
            self.num_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = ResNet(args, num_classes)
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()