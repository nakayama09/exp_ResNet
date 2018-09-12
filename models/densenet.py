'''myDenseNet in PyTorch from Keras script'''
import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from se_module import SELayer

class SE_Bottleneck_Block(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(SE_Bottleneck_Block, self).__init__()

        #print('in:'+str(in_planes))
        #print('gr:'+str(growth_rate))
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.99)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate, momentum=0.99)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3 , padding=1, bias=False)
        self.se = SELayer(growth_rate, 1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.se(out)
        out = torch.cat([out,x], 1)

        return out

class Basic_Block(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Basic_Block, self).__init__()

        self.bn = nn.BatchNorm2d(in_planes, momentum=0.99)
        self.conv = nn.Conv2d(in_planes, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn(x))
        out = self.conv(out)
        out = torch.cat([out,x], 1)

        return out

class Bottleneck_Block(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck_Block, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.99)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate, momentum=0.99)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3 , padding=1, bias=False)
        
    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)

        return out

class Transition_Block(nn.Module):
    def __init__(self, in_planes, out_planes, last):
        super(Transition_Block, self).__init__()
        self.last = last
        self.bn = nn.BatchNorm2d(in_planes, momentum=0.99)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn(x))
        out = F.avg_pool2d(self.conv(out), 2)

        return out

class DenseNet(nn.Module):
    def __init__(self, args, num_classes=10):
        super(DenseNet, self).__init__()
        depth = args.depth
        reduction = args.reduction
        #block = args.block
        self.growth_rate = args.growth_rate

        if args.block ==  'Basic_Block':
            block = Basic_Block
            print('Deploying basic block')
        elif args.block == 'Bottleneck_Block':
            block = Bottleneck_Block
            print('Deploying bottleneck block')
        elif args.block == 'SE_Bottleneck_Block':
            block = SE_Bottleneck_Block
            print('Deploying se bottleneck block')

        last = False

        num_stages = 3
        num_blocks = (depth-4)/num_stages #num_blocks per stages

        if not block == Basic_Block:
            print('bottleneck')
            num_blocks /= 2
            num_planes = 2*self.growth_rate
        else:
            num_planes = 16

        print(type(block))
        print(block)

        self.conv = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        
        self.layers = []
        for i in xrange(num_stages):
            for j in xrange(num_blocks):
                self.layers.append(block(num_planes, self.growth_rate))
                #num_planes += self.growth_rate
                num_planes += int(math.floor(self.growth_rate*0.5))
                # print(num_planes)
            if i == (num_stages-1):
                last = True
            out_planes = int(math.floor(num_planes*reduction))
            if not i == (num_stages-1):
                self.layers.append(Transition_Block(num_planes, out_planes, last))
                num_planes = out_planes

        self.dense = nn.Sequential(*self.layers)

        self.bn = nn.BatchNorm2d(num_planes, momentum=0.99)
        self.linear = nn.Linear(num_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        #print(out.type())
        out = self.dense(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def test():
    net = DenseNet(args, num_classes)
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
