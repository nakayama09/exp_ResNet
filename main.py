'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys

from models import *
import utils
from utils import progress_bar
from utils import learning_rate

import opt

from torchsummary import summary

from autoaugment import CIFAR10Policy


# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()

args=opt.args

dataset=args.dataset.strip()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
# print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    #CIFAR10Policy(),
    transforms.ToTensor(),
    transforms.Normalize(utils.mean[args.dataset], utils.std[args.dataset]),
    #if dataset == 'cifar10':
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # else if dataset == 'cifar100':
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(utils.mean[args.dataset], utils.std[args.dataset]),
    #if dataset == 'cifar10':
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # else if dataset == 'cifar100':
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

if dataset == 'cifar10':
    print("| Preparing CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 10
elif dataset == 'cifar100':
    print("| Preparing CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batchSize, shuffle=True, num_workers=16, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batchSize, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

net = ResNet(args, num_classes)
# net = ResNet20()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#summary(net, (3, 32, 32))

#for parameter in net.parameters():
#    print(parameter)

params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(params)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate(args.lr, epoch, last_epoch), momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % (epoch+1))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lr=learning_rate(args.lr, epoch, args.nEpochs)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs.cuda()
        targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.3f'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))
    # print('Loss: {:.3f} | Acc: {:.3f}% ({:d}/{:d}) | lr: {:.3f}'.format(train_loss/(batch_idx+1), 100.*correct/total, correct, total, lr))
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Update Best Acc from {:.3f} to {:.3f}'.format(best_acc, acc))
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    print('Best Acc is {:.3f}'.format(best_acc))


for epoch in range(start_epoch, args.nEpochs):
    train(epoch)
    test(epoch)
