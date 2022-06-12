#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='1'

import torch
import torch.nn as nn


import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import models
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import time
import os


batch_size = 50
learning_rate = 0.01
root_dir = './'
default_directory = './saved'
# Data Augmentation
transform = transforms.Compose([
    transforms.Resize(224, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                         std=(0.2471, 0.2436, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


class_names = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
print(class_names)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)


'''model_ft.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5,inplace=False),
    nn.Linear(4096, 1024,bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(0.4,inplace=False),
    nn.Linear(1024, 10,bias=True),
)'''
model_ft.fc = nn.Linear(512, 10, bias=True)

'''
for param in model_ft.parameters():
    param.requires_grad = False

for param in model_ft.fc.parameters():
    param.requires_grad = True
'''

model_ft = model_ft.to(device)


criterion = nn.CrossEntropyLoss().to(device)

# Resnet 18
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learning_rate, momentum=0.9)


def train(epoch):
    model_ft.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.to(device)), Variable(target.to(device))

        optimizer_ft.zero_grad()
        output = model_ft(data).to(device)
        loss = criterion(output, target).to(device)

        loss.backward()
        optimizer_ft.step()

        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)

        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        # batch size 조정
        if batch_idx % 100 == 0:
            print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def save_checkpoint(directory, state, filename='latest.tar.gz'):

    if not os.path.exists(directory):
        os.makedirs(directory)

    model_filename = os.path.join(directory, filename)
    torch.save(state, model_filename)
    print("=> saving checkpoint")


def load_checkpoint(directory, filename='latest.tar.gz'):

    model_filename = os.path.join(directory, filename)
    if os.path.exists(model_filename):
        print("=> loading checkpoint")
        state = torch.load(model_filename)
        return state
    else:
        return None


def test():
    model_ft.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(testloader):
        data, target = Variable(data.to(device)), Variable(target.to(device))
        outputs = model_ft(data)
        loss = criterion(outputs, target)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()
    print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
          .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return 100 * correct/total


torch.cuda.empty_cache()
start_epoch = 0
start_time = time.time()
checkpoint = load_checkpoint(default_directory)

if not checkpoint:
    pass
else:
    start_epoch = checkpoint['epoch'] + 1
    model_ft.load_state_dict(checkpoint['state_dict'])
    optimizer_ft.load_state_dict(checkpoint['optimizer'])

best_acc = float(0)
for epoch in range(start_epoch, 20):
    torch.cuda.empty_cache()
    if epoch < 5:
        lr = learning_rate
    elif epoch < 10:
        lr = learning_rate * 0.5
    else:
        lr = learning_rate * 0.001
    for param_group in optimizer_ft.param_groups:
        param_group['lr'] = lr

    train(epoch)
    save_checkpoint(default_directory, {
        'epoch': epoch,
        'model': model_ft,
        'state_dict': model_ft.state_dict(),
        'optimizer': optimizer_ft.state_dict(),
    })

    now = test()
    if now > best_acc:
        save_checkpoint(default_directory, {
            'epoch': epoch,
            'model': model_ft,
            'state_dict': model_ft.state_dict(),
            'optimizer': optimizer_ft.state_dict(),
        }, filename='best_acc_with'+str(now)+'.tar.gz')
        best_acc = now

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(
    now.tm_hour, now.tm_min, now.tm_sec))
