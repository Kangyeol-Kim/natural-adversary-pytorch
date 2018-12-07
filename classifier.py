import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os
import argparse

# Lenet
# CIFAR10에 맞게 고쳐야함
class LeNet(nn.Module):
    def __init__(self, in_dim, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# Resnet
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,  in_dim=3, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    """
    lenet 
    $ python classifier.py --cls_arc=lenet --dataset=mnist --n_iters=5000
    $ python classifier.py --cls_arc=lenet --dataset=cifar --n_iters=5000

    resnet18
    $ python classifier.py --cls_arc=resnet18 --dataset=mnist --n_iters=5000
    $ python classifier.py --cls_arc=resnet18 --dataset=cifar --n_iters=5000
    """
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_arc', type=str, default='lenet', choices=['lenet', 'resnet18'])
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--n_iters', type=int, default=5000)
    parser.add_argument('--n_gpus', type=int, default=2)
    args = parser.parse_args()
    if args.dataset in ['mnist', 'cifar10']:
        args.num_classes = 10
    print(args)

    # Create directory if it doesn't exist.
    if not os.path.exists('./classifier'):
        os.makedirs('./classifier')

    # Load dataset
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data/mnist',
                                       train=True,
                                       download=True,
                                       transform=transform)
        val_dataset = datasets.MNIST(root='./data/mnist',
                                    train=False,
                                    transform=transform)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=16,
                                  drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=16,
                                drop_last=True)
    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data/cifar10',
                                       train=True,
                                       download=True,
                                       transform=transform)
        val_dataset = datasets.CIFAR10(root='./data/cifar10',
                                     train=False,
                                     transform=transform)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=16,
                                  drop_last=True)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=32,
                                shuffle=False,
                                num_workers=16,
                                drop_last=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model
    if args.cls_arc == 'lenet':
        if args.dataset == 'mnist':
            net = LeNet(in_dim=1, num_classes=args.num_classes).to(device)
        elif args.dataset == 'cifar10':
            net = LeNet(in_dim=3, num_classes=args.num_classes).to(device)
    elif args.cls_arc == 'resnet18':
        # currently using resnet18
        if args.dataset == 'mnist':
            net = ResNet(BasicBlock, [2, 2, 2, 2], in_dim=1, num_classes=args.num_classes).to(device)
        elif args.dataset == 'cifar10':
            net = ResNet(BasicBlock, [2, 2, 2, 2], in_dim=3, num_classes=args.num_classes).to(device)
    if args.n_gpus > 1:
        print('Multiple GPUs are being used[{}]'.format(args.n_gpus))
        net = nn.DataParallel(net)
        
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # ===================== Training Phase ===================== #
    print('===> Training {}({} iters)...'.format(args.cls_arc,args.n_iters))
    net.train()
    start_time = time.time()
    train_iter = iter(train_loader)
    for i in range(args.n_iters):
        try:
            images, labels = next(train_iter)
        except:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images_var = images.to(device)
        labels_var = labels.to(device)
        outputs = net(images_var)

        loss = criterion(outputs, labels_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print out training information.
        if (i+1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print('Elapsed time [{:.4f}], Iteration [{}/{}]\t'
                    'Loss: {:.4f}'.format(
                    elapsed_time, i+1, args.n_iters, loss.item()))

        # Save model checkpoints.
        if (i+1) % 1000 == 0:
            # Log validation accuracy 
            print('##### Validation from {}_iter ckpt!!'.format(i+1))
            net.eval()
            counter = 0

            for j, (images, labels) in enumerate(val_loader):
                images_var = images.to(device)
                labels_var = labels.to(device)
                outputs = net(images_var)

                # Count correct one
                pred = torch.argmax(outputs, dim=1)
                counter += torch.sum(pred.data.cpu() == labels) 
            acc = counter.float() / len(val_loader.dataset)
            model_path = './classifier/{}_{}_{}_{}.ckpt' \
                    .format(i+1, args.dataset, args.cls_arc, str((acc*1000).item())[0:3])
            torch.save(net.state_dict(), model_path)
            print('Saved model checkpoints into {} with accuracy-{}'.format(model_path, str(acc.item())))
