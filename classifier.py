import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import os
import argparse

# Lenet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

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

# VGGNet
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'] #From vgg11, detach last block

class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1 #NOTE: Only for grey scale data such as mnist
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls_arc', type=str, default='lenet', choices=['lenet', 'vgg'])
    parser.add_argument('--n_iters', type=int, default=10000)
    args = parser.parse_args()


    # Create directory if it doesn't exist.
    if not os.path.exists('./classifier'):
        os.makedirs('./classifier')

    # Dataset setting TODO: Fix it to replace dataset if you want to
    train_dataset = datasets.MNIST(root='./data/mnist',
                                   train=True,
                                   download=True,
                                   transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data/mnist',
                                  train=False,
                                  transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=2,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=32,
                                              shuffle=False,
                                              num_workers=2,
                                              drop_last=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cls_arc == 'lenet':
        net = LeNet().to(device)
    elif args.cls_arc == 'vgg':
        net = VGG(make_layers(cfg=cfg, batch_norm=True)).to(device)
        
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    # Start training.
    print('===> Training {}({} iters)...'.format(args.cls_arc,args.n_iters))
    start_time = time.time()
    for i in range(args.n_iters):
        for j, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print out training information.
        elapsed_time = time.time() - start_time
        print('Elapsed time [{:.4f}], Iteration [{}/{}]\t'
              'Loss: {:.4f}'.format(
               elapsed_time, i+1, args.n_iters, loss.item()))

        # Save model checkpoints.
        if (i+1) % 1000 == 0:
            model_path = './classifier/{}_{}.ckpt'.format(i+1, args.cls_arc)
            torch.save(net.state_dict(), model_path)
            print('Saved model checkpoints into {}...'.format(model_path))