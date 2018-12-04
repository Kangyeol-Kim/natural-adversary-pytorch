import torch
import torch.nn as nn


class WganG(nn.Module):
    """ Generator. """
    def __init__(self, z_dim=100, ngf=64):
        super(WganG, self).__init__()
        self.z_dim = z_dim
        self.ngf = ngf

        self.linear = nn.Sequential(
            nn.Linear(self.z_dim, self.ngf * 64),
            nn.ReLU())
        
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU())
        
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 2, self.ngf * 1, 4, 2, 1, bias=False),
            nn.ReLU(),

            nn.ConvTranspose2d(self.ngf * 1, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

    def forward(self, z):
        out = self.linear(z)
        out = out.view(out.size(0), self.ngf*4, 4, 4) # 4x4

        out = self.deconv_1(out) # 8x8
        out = out[:, :, :7, :7] # 7x7

        out = self.deconv_2(out) # 28 x 28

        return out


class WganD(nn.Module):
    """ Discriminator. """
    def __init__(self, ngf=64):
        super(WganD, self).__init__()
        self.ngf = ngf

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.ngf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(), # 14 x 14

            nn.Conv2d(self.ngf, self.ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(), # 7 x 7 
            
            nn.Conv2d(self.ngf * 2, self.ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU() # 4 x 4
        )

        self.linear = nn.Sequential(
            nn.Linear(self.ngf * 64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.ngf * 64)
        out = self.linear(out)
        return out

class Inverter(nn.Module):
    """ Inverter. """
    def __init__(self, ngf=64, z_dim=64):
        super(Inverter, self).__init__()
        self.ngf = ngf
        self.z_dim = z_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, self.ngf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(), # 14 x 14

            nn.Conv2d(self.ngf, self.ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(), # 7 x 7 
            
            nn.Conv2d(self.ngf * 2, self.ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU() # 4 x 4
        )

        self.linear = nn.Sequential(
            nn.Linear(self.ngf * 64, self.z_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BottleNeck(nn.Module):
    """ Bottleneck Layer for resize SVD vector """
    def __init__(self, in_dim, z_dim=64):
        super(BottleNeck, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, z_dim),
            nn.Tanh()) # scaling range [-1, 1]
    
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
     G = WganG()



