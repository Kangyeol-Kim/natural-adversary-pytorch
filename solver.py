import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from model import WganG, WganD, Inverter, BottleNeck
from classifier import LeNet, VGG
from utils import gen_svd_vec
from logger import Logger

import time
import datetime
import sys
import os
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np

class Solver():
    def __init__(self, args, train_loader=None, val_loader=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Build Model & Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.build_model()

        # Create Logger
        if self.args.use_tb:
            self.logger = Logger('./logs')

        # Train Start ==>
        self.train()

    def build_model(self):
        self.G = WganG(z_dim=self.args.z_dim).to(self.device)
        self.D = WganD().to(self.device)
        self.Inverter = Inverter(z_dim=self.args.z_dim).to(self.device)
        self.BottleNeck = BottleNeck(in_dim=self.args.t_v*(2*self.args.image_size+1),
                                     z_dim=self.args.z_dim).to(self.device)
        if self.args.mode == 'ori_train':
            self.g_optim = optim.Adam(self.G.parameters(),
                                    lr=self.args.g_lr,
                                    betas=(self.args.beta1, self.args.beta2))
        elif self.args.mode == 'svd_train':
            self.i_optim = optim.Adam(chain(self.G.parameters(), self.BottleNeck.parameters()),
                                        lr=self.args.g_lr,
                                        betas=(self.args.beta1, self.args.beta2))
        self.d_optim = optim.Adam(self.D.parameters(),
                                  lr=self.args.d_lr,
                                  betas=(self.args.beta1, self.args.beta2))
        self.i_optim = optim.Adam(self.Inverter.parameters(),
                                    lr=self.args.i_lr,
                                    betas=(self.args.beta1, self.args.beta2))


        self.MSELoss = nn.MSELoss()
        if self.args.n_gpus > 1:
            print('===> Use multiple gpus : %d' % (self.args.n_gpus))
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.Inverter = nn.DataParallel(self.Inverter)
            self.BottleNeck = nn.DataParallel(self.BottleNeck)


    def train(self):    
        # Initializing Logging Object and Data Loader
        self.train_d_loss = AverageMeter()
        self.train_g_loss = AverageMeter()
        self.train_i_loss = AverageMeter()
        self.val_d_loss = AverageMeter()
        self.val_g_loss = AverageMeter()
        self.val_i_loss = AverageMeter()
        train_iter = iter(self.train_loader)

        # MISC
        iter_per_epoch = len(self.train_loader.dataset) // self.args.batch_size
        if len(self.train_loader.dataset) % self.args.batch_size != 0:
            iter_per_epoch += 1
        self.epoch = 0 # NOTE: Temporary fixed
        self.eval_loss = sys.maxsize # NOTE: Temporary fixed

        # Fixed inputs for sampling.
        self.fixed_noise = torch.randn(self.args.batch_size, self.args.z_dim).to(self.device)
        self.fixed_images = next(iter(self.train_loader))[0].to(self.device)

        start_time = time.time()
        # Training Phase
        for i in range(self.args.n_iters):
            self.G.train()
            self.D.train()
            self.Inverter.train()
            
            try:
                real_images, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                real_images, _ = next(train_iter)
            
            real_images = real_images.to(self.device)
            #NOTE : ADDED
            if self.args.mode == 'ori_train':
                noise = torch.randn(self.args.batch_size, self.args.z_dim).to(self.device)
            elif self.args.mode == 'svd_train':
                noise = gen_svd_vec(x, t=3).to(self.device)
                noise = self.BottleNeck(noise) # Compress z_dim to 100

            # =============== Train D =============== #
            # Compute Critic Loss.
            d_real_loss = self.D(real_images)
            fake_images = self.G(noise)
            d_fake_loss = self.D(fake_images)

            # Comput Gradient Penalty
            eps = torch.rand(real_images.size(0), 1, 1, 1).to(self.device)
            xhat = eps*real_images + (1.-eps)*fake_images
            d_gp_loss = self.calc_grad_pn(out=self.D(xhat), x=xhat)

            # Discriminator Backprop.
            d_loss = torch.mean(d_fake_loss) - torch.mean(d_real_loss) + self.args.gp_weight * d_gp_loss
            self.reset_grad()
            d_loss.backward(retain_graph=True) # to backpropagate through D several times
            self.d_optim.step()

            # =============== Train G =============== #
            g_loss = -torch.mean(self.D(fake_images))
            self.reset_grad()
            g_loss.backward()
            self.g_optim.step()

            # =============== Train I =============== #
            # Reconstruct Original ones.
            recon_images = self.G(self.Inverter(real_images))
            recon_noise = self.Inverter(self.G(noise))

            # Compute losses of each one.
            recon_loss = self.MSELoss(real_images, recon_images)
            if self.args.mode == 'ori_train':
                div_loss = self.MSELoss(noise, recon_noise)
            elif self.args.mode == 'svd_train':
                div_loss = self.MSELoss(noise, recon_noise)
            # TODO FIXED

            # Inverter Backpropagation
            i_loss = recon_loss + div_loss
            self.reset_grad()
            i_loss.backward()
            self.i_optim.step()

            # =============== Logging =============== #
            # Update Loss Objects
            self.train_d_loss.update(d_loss.item())
            self.train_g_loss.update(g_loss.item())
            self.train_i_loss.update(i_loss.item())

            # Print Logging
            if (i + 1) % self.args.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print('Iteration [{0}/{1}]\t'
                      'Elapsed : {2}\t'
                      'D_loss : {d_loss.val:.4f}({d_loss.avg:.4f})\t'
                      'G_loss : {g_loss.val:.4f}({g_loss.avg:.4f})\t'
                      'I_loss : {i_loss.val:.4f}({i_loss.avg:.4f})'.format(
                          i + 1, self.args.n_iters, elapsed,
                          d_loss=self.train_d_loss,
                          g_loss=self.train_g_loss,
                          i_loss=self.train_i_loss))

            # Use tensorboard option 
            if self.args.use_tb:
                info = { 'train_d_loss': d_loss.item(), 
                         'train_g_loss': g_loss.item(),
                         'train_i_recon_loss': recon_loss.item(),
                         'train_i_div_loss': div_loss.item()}

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, i+1)

            # Validation start ==> 
            if (i + 1) % iter_per_epoch:
                self.epoch += 1
                self.validate()


    def validate(self):      
        # Model Mode Conversion
        self.G.eval()
        self.D.eval()
        self.Inverter.eval()
        # Reset for Evaluating Loss at This Epoch
        self.val_d_loss.reset()
        self.val_g_loss.reset()
        self.val_i_loss.reset()


        # Validation Phase
        print('##### Validation at {}-epoch starting'.format(self.epoch))
        for i, (real_images, _) in enumerate(self.val_loader):
            
            start_time = time.time()
            real_images = real_images.to(self.device)
            if self.args.mode == 'ori_train':
                noise = torch.randn(self.args.batch_size, self.args.z_dim).to(self.device)
            elif self.args.mode == 'svd_train':
                noise = gen_svd_vec(x, t=3).to(self.device)
                noise = self.BottleNeck(noise) # Compress z_dim to 100


            # =============== Validate D =============== #
            # Compute Critic Loss.
            d_real_loss = self.D(real_images)
            fake_images = self.G(noise)
            d_fake_loss = self.D(fake_images)

            # Comput Gradient Penalty
            eps = torch.rand(real_images.size(0), 1, 1, 1).to(self.device)
            xhat = eps*real_images + (1.-eps)*fake_images
            d_gp_loss = self.calc_grad_pn(out=self.D(xhat), x=xhat)

            # Discriminator Loss.
            d_loss = torch.mean(d_fake_loss) - torch.mean(d_real_loss) + self.args.gp_weight * d_gp_loss
            
            # =============== Validate G =============== #
            g_loss = -torch.mean(self.D(fake_images))
            
            # =============== Validate I =============== #
            # Reconstruct Original ones.
            recon_images = self.G(self.Inverter(real_images))
            recon_noise = self.Inverter(self.G(noise))

            # Compute losses of each one.
            recon_loss = self.MSELoss(real_images, recon_images)
            div_loss = self.MSELoss(noise, recon_noise)

            # Inverter Backpropagation
            i_loss = recon_loss + div_loss
            
            # =============== Logging =============== #
            # Update Loss Objects
            self.val_d_loss.update(d_loss.item())
            self.val_g_loss.update(g_loss.item())
            self.val_i_loss.update(i_loss.item())

            if (i + 1) % self.args.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print('##### Validation {0}'
                      'Elapsed : {1}\t'
                      'D_loss : {d_loss.val:.4f}({d_loss.avg:.4f})\t'
                      'G_loss : {g_loss.val:.4f}({g_loss.avg:.4f})\t'
                      'I_loss : {i_loss.val:.4f}({i_loss.avg:.4f})'.format(
                          self.epoch, elapsed,
                          d_loss=self.val_d_loss,
                          g_loss=self.val_g_loss,
                          i_loss=self.val_i_loss))
            
            # Use tensorboard
            if self.args.use_tb:
                info = { 'val_d_loss': d_loss.item(), 
                         'val_g_loss': g_loss.item(),
                         'val_i_recon_loss': recon_loss.item(),
                         'val_i_div_loss': div_loss.item()}

            # Generate from noise.
            output = self.G(self.fixed_noise)
            sample_path = os.path.join(self.args.sample_save_path, '{}_samples.jpg'.format(self.epoch))
            save_image(output.data.cpu(), sample_path)
            print('Saved generated images into {}...'.format(sample_path))

            # Reconstruct images.
            reconst_images = self.G(self.Inverter(self.fixed_images))
            comparison = torch.zeros((self.fixed_images.size(0) * 2,
                                      self.fixed_images.size(1),
                                      self.fixed_images.size(2),
                                      self.fixed_images.size(3)),
                                      dtype=torch.float).to(self.device)
            for k in range(self.fixed_images.size(0)):
                comparison[2*k] = self.fixed_images[k]
                comparison[2*k+1] = reconst_images[k]

            sample_path = os.path.join(self.args.sample_save_path, '{}_reconstructions.jpg'.format(self.epoch))
            save_image(comparison.data.cpu(), sample_path)

            # If Average loss is lower than previous epoch, Save Model
            if (self.val_d_loss.avg + self.val_g_loss.avg + self.val_i_loss.avg) < self.eval_loss:
                print('===> Set of Models is saving at epoch %d!!!' % (self.epoch))
                self.eval_loss = (self.val_d_loss.avg + self.val_g_loss.avg + self.val_i_loss.avg)
                ckpt = {
                    'epoch':self.epoch,
                    'D_state_dict':self.D.state_dict(),
                    'G_state_dict':self.G.state_dict(),
                    'I_state_dict':self.Inverter.state_dict(),
                    'B_state_dict':self.BottleNeck.state_dict(),
                    'd_optim_state_dict':self.d_optim.state_dict(),
                    'g_optim_state_dict':self.g_optim.state_dict(),
                    'i_optim_state_dict':self.i_optim.state_dict()}
                torch.save(ckpt, os.path.join(self.args.model_save_path, 'ckpt_%d' % (self.epoch)))


    def calc_grad_pn(self, out, x):
        weight = torch.ones(out.size()).to(self.device)
        dout = torch.autograd.grad(outputs=out,
                                   inputs=x,
                                   grad_outputs=weight,
                                   only_inputs=True,
                                   retain_graph=True,
                                   create_graph=True)[0]
        dout = dout.view(dout.size(0), -1)
        dout_l2 = torch.sqrt(torch.sum(dout**2, dim=1))
        gp = torch.mean((dout_l2 - 1)**2)
        return gp
        

    def reset_grad(self):
        self.d_optim.zero_grad()
        self.g_optim.zero_grad()
        self.i_optim.zero_grad()


class AdversaryGen():
    def __init__(self, args, val_loader=None):
        self.args = args
        #NOTE: Generate adversary example with val loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build & Load pretrained models: GAN, Classifier
        self.load_pre_model()
        
    def load_pre_model(self):
        
        # Load pretrained models
        self.G = WganG(z_dim=self.args.z_dim).to(self.device)
        self.Inverter = Inverter().to(self.device)
        self.BottleNeck = BottleNeck(in_dim=self.args.t_v*(2*self.args.image_size+1),
                                     z_dim=self.args.z_dim).to(self.device)
        ckpt = torch.load(self.args.ckpt_path)
        self.G.load_state_dict(ckpt['G_state_dict'])
        self.Inverter.load_state_dict(ckpt['I_state_dict'])
        self.BottleNeck.load_state_dict(ckpt['B_state_dict'])

        # Load 
        if self.args.cls_arc == 'lenet':
            self.C = LeNet().to(self.device)
            cls_path = os.path.join(self.args.cls_path)
            self.C.load_state_dict(torch.load(cls_path, map_location=lambda storage, loc: storage))

        print('(G, I) Models[Pretrained epoch: %d] are loaded!!\n' 
              '(C) model %s is loaded!!\n'
              'Search algorithm : %s' % (ckpt['epoch'], self.args.cls_arc, self.args.search))


    def to_np(self, x):
        return x.data.cpu().numpy()
    
    def generate_adversary(self):    
        # Generate adversary examples.
        for j, (images, labels) in enumerate(self.val_loader):
            for i in range(32):
                x = images[i].unsqueeze(0).to(self.device)
                y = labels[i].to(self.device)

                adversary, _ = self.iterative_search(x, y)
                sample_save_path = os.path.join(self.args.sample_save_path,
                                 '{}_{}_{}.jpg'.format(self.args.cls_arc, j+1, i+1))
                self.save_adversary(adversary, sample_save_path)
                print('Saved natural adversary example:{}...'.format(sample_save_path))

    def save_adversary(self, adversary, filename):
        fig, ax = plt.subplots(1, 2, figsize=(7, 3))

        ax[0].imshow(adversary['x'],
                     interpolation='none', cmap=plt.get_cmap('gray'))
        ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=20)
        ax[0].axis('off')

        ax[1].imshow(adversary['x_adv'],
                     interpolation='none', cmap=plt.get_cmap('gray'))
        ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=20)
        ax[1].axis('off')

        fig.savefig(filename)
        plt.close()


    # Codes from https://github.com/zhengliz/natural-adversary/blob/master/image/search.py
    def iterative_search(self, x, y, y_target=None, z=None,
                         lower=0., upper=10., p=2):
        """
        Related Objects
        :param x: SINGLE input instance
        :param y: SINGLE label
        :param y_target: target label for adversary
        :param z: latent vector corresponding to x
        :param l: lower bound of search range
        :param h: upper bound of search range
        :param p: indicating norm order
        :return: adversary for x against the classifier (d_adv is delta z between z and z_adv)
        """
        # Init
        G = self.G
        I = self.Inverter
        C = self.C
        n_samples = self.args.n_samples
        delta_r = self.args.delta_r
        curr_l = lower
        curr_h = upper + delta_r
        x_adv, y_adv, z_adv, d_adv = None, None, None, None

        if z is None:
            z = I(x)

        while True:
            delta_z = np.random.randn(n_samples, z.size(1))
            d = np.random.rand(n_samples) * (curr_h - curr_l) + curr_l # random values between the search range (r, r + delta r]
            delta_z_norm = np.linalg.norm(delta_z, ord=p, axis=1)      # Lp norm of delta z along axis=1
            d_norm = np.divide(d, delta_z_norm).reshape(-1, 1)         # rescale/normalize d
            delta_z = np.multiply(delta_z, d_norm)                     # norm (r, r + delta r] x norm (1) => norm (r, r + delta r]
            delta_z = torch.from_numpy(delta_z).float().to(self.device)
            z_tilde = z + delta_z
            x_tilde = G(z_tilde)
            y_tilde = torch.argmax(C(x_tilde), dim=1)

            if y_target is None:
                indices_adv = np.where(y_tilde != y)[0]
            else:
                indices_adv = np.where(y_tilde == y_target)[0]

            # No candidate generated.
            if len(indices_adv) == 0:
                print('No candidate generated within [{},{}]'.format(curr_l,curr_h))
                curr_l = curr_h
                curr_h = curr_l + delta_r

            # Certain candidates generated.
            else:
                # Choose the data index with the least perturbation.
                idx_adv = indices_adv[np.argmin(d[indices_adv])]

                if y_target is None:
                    assert (y_tilde[idx_adv] != y)

                else:
                    assert (y_tilde[idx_adv] == y_target)

                # Save natural adversary example.
                if d_adv is None or d[idx_adv] < d_adv:
                    x_adv = x_tilde[idx_adv]
                    y_adv = y_tilde[idx_adv]
                    z_adv = z_tilde[idx_adv]
                    d_adv = d[idx_adv]

                    if y_target is None:
                        print("Untarget y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))
                    else:
                        print("Targeted y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))

                    break

        # Output : Adversary examples
        adversary = {'x': self.to_np(x.squeeze()),
                     'y': self.to_np(y),
                     'z': self.to_np(z.squeeze()),
                     'x_adv': self.to_np(x_adv.squeeze()),
                     'y_adv': self.to_np(y_adv),
                     'z_adv': self.to_np(z_adv),
                     'd_adv': d_adv}

        # POINT : COMPUTATION COST CAN BE DECREASED IF WE USE THE INFORMATION OF IMAGE
        # Output : Perturbation magnitude, n_sample 
        cost = {'ptb':np.linalg.norm(d_adv, norm=2),
                'n_samples':n_samples}

        return adversary, cost
    
    def svd_iterative_search(self,):
        """ Search using svd vector """
        pass

    



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count