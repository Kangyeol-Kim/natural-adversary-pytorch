import torch
import torch.nn as nn
import torch.optim as optim

from model import WganG, WganD, Inverter

import time
import datetime
import sys

class Solver():
    def __init__(self, args, train_loader=None, val_loader=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # Build Model & Data Loader
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.build_model()

        # Train Start ==>
        self.train()

    def build_model(self):
        self.G = WganG(z_dim=self.args.z_dim).to(self.device)
        self.D = WganD().to(self.device)
        self.Inverter = Inverter().to(self.device)
        self.g_optim = optim.Adam(self.G.parameters(),
                                  lr=self.args.g_lr,
                                  self.betas=(self.args.beta1, self.args.beta2))
        self.d_optim = optim.Adam(self.D.parameters(),
                                  lr=self.args.d_lr,
                                  self.betas=(self.args.beta1, self.args.beta2))
        self.i_optim = optim.Adam(self.Inverter.parameters(),
                                  lr=self.args.i_lr,
                                  self.betas=(self.args.beta1, self.args.beta2))
        self.MSELoss = nn.MSELoss()
        if self.args.n_gpus > 1:
            print('===> Use multiple gpus : %d' % (self.args.n_gpus))
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.Inverter = nn.DataParallel(self.Inverter)


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
        iter_per_epoch = len(self.train_loader.dataset) % self.args.batch_size
        if len(self.train_loader.dataset) // self.args.batch_size != 0:
            iter_per_epoch += 1
        self.epoch = 0 # NOTE: Temporary fixed
        self.eval_loss = sys.maxsize # NOTE: Temporary fixed

        # Training Phase
        for iter in range(self.args.n_iters):
            self.G.train()
            self.D.train()
            self.Inverter.train()
            start_time = time.time()
            
            try:
                real_images, _ = next(train_iter)
            except:
                train_iter = iter(self.train_loader)
                real_images, _ = next(train_iter)
            
            real_images = real_images.to(self.device)
            noise = torch.randn(self.args.batch_size, self.z_dim).to(self.device)

            # =============== Train D =============== #
            # Compute Critic Loss.
            d_real_loss = self.D(real_images)
            fake_images = self.G(noise)
            d_fake_loss = self.D(fake_images)

            # Comput Gradient Penalty
            eps = torch.rand(real.images.size(0), 1, 1, 1).to(self.device)
            xhat = eps*real_images + (1.-eps)*fake_images
            d_gp_loss = self.calc_grad_pn(out=self.D(xhat), x=xhat)

            # Discriminator Backprop.
            d_loss = torch.mean(d_fake_loss) - torch.mean(d_real_loss) + self.args.gp_weight * d_gp_loss
            self.reset_grad()
            d_loss.backward()
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
            div_loss = self.MSELoss(noise, recon_noie)

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

            if (i + 1) % self.args.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print('Iteration [{0}/{1}]\t'
                      'Elapsed : {2}\t'
                      'D_loss : {d_loss.val:.4f}({d_loss.avg:.4f})\t',
                      'G_loss : {g_loss.val:.4f}({g_loss.avg:.4f})\t',
                      'I_loss : {i_loss.val:.4f}({i_loss.avg:.4f})'.format(
                          i + 1, self.args.n_iters, elapsed,
                          d_loss=self.train_d_loss,
                          g_loss=self.train_g_loss,
                          i_loss=self.train_i_loss))
            
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
        for i, (real_images, _) in enumerate(self.val_loader):
            
            start_time = time.time()
            real_images = real_images.to(self.device)
            noise = torch.randn(self.args.batch_size, self.z_dim).to(self.device)

            # =============== Validate D =============== #
            # Compute Critic Loss.
            d_real_loss = self.D(real_images)
            fake_images = self.G(noise)
            d_fake_loss = self.D(fake_images)

            # Comput Gradient Penalty
            eps = torch.rand(real.images.size(0), 1, 1, 1).to(self.device)
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
            div_loss = self.MSELoss(noise, recon_noie)

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
                      'D_loss : {d_loss.val:.4f}({d_loss.avg:.4f})\t',
                      'G_loss : {g_loss.val:.4f}({g_loss.avg:.4f})\t',
                      'I_loss : {i_loss.val:.4f}({i_loss.avg:.4f})'.format(
                          self.epoch, elapsed,
                          d_loss=self.val_d_loss,
                          g_loss=self.val_g_loss,
                          i_loss=self.val_i_loss))
            # If Average loss is lower than previous epoch, Save Model
            if (val_d_loss.avg + val_g_loss.avg + val_i_loss.avg) < self.eval_loss:
                print('===> Set of Model is saving at epoch %d!!!' % (self.epoch))
                self.eval_loss = (val_d_loss.avg + val_g_loss.avg + val_i_loss.avg)
                ckpt = {

                }
                # TODO: SAVE MODEL PART
                pass


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


class Tester():
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