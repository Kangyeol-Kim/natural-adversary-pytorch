import os, sys
import numpy as np
import argparse

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from solver import Solver, Tester


def main(args):

    # Make directories
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    # Get DataLoader
    if args.data == 'mnist':
        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
        
        train_loader = DataLoader(datasets.MNIST('./data', 
                                  train=True, 
                                  download=True,
                                  transform=transform), 
                                  batch_size=args.batch_size, 
                                  shuffle=True)
        
        val_loader = DataLoader(datasets.MNIST('./data', 
                                train=False, 
                                download=True,
                                transform=transform), 
                                batch_size=args.batch_size, 
                                shuffle=True)
    elif args.data == 'lsun':
        pass

    # Training or Test
    if args.mode == 'train':
        solver = Solver()
        solver.train()
    elif args.mode == 'test:
        tester = Tester()
        tester.test()

    

        



def str2bool(x):
    return x.lower() in ['true', 1, 'yes', 'y']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task Specification
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist', 'lsun'])

    # Model Cofiguration
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=28)

    # Hyper-parameters
    parser.add_argument('--gp_weight', type=float, default=10.0, help='Gradient penalty weight for training WGAN')
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--i_lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)

    # Getting Adversary Examples Configuration
    parser.add_argument('--gan_path', type=str, default=None, help='Path for loading pretrained GAN')
    parser.add_argument('--cls_path', type=str, default=None, help='Path for loading pretrained Classifier')
    parser.add_argument('--delta_r', type=float, default=0.01, help='Increment of search range')
    parser.add_argument('--search', type=str, default='iterative', choice=['iterative', 'recursive'])
    parser.add_argument('--n_samples', type=int, default=5000, help='Numner of samples in each search iteration')
    

    # Paths for saving results
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None, help='Path for saving adversary')

    # Log Configurations
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--save_step', type=int, default=2)

    # Misc
    parser.add_argument('--n_gpus', type=int, default=1)


    args = parser.parse_args()
    print(args)
    main(args)
    