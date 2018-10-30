# natural-adversary-pytorch

This repository provides the code for the paper "[Generating Natural Adversarial Examples](https://arxiv.org/abs/1710.11342)", ICLR 2018 in pytorch version and also I added extra works to improve the paper's work.

## List of extra works
1. Add different noise based on singular value decomposition.
2. I wonder that the attention of classifier is different depending on whether the image is natural or not. So I incorporate Grad-CAM module for tracing classifer's view.
3. To my best knowledge, natural image has statistical properties that related to RGB values. .. WHAT ABOUT MNIST DATA?

## TODO
- [ ] Convert the code that is provided by paper's author into pytorch version
- [ ] Investigate the statistical properties of natural image.
 
