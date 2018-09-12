# Exp ResNet for GPU server test in PyTorch
This repository contains the implementation for GPU server test in PyTorch

## Usage
Install [PyTorch](https://pytorch.org/), `numpy`, `Pillow`, and run:
~~~  
git clone https://github.com/nakayama09/exp_ResNet.git
~~~
## Training
To run the training, simply run main.py. By default, the script runs ResNet-20 on CIFAR-10 with all GPUs (default Batch Size is 256 for 4 GPUs).

To train ResNet-110 on 2 GPUs:
~~~
CUDA_VISIBLE_DEVICES=0,1 python main.py -depth 110 -train_bs 128
~~~
