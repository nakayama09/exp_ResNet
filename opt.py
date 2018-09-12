import argparse

parser = argparse.ArgumentParser(description='Pytorch experiment',fromfile_prefix_chars='@')
parser.add_argument('-dataset', dest='dataset', action='store',
										default='cifar10',
										help='select dataset(cifar10, cifar100) (default: cifar10)')

parser.add_argument('-train_bs',type=int, dest='train_batchSize', action='store',
										default=128,
										help='set batchsize (default: 128)')

parser.add_argument('-test_bs',type=int, dest='test_batchSize', action='store',
										default=100,
										help='set batchsize for test data (default: 100)')

parser.add_argument('-lr',type=float, dest='lr', action='store',
										default=0.1,
										help='set learning rate (default: 0.1)')

parser.add_argument('-nEpochs',type=int, dest='nEpochs', action='store',
										default=200,
										help='set epoch (default: 300)')

parser.add_argument('-start_epoch',type=int, dest='start_epoch', action='store',
										default=0,
										help='set epoch (default: 0)')

parser.add_argument('-block', type=str, dest='block', action='store',
										default='Basic_Block',
										help='use block (default: Bottleneck_Block)')

parser.add_argument('-baseWidth',type=int, dest='baseWidth', action='store',
										default=16,
										help='base number of filter (default: 64)')

parser.add_argument('-depth',type=int, dest='depth', action='store',
										default=20,
										help='number of network depth (default: 100)')

parser.add_argument('-weight_decay',type=float, dest='weight_decay', action='store',
										default=0.0005,
										help='weight decay (default: 5.0e-4)')

parser.add_argument('-resume', '-r', dest='resume', action='store_true',
										help='resume from checkpoint')

args = parser.parse_args()

