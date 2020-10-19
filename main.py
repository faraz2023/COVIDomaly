import os, argparse
import torch
from numpy import savetxt
import utils.DataLoader as ds
import utils.Solver as solver
import torch.nn as nn
from utils.NeuralNet import Generator


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        m.weight.data.normal_(0, 0.05)
        m.bias.data.normal_(0, 0.025)

def main():

    dataset = ds.Dataset(normalize=FLAGS.datnorm, normal=FLAGS.normal, n_exp=FLAGS.n_exp, n_splits=FLAGS.k_fold)

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    use_cuda = torch.cuda.is_available()
    G = Generator(height=dataset.height, width=dataset.width, channel=dataset.channel,
                  device=device, ngpu=FLAGS.ngpu,
                  ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr)

    print('use_cuda:', use_cuda)
    if use_cuda:
        G = G.cuda()
    G.apply(init_weights)



    model_name = "01-f{}of{}-NandP".format(FLAGS.n_exp, FLAGS.k_fold)
    if(len(FLAGS.normal) == 1):
        model_name = "02-f{}of{}-Nonly".format(FLAGS.n_exp, FLAGS.k_fold)

    if(FLAGS.train):
        solver.train(G=G, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch, lr= FLAGS.lr,
                     model_name=model_name,snapshot_number=FLAGS.snapshot_number, load=FLAGS.loadModel, snapshot=FLAGS.doSnapshot)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--datnorm', type=str2bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=5, help='kernel size for constructing Neural Network')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=20, help='Training epoch')
    parser.add_argument('--batch', type=int, default=100, help='Mini batch size')
    parser.add_argument('-n', '--normal', nargs='+', help='Indicate what is Normal', required=True)
    parser.add_argument('--train', type=str2bool, default=True, help='Training or just Testing')
    parser.add_argument('--saveMemory', type=str2bool, default=True, help='Keeps only the most updated model params file')
    parser.add_argument('--doSnapshot', type=str2bool, default=True, help='Whether to run snapshots on the test set while training.')
    parser.add_argument('--loadModel', type=str2bool, default=True, help='Whether to load a previously trained model.'
                                                                         'Model must be available at Models/<modelname>/params/params-epoch<#>-G.')

    parser.add_argument('--k_fold', type=int, default=3, help='The number of folds for Kfold Crossvalidation')
    parser.add_argument('--n_exp', type=int, default=1, help='The number of fold to validate')
    parser.add_argument('--snapshot_number', type=int, default=10, help='At each snapshot_number of epochs in the training'
                                                                        ' process a snapshot on the test set is made and the model is saved')

    FLAGS, unparsed = parser.parse_known_args()
    main()





