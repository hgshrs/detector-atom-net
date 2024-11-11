import importlib
import moabb.datasets
from moabb.paradigms import P300
import torch
import torch.nn as nn
import torch.optim as optim
import danet
importlib.reload(danet)
import train
importlib.reload(train)
import matplotlib.pylab as plt
import numpy as np

if __name__=='__main__':
    device = 'cpu'
    # device = 'cuda:0'
    print('Available CUDA device:', device) # must choose available cuda device or cpu

    sfreq = 250 # sampling frequency of input signals
    net_path = 'tmp_tl.pth' # path for saving decomposer network parameters
    n_components = 8 # number of the decomposed signals
    param_detector = {'kernel_size':250, 'n_mid_channels':2, 'n_mid_layers':2} # parameters for conv layers for detectors
    alpha_l1 = 0 # coefficient for the sparsity loss
    n_epochs = 10000
    # n_epochs = 0
    lr = 1e-5
    beta1 = .5
    # reassign_atoms_every = 10 # every this number of epochs, run the atom reassignment

    # =======================================
    print('\nGet epoched signals for individual class...')
    # =======================================
    dataset = moabb.datasets.BNCI2014_008()
    paradigm = P300(resample=sfreq, channels=['Cz'])
    x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1,]) # size of tensor x: n_samples x n_channels x n_timepoints
    x = torch.from_numpy(x).float()
    dl = torch.utils.data.DataLoader(x.view(x.shape[0] * x.shape[1], -1), batch_size=1000, shuffle=True, num_workers=0)

    # =======================================
    print('\nTrain decomposer...')
    # =======================================
    slen = x.shape[2]
    param_atom = {'kernel_size':slen} # parameters for conv layers for atoms. kernel_size corresponds the signal length of an atom.
    net = danet.decomposer_time_locked(n_components, param_detector, param_atom).to(device)
    # x_ = net(x.view(x.size(0) * x.size(1), -1))
    net, loss = danet.load_net(net, n_components, param_detector, param_atom, net_path=net_path, with_loss=True, device=device, verbose=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999), weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='none')
    net, loss = train.train(device, net, loss, criterion, alpha_l1, optimizer, dl, n_epochs, net_path=net_path, save_every=100)

    # =======================================
    # Decompose
    # =======================================
    decomposed_signals = net(x[0, 0].view(1, -1)) # out sample x n_components x time point

    # =======================================
    # Visualize the training result
    # =======================================
    plt.figure(1); plt.clf()
    plt.semilogy(loss)
    plt.savefig('loss_tl.pdf', bbox_inches='tight', transparent=True)

    sig = x[0, 0].numpy()
    dcm = danet.DAnet(net=net, sfreq=sfreq, norm=True)
    epoch_t = np.arange(sig.shape[0]) / sfreq
    decomposed_signals = dcm.transform(sig[np.newaxis, np.newaxis])[0]
    plt.figure(2).clf()
    for cc in range(n_components):
        plt.subplot(n_components+1, 1, cc+1)
        plt.plot(epoch_t, sig, c='gray', lw=2, alpha=.5)
        plt.plot(epoch_t, decomposed_signals[cc])
    plt.subplot(n_components+1, 1, 9)
    plt.plot(epoch_t, sig, c='gray', lw=2, alpha=.5)
    plt.plot(epoch_t, decomposed_signals.sum(0))
    plt.legend(['Orignal', 'Reconstructed'])
