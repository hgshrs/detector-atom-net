import warnings
import importlib
import moabb.datasets
from moabb.paradigms import SSVEP
import torch
import torch.nn as nn
import torch.optim as optim
import danet
importlib.reload(danet)
import train
importlib.reload(train)
import numpy as np
import sklearn.preprocessing
import matplotlib.pylab as plt

if __name__=='__main__':
    device = 'cpu'
    # device = 'cuda:0'
    print('Available CUDA device:', device) # must choose available cuda device or cpu

    sfreq = 250 # sampling frequency of input signals
    net_path = 'tmp_sa.pth' # path for saving decomposer network parameters
    n_components = 8 # number of the decomposed signals
    param_detector = {'kernel_size':250, 'n_mid_channels':2, 'n_mid_layers':2} # parameters for conv layers for detectors
    param_atom = {'kernel_size':125} # parameters for conv layers for atoms. kernel_size corresponds the signal length of an atom.
    alpha_l1 = 1e-5 # coefficient for the sparsity loss
    ns_epochs = [0, 100000] # without, with sparsity
    # ns_epochs = [0, 0] # without, with sparsity
    lr = 1e-5
    beta1 = .5
    # reassign_atoms_every = 10 # every this number of epochs, run the atom reassignment

    # =======================================
    print('\nGet epoched signals for individual class...')
    # =======================================
    dataset = moabb.datasets.Nakanishi2015()
    paradigm = SSVEP(resample=sfreq, channels=['POz'])
    x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[1,]) # size of tensor x: n_samples x n_channels x n_timepoints
    # x = x[:100] # Reduce samples for demo purpose
    x = torch.from_numpy(x[:100]).float()
    y = y[:x.shape[0]]
    y_oh = torch.tensor(sklearn.preprocessing.OneHotEncoder().fit_transform(y[:, np.newaxis]).toarray())
    ds = torch.utils.data.TensorDataset(x.view(x.shape[0] * x.shape[1], -1), y_oh)
    dl = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)

    # =======================================
    print('\nTrain decomposer...')
    # =======================================
    n_components = y_oh.shape[1]
    net = danet.decomposer_shared_atom(n_components, param_detector, param_atom).to(device)
    # net = danet.decomposer(n_components, param_detector, param_atom).to(device)
    net, loss = danet.load_net(net, n_components, param_detector, param_atom, net_path=net_path, with_loss=True, device=device, verbose=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999), weight_decay=1e-5)
    criterion = train.supervised_loss()
    for sse, n_epochs in enumerate(ns_epochs):
        if sse == 0:
            _alpha_l1 = 0.
            print('No sparsity loss...')
        elif sse == 1:
            _alpha_l1 = alpha_l1
            print('With sparsity loss...')
        net, loss = train.train(device, net, loss, criterion, _alpha_l1, optimizer, dl, n_epochs, mode='supervised', net_path=net_path, save_every=100)

    # =======================================
    # Decompose
    # =======================================
    net.eval()
    sig = x[10, 0]
    epoch_t = np.arange(sig.shape[0]) / sfreq
    decomposed_signals = net(sig.view(1, -1)).detach().numpy().squeeze() # out sample x n_components x time point

    # =======================================
    # Visualize the training result
    # =======================================
    plt.figure(1); plt.clf()
    plt.semilogy(loss)
    plt.savefig('loss_tl.pdf', bbox_inches='tight', transparent=True)

    atom = net.state_dict()['layers.{}.0.weight'.format(n_components)].numpy().squeeze()
    detout = np.zeros([n_components, sig.shape[0]])
    plt.figure(2).clf()
    plt.figure(3).clf()
    for cc in range(n_components):
        detout[cc] = net.layers[cc](sig.view(1, 1, sig.shape[0])).detach().numpy()
        plt.figure(2)
        plt.subplot(n_components, 1, cc+1)
        plt.plot(epoch_t, detout[cc])
        plt.figure(3)
        plt.subplot(n_components, 1, cc+1)
        plt.plot(epoch_t, sig, c='gray', lw=2, alpha=.5)
        plt.plot(epoch_t, decomposed_signals[cc])
    plt.figure(4).clf()
    plt.subplot(311)
    plt.plot(epoch_t, sig, c='gray', lw=2, alpha=.5)
    plt.plot(epoch_t, decomposed_signals.sum(0))
    plt.subplot(312)
    plt.plot(epoch_t, decomposed_signals.T)
    plt.subplot(313)
    plt.plot(epoch_t[:atom.shape[0]], atom)
