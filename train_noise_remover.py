import sys
import importlib
import moabb.datasets
from moabb.paradigms import FixedIntervalWindowsProcessing
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pylab as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
import numpy as np
import train
importlib.reload(train)
import danet
importlib.reload(danet)

def set_processing(length, sfreq, band, baseline, channels=None):
    processing = FixedIntervalWindowsProcessing(
        # new parameters:
        length = length,
        stride = length,
        start_offset = 60,
        stop_offset = None,
        resample = sfreq,
        fmin = band[0], # lower cutoff [Hz]
        fmax = band[1], # higher cutoff [Hz]
        baseline = baseline,
        channels=channels,
    )
    return processing

if __name__=='__main__':
    device = 'cpu'
    # device = 'cuda:0'
    print('Available CUDA device:', device) # must choose available cuda device or cpu

    sfreq = 250 # sampling frequency of input signals
    length = 3 # signal length [sec] for Fixedintervalwindows
    band = [.5, 99] # band for bandpass filter
    baseline = (0, length) # the time period for baseline calculation
    slen = int(sfreq * length) # number of time points
    channels = ['F3']
    threshold_ref = 100

    net_path_s = 'tmp_s.pth' # path for saving decomposer network parameters
    net_path_n = 'tmp_n.pth' # path for saving decomposer network parameters
    n_components = 8
    param_detector = {'kernel_size':125, 'n_mid_channels':8, 'n_mid_layers':4} # parameters for conv layers for detectors
    param_atom = {'kernel_size':125} # parameters for conv layers for atoms. kernel_size corresponds the signal length of an atom.
    alpha_l1 = 1e-5 # coefficient for the sparsity loss
    ns_epochs = [100, 1000] # without, with sparsity
    # ns_epochs = [0, 0] # without, with sparsity
    n_epochs_alt = 10
    lr = 1e-5
    beta1 = .5
    reassign_atoms_every = 10 # every this number of epochs, run the atom reassignment

    # =======================================
    print('\nGet epoched signals by FixedIntervalWindows...')
    # =======================================
    dataset = moabb.datasets.Shin2017A(accept=True)
    sbjs = [3,]
    processing = set_processing(length, sfreq, band, baseline, channels=channels) # set the parameters for get_data
    x, y, metadata = processing.get_data(dataset=dataset, subjects=sbjs) # size of tensor x: n_samples x n_channels x n_timepoints
    processing = set_processing(length, sfreq, band, baseline=None, channels=['VEOG', 'HEOG']) # set the parameters for get_data
    x_eog, y, metadata = processing.get_data(dataset=dataset, subjects=sbjs) # size of tensor x: n_samples x n_channels x n_timepoints
    epoch_t = np.arange(x.shape[2]) / sfreq

    noise_event_period = np.zeros_like(x_eog)
    noise_event_period[np.abs(x_eog) > threshold_ref] = 1
    noise_event_period = np.logical_or(noise_event_period[:, 0], noise_event_period[:, 1])
    print('{} points from {} are supposed to be noise-event periods.'.format(noise_event_period.sum(), np.prod(noise_event_period.shape)))

    tidx = 8
    plt.figure(1).clf()
    plt.subplot(311)
    plt.plot(epoch_t, x_eog[tidx, 0].T, label='VEOG')
    plt.plot(epoch_t, x_eog[tidx, 1].T, label='HEOG')
    for ee in range(len(epoch_t) - 1):
        if noise_event_period[tidx, ee]:
            plt.axvspan(epoch_t[ee], epoch_t[ee+1], color='gray', alpha=.2)
    plt.legend()
    plt.subplot(312)
    plt.plot(epoch_t, x[tidx].T)
    plt.legend(channels)

    x = torch.from_numpy(x).float().view(x.shape[0] * x.shape[1], -1)
    y = torch.from_numpy(noise_event_period).float()
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=True, num_workers=0)

    # =======================================
    print('\nTrain decomposer...')
    # =======================================
    net_s = danet.decomposer(n_components, param_detector, param_atom).to(device)
    net_s, loss_s = danet.load_net(net_s, n_components, param_detector, param_atom, net_path=net_path_s, with_loss=True, verbose=True)
    net_n = danet.decomposer(n_components, param_detector, param_atom).to(device)
    net_n, loss_n = danet.load_net(net_n, n_components, param_detector, param_atom, net_path=net_path_n, with_loss=True, verbose=True)
    # criterion_s = nn.MSELoss(reduction='none')
    criterion_s = train.labeled_MSELoss(label=0)
    criterion_n = train.labeled_MSELoss(label=1)
    optimizer_s = optim.Adam(net_s.parameters(), lr=lr, betas=(beta1, .999), weight_decay=1e-5)
    optimizer_n = optim.Adam(net_n.parameters(), lr=lr, betas=(beta1, .999), weight_decay=1e-5)

    if loss_s.shape[0] > 0:
        s = torch.zeros_like(x)
    else:
        s = net_s(x).detach().sum(1)
    for sse, n_epochs in enumerate(ns_epochs):
        for ee1 in range(0, n_epochs, n_epochs_alt):
            if sse == 0:
                _alpha_l1 = 0.
            else:
                _alpha_l1 = alpha_l1
            xn = x - s
            dsn = torch.utils.data.TensorDataset(x, xn, y)
            dln = torch.utils.data.DataLoader(dsn, batch_size=100, shuffle=True, num_workers=0)
            net_n, loss_n = train.train(device, net_n, loss_n, criterion_n, _alpha_l1, optimizer_n, dln, n_epochs_alt, mode='labeled_mse', net_path=net_path_n)
            n = net_n(x).detach().sum(1)
            xs = x - n
            dss = torch.utils.data.TensorDataset(x, xs, y)
            dls = torch.utils.data.DataLoader(dss, batch_size=100, shuffle=True, num_workers=0)
            net_s, loss_s = train.train(device, net_s, loss_s, criterion_s, _alpha_l1, optimizer_s, dls, n_epochs_alt, mode='labeled_mse', net_path=net_path_s)
            s = net_s(x).detach().sum(1)

            plt.figure(1)
            plt.subplot(312); plt.gca().cla()
            plt.plot(epoch_t, x[tidx].T, lw=4, c='grey', label=channels[0])
            plt.plot(epoch_t, s[tidx].T, label='Signal')
            plt.ylabel('Noise-reduced\n' + r'EEG [$\mu$V]')
            plt.subplot(313); plt.gca().cla()
            plt.plot(epoch_t, x[tidx].T, lw=4, c='grey', label=channels[0])
            plt.plot(epoch_t, n[tidx].T, label='Noise')
            plt.ylabel('Noise-related\n' + r'EEG [$\mu$V]')
            plt.pause(.1)

            plt.figure(2).clf()
            plt.subplot(121)
            plt.semilogy(loss_s)
            plt.subplot(122)
            plt.semilogy(loss_n)
            plt.pause(.1)

    noise_estimator = danet.DAnet(net=net_n, sfreq=sfreq, norm=False)
    signal_estimator = danet.DAnet(net=net_s, sfreq=sfreq, norm=False)
    sig = x[tidx].numpy()
    fig_s = plt.figure(1); fig_s.clf()
    danet.viz_decomposer(fig_s, signal_estimator, sig, epoch_t=epoch_t, plot_interval=[0, length])
    fig_n = plt.figure(2); fig_n.clf()
    danet.viz_decomposer(fig_n, noise_estimator, sig, epoch_t=epoch_t, plot_interval=[0, length])
