import importlib
import moabb.datasets
from moabb.paradigms import FixedIntervalWindowsProcessing
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt
import danet
importlib.reload(danet)

def set_processing(length, sfreq, band, baseline):
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
        channels=None,
    )
    return processing

def train(device, net, loss, criterion, alpha_l1, optimizer, dataloader, n_epochs,
        mode='unsupervised', reassign_atoms_every=None,
        net_path='', save_every=None):

    if reassign_atoms_every == None:
        reassign_atoms_every = n_epochs
    if save_every == None:
        save_every = n_epochs
    if net_path == '':
        save_every = n_epochs

    n_total_epochs = loss.shape[0]
    loss = torch.cat([loss, torch.zeros([n_epochs, 3])], axis=0)
    for epoch in tqdm(range(n_epochs)):
        batch_loss = torch.zeros([len(dataloader), 3])
        for itr, batch_data in enumerate(dataloader):
            if mode == 'unsupervised':
                eeg = batch_data.to(device)
            elif mode == 'supervised':
                eeg, y = batch_data
            net.zero_grad()
            decsig = net(eeg).to(device)
            if mode == 'unsupervised':
                main_loss = criterion(eeg, decsig.sum(1)).mean()
            elif mode == 'supervised':
                main_loss = criterion(eeg, decsig, y)

            det_l1_loss = torch.tensor(0., requires_grad=True)
            for cc in range(net.n_components):
                if alpha_l1 > 0:
                    det = net.layers[cc][0](eeg[:, np.newaxis],).squeeze()
                    nl = torch.norm(det, 1, dim=1)
                    det_l1_loss = det_l1_loss + nl[torch.logical_not(torch.isnan(nl))].mean()

            total_loss = main_loss + alpha_l1 * det_l1_loss
            total_loss.backward()
            optimizer.step()
            batch_loss[itr, 0] = total_loss.item()
            batch_loss[itr, 1] = main_loss.item()
            batch_loss[itr, 2] = det_l1_loss.item()
        loss[n_total_epochs + epoch] = batch_loss.sum(0)

        # Atom reassignment
        if (epoch % reassign_atoms_every == 0) & (epoch > 0):
            net = atom_reassign(net, device, verbose=True)

        # Save
        if (epoch % save_every == 0) & (epoch > 0):
            _loss = loss[:(n_total_epochs + epoch)]
            torch.save({'net':net.state_dict(), 'loss':_loss}, net_path)
    if net_path != '':
        torch.save({'net':net.state_dict(), 'loss':loss}, net_path)
    return net, loss

def atom_reassign(net, device, verbose=False):
    load_components, atom_power = index_effective_atoms(net)
    if len(load_components) < net.n_components:
        net_init = danet.decomposer(net.n_components, net.param_detector, net.param_atom).to(device)
        orig_dict = net.state_dict()
        init_dict = net_init.state_dict()

        div_atom_idx = np.random.permutation(load_components)[0]
        if verbose:
            print('Load weights of {}'.format(load_components))
            print('Initialize weights based on {}'.format(div_atom_idx))
        div_atom = orig_dict['layers.{}.1.0.weight'.format(div_atom_idx)].clone()
        orig_dict['layers.{}.1.0.weight'.format(div_atom_idx)][:] = 0.
        load_components.remove(div_atom_idx)
        n_divs = net.n_components - len(load_components)
        div_idxs = list(range(0, div_atom.shape[2], int(np.ceil(div_atom.shape[2] / n_divs)))) + [div_atom.shape[2]] * 2
        divided_atoms = []
        for dd in range(n_divs):
            _atom = div_atom.clone()
            _atom[:, :, div_idxs[dd + 1]:] = 0.
            div_atom[:, :, :div_idxs[dd + 1]] = 0.
            divided_atoms.append(_atom)

        d_count = 0
        for name, param in orig_dict.items():
            name_list = name.split('.')
            if int(name_list[1]) in load_components:
                init_dict[name].copy_(param)
                # print('loaded for {}'.format(name))
            else:
                if name_list[2] == '0':
                    # nlist = list(name)
                    # nlist[7] = str(div_atom_idx)
                    name_list[1] = str(div_atom_idx)
                    name2 = '{}.{}.{}.{}.{}'.format(*name_list)
                    init_dict[name].copy_(orig_dict[name2])
                elif name_list[2] == '1':
                    init_dict[name].copy_(divided_atoms[d_count])
                    d_count += 1
        net.load_state_dict(init_dict)
    return net

def index_effective_atoms(net):
    load_components = []
    atom_power = np.zeros(net.n_components)
    for cc in range(net.n_components):
        atom = net.state_dict()['layers.{}.1.0.weight'.format(cc)]
        atom_power[cc] = torch.norm(atom)
        if atom_power[cc] > 1e-4:
            load_components.append(cc)
    return load_components, atom_power

class supervised_loss(nn.Module):
    def __init__(self):
        super(supervised_loss, self).__init__()
        # self.class_labels = class_labels
        # self.n_components = len(class_labels)

    def forward(self, eeg, decsig, y):
        n_classes = y.shape[1]
        outeeg = torch.zeros_like(eeg)
        z2 = torch.zeros_like(eeg)
        loss2 = 0.
        for ss, class_labels in enumerate(y):
            class_idx = y[ss].argmax().item()
            outeeg[ss] = decsig[ss, class_idx]
            for diff_idx in np.setdiff1d(range(n_classes), [class_idx]):
                loss2 += torch.dot(decsig[ss][diff_idx], decsig[ss][diff_idx])
        z1 = eeg - outeeg
        loss1 = torch.dot(z1.reshape([-1]), z1.reshape([-1]))
        return loss1 + loss2

if __name__=='__main__':
    device = 'cpu'
    # device = 'cuda:0'
    print('Available CUDA device:', device) # must choose available cuda device or cpu

    sfreq = 250 # sampling frequency of input signals
    length = 3 # signal length [sec] for Fixedintervalwindows
    band = [.5, 100] # band for bandpass filter
    baseline = (0, length) # the time period for baseline calculation
    slen = int(sfreq * length) # number of time points

    net_path = 'tmp.pth' # path for saving decomposer network parameters
    n_components = 8 # number of the decomposed signals
    param_detector = {'kernel_size':125, 'n_mid_channels':8, 'n_mid_layers':4} # parameters for conv layers for detectors
    param_atom = {'kernel_size':125} # parameters for conv layers for atoms. kernel_size corresponds the signal length of an atom.
    alpha_l1 = 1e-5 # coefficient for the sparsity loss
    ns_epochs = [5000, 5000] # without, with sparsity
    # ns_epochs = [0, 0] # without, with sparsity
    lr = 1e-5
    beta1 = .5
    reassign_atoms_every = 10 # every this number of epochs, run the atom reassignment

    # =======================================
    print('\nGet epoched signals by FixedIntervalWindows...')
    # =======================================
    dataset = moabb.datasets.BNCI2014_001()
    processing = set_processing(length, sfreq, band, baseline) # set the parameters for get_data
    x, y, metadata = processing.get_data(dataset=dataset, subjects=[1,]) # size of tensor x: n_samples x n_channels x n_timepoints
    x = torch.from_numpy(x[:10]).float() # Reduce samples for demo purpose

    # DataLoader must include a signal set, x, which a torch tensor of n_samples x n_timepoints.
    # Because the dataset has multi-channel, the signals is reshaped from n_samples x n_channels x n_timepoints to (n_samples x n_channels) x n_timepoints.
    dl = torch.utils.data.DataLoader(x.view(x.shape[0] * x.shape[1], -1), batch_size=100, shuffle=True, num_workers=0)

    # =======================================
    print('\nTrain decomposer...')
    # =======================================
    net = danet.decomposer(n_components, param_detector, param_atom).to(device)
    net, loss = danet.load_net(net, n_components, param_detector, param_atom, net_path=net_path, with_loss=True, verbose=True)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, .999), weight_decay=1e-5)
    criterion = nn.MSELoss(reduction='none')
    for sse, n_epochs in enumerate(ns_epochs):
        if sse == 0:
            _alpha_l1 = 0.
            print('No sparsity loss...')
        elif sse == 1:
            _alpha_l1 = alpha_l1
            print('With sparsity loss...')
        net, loss = train(device, net, loss, criterion, _alpha_l1, optimizer, dl, n_epochs, reassign_atoms_every)
        torch.save({'net':net.state_dict(), 'loss':loss}, net_path)

    # =======================================
    # Decompose
    # =======================================
    net.eval()
    decomposed_signals = net(x[0, 0].view(1, -1)) # out sample x n_components x time point

    # =======================================
    # Visualize the training result
    # =======================================
    fig_loss = plt.figure(1); fig_loss.clf()
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.semilogy(loss)
    fig_loss.savefig('loss.pdf', bbox_inches='tight', transparent=True)
    fig_dec = plt.figure(2); fig_dec.clf()
    dcm = danet.DAnet(net=net, sfreq=sfreq, norm=True)
    sig = x[0, 0].numpy()
    danet.viz_decomposer(fig_dec, dcm, sig, epoch_t=np.arange(slen)/sfreq)
    fig_dec.savefig('atoms.pdf', bbox_inches='tight', transparent=True)
