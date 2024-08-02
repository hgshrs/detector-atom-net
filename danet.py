import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics
from sklearn.base import BaseEstimator
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import moabb.datasets
import moabb.paradigms
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation

class decomposer(nn.Module):
    def __init__(self, n_components, param_detector={}, param_atom={}):
        super(decomposer, self).__init__()
        # self.conv_layers = make_filter_layers()
        self.n_components = n_components
        self.param_detector = param_detector
        self.param_atom = param_atom
        layers = []
        for cc in range(n_components):
            layers.append(nn.ModuleList([
                create_detector(**param_detector),
                create_atom(**param_atom),
                ]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        out = torch.zeros([x.shape[0], self.n_components, x.shape[1]])
        for ll, layer in enumerate(self.layers):
            _x = x[:, np.newaxis, ]
            _x = layer[0](_x) # detector
            _x = layer[1](_x) # kernel
            # out[:, ll, :] = _x.view(-1, _x.size(1) * _x.size(2))
            out[:, ll, :] = _x.squeeze()
        return out

def create_detector(kernel_size=25, n_mid_channels=4, n_mid_layers=2):
    layer_list = [
            nn.Conv1d(1, n_mid_channels, kernel_size, padding='same'),
            nn.ReLU(),
            ]
    for ll in range(n_mid_layers):
        layer_list += [
            nn.Conv1d(n_mid_channels, n_mid_channels, kernel_size, padding='same'),
            nn.ReLU(),
            ]
    layer_list += [
        nn.Conv1d(n_mid_channels, 1, kernel_size, padding='same'),
        nn.ReLU(),
        ]
    layers = nn.Sequential(*layer_list)
    return layers

def create_atom(kernel_size=25):
    layers = nn.Sequential(
        # nn.Conv1d(1, 1, kernel_size, bias=True, padding='same'),
        nn.Conv1d(1, 1, kernel_size, bias=False, padding='same'),
        )
    return layers 

def load_net(device, n_components, param_detector, param_atom, net_path='tmp/tmp', reassign_atoms=False, verbose=False):
    net = decomposer(n_components, param_detector, param_atom).to(device)
    try:
        net.load_state_dict(torch.load(net_path + '.pth', map_location=torch.device(device)))
        if verbose:
            print('Succeeded to load the saved model. [{}]'.format(net_path))
    except:
        if verbose:
            print('Failed to load the saved model. [{}]'.format(net_path))

    if reassign_atoms:
        load_components, atom_power = index_effective_atoms(net)
        if len(load_components) < net.n_components:
            # load_components = [2, 5, 6, 7]
            net_init = decomposer(n_components, param_detector, param_atom).to(device)
            orig_dict = net.state_dict()
            init_dict = net_init.state_dict()

            # div_atom_idx = atom_power.argmax()
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

class DAnet(BaseEstimator):
    def __init__(self, net, sfreq=250, norm=True, outtype='decomposed', output_components=np.array([]), update_oc_by_fit=False):
        net.eval()
        self.net = net
        self.sfreq = sfreq
        self.outtype = outtype # outtype in  {'decoposed', 'reconstructed', 'detector'}
        self.norm = norm
        if output_components.shape[0] == 0:
            self.output_components = np.arange(self.net.n_components)
        else:
            self.output_components = output_components
        self.update_oc_by_fit = update_oc_by_fit # if True, decompose traning samples and remove atoms which always output zero from the decomposer

    def fit(self, X, y):
        if self.update_oc_by_fit:
            _X = X
            D = self.transform(_X, outtype='decomposed')
            p = np.abs(D.reshape(_X.shape[0], _X.shape[1], len(self.output_components), _X.shape[2])).max(3).max(1).max(0)
            zc = np.where(p > 1e-6)[0]
            self.output_components = self.output_components[zc]
            if len(self.output_components) == 0:
                self.output_components = np.arange(self.net.n_components)
        return self

    def transform(self, X, outtype=None):
        if outtype == None:
            outtype = self.outtype
        # X: n_samples x n_channels x n_times
        X_fl = torch.from_numpy(X.astype(np.float32).reshape(X.shape[0] * X.shape[1], -1))
        if self.norm:
            p = torch.norm(X_fl, dim=1)
            p[p < 1e-12] = 1e-12
            X_fl = X_fl / p.unsqueeze(1).repeat(1, X.shape[2])
        if outtype == 'decomposed':
            D = self.net(X_fl)[:, self.output_components]
            if self.norm:
                D = D * p.unsqueeze(1).unsqueeze(2).repeat(1, len(self.output_components), X.shape[2])
            o = D.reshape(X.shape[0], X.shape[1] * len(self.output_components), X.shape[2])
        elif outtype == 'detector':
            D = torch.zeros([X.shape[0] * X.shape[1], len(self.output_components), X.shape[2]])
            for cc1, cc2 in enumerate(self.output_components):
                for ss in range(X_fl.shape[0]):
                    det = self.net.layers[cc2][0](X_fl[ss].unsqueeze(0))
                    D[ss, cc1] = det
            o = D.reshape(X.shape[0], X.shape[1] * len(self.output_components), X.shape[2])
        elif outtype == 'reconstructed':
            D = self.net(X_fl)[:, self.output_components]
            if self.norm:
                D = D * p.unsqueeze(1).unsqueeze(2).repeat(1, len(self.output_components), X.shape[2])
            o = D.sum(1).reshape(X.shape[0], X.shape[1], X.shape[2])
        return o.detach().numpy().astype(np.float64)

    def reshape2decomposed(self, D):
        return D.reshape(D.shape[0], int(D.shape[1] / len(self.output_components)), len(self.output_components), D.shape[2])

def viz_decomposer(fig, decomposer, sig, epoch_t, plot_interval=[0, 1]):
    fig.clf()
    n_components = len(decomposer.output_components)
    slen = len(sig)
    interval_tp = np.arange(np.abs(epoch_t - plot_interval[0]).argmin(), np.abs(epoch_t - plot_interval[1]).argmin())
    decsig = decomposer.transform(sig[np.newaxis, np.newaxis, :])[0]
    detout = decomposer.transform(sig[np.newaxis, np.newaxis, :], outtype='detector')[0]
    mae = sklearn.metrics.mean_absolute_error(sig[interval_tp], decsig.sum(0)[interval_tp])
    nmae = mae / sklearn.metrics.mean_absolute_error(sig[interval_tp], np.zeros_like(sig[interval_tp]))
    cs = sklearn.metrics.pairwise.cosine_similarity(sig[np.newaxis, interval_tp], decsig.sum(0)[np.newaxis, interval_tp])[0, 0]
    txt = '[Similarity measure] MAE: {:.2e}, NMAE: {:.2%}, Cos: {:.2f}'.format(mae, nmae, cs)
    print(txt)

    tick_fontsize = 10
    gs = gridspec.GridSpec(n_components+1, 5, figure=fig)
    # ax = fig.add_subplot(n_components + 1, 3, 27)
    ax = fig.add_subplot(gs[n_components, 3:5])
    lines = ax.plot(epoch_t[interval_tp], decsig[:, interval_tp].T, '--', alpha=.0)
    # ax.plot(epoch_t, sig - decsig.sum(0), '--', label='Residue')
    ax.plot(epoch_t[interval_tp], sig[interval_tp], 'k', alpha=.2, lw=4, label='Original')
    ylim = ax.get_ylim()
    ax.plot(epoch_t[interval_tp], decsig.sum(0)[interval_tp], 'k', label='Reconstructed')
    ax.set_ylim(ylim)
    ax.set_xlabel('Time [s]')
    # ax.set_ylabel(r'[$\mu$V]')
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], r'[$\mu$V]', fontsize=tick_fontsize, ha='right')
    ax.set_yticks([-10, 0, 10])
    ax.set_yticklabels([-10, 0, 10], fontsize=tick_fontsize)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([0, 1], fontsize=tick_fontsize)

    atoms = []
    for cc in range(n_components):
        atom = decomposer.net.state_dict()['layers.{}.1.0.weight'.format(decomposer.output_components[cc])].numpy().squeeze()
        atoms.append(atom)
    atoms = np.array(atoms).astype(np.float64)
    atom_t = np.arange(atoms[cc].shape[0]) / sfreq

    for cc in range(n_components):
        c = lines[cc].get_color()

        # ax_atom = fig.add_subplot(n_components + 1, 3, (cc + 0) * 3 + 1)
        ax_atom = fig.add_subplot(gs[cc, 0])
        # ax_atom.plot([epoch_t[0], epoch_t[len(atom)]], [0, 0], '--', c='k')
        ax_atom.plot(atom_t, atoms[cc], c=c)
        # if np.abs(atom).max() < .1:
            # ax_atom.set_yticks([-.1, 0, .1])
        ax_atom.set_ylim(-atoms.max(), atoms.max())
        ax_atom.set_yticks([0])
        ax_atom.set_yticklabels([0], fontsize=tick_fontsize)
        ax_atom.set_xticks([0, .5])
        ax_atom.set_xticklabels([0, .5], fontsize=tick_fontsize)

        # ax_det = fig.add_subplot(n_components + 1, 3, (cc + 0) * 3 + 2)
        ax_det = fig.add_subplot(gs[cc, 1:3])
        # ax_det.plot(epoch_t, detout[cc], c=c)
        ax_det.plot(epoch_t[interval_tp], detout[cc, interval_tp], c=c)
        ax_det.set_yticks([0])
        ax_det.set_yticklabels([0], fontsize=tick_fontsize)
        ax_det.set_ylim(-0.002, detout.max() + 1 / sfreq)
        ax_det.set_xticks([0, 1])
        ax_det.set_xticklabels([0, 1], fontsize=tick_fontsize)

        # ax_dec = fig.add_subplot(n_components + 1, 3, (cc + 0) * 3 + 3)
        ax_dec = fig.add_subplot(gs[cc, 3:5])
        ax_dec.plot(epoch_t[interval_tp], sig[interval_tp], 'k', alpha=.2)
        ax_dec.plot(epoch_t[interval_tp], decsig[cc, interval_tp], c=c)
        ax_dec.set_xticklabels([])
        ax_dec.set_yticklabels([])
        ax_dec.set_ylim(ylim)
        # ax_dec.set_xlim(plot_interval)
        # ax_dec.set_xlim([-.1, 1.1])

        if cc < n_components - 1:
            ax_atom.set_yticklabels([])
            ax_atom.set_xticklabels([])
            ax_det.set_yticklabels([])
            ax_det.set_xticklabels([])
        else:
            ax_atom.set_ylabel('Amplitude')
    return decsig, detout, atoms, mae, cs

if __name__=='__main__':
    net_path = 'danet_parameters'
    param_detector = {'kernel_size':125, 'n_mid_channels':8, 'n_mid_layers':4}
    param_atom = {'kernel_size':125}
    n_components = 8
    sfreq = 250
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    net = load_net(device, n_components, param_detector, param_atom, net_path=net_path, verbose=True)
    dcm = DAnet(net=net, sfreq=sfreq, norm=True, outtype='decomposed', update_oc_by_fit=False)

    # =======================================
    print('\nVisualize pre-trained decomposer with a randomly-generated artificial signal')
    # =======================================
    time = np.arange(-1, 2, 1 / sfreq)
    sig = np.random.normal(loc=.0, scale=10., size=time.shape[0])
    # detout = dcm.transform(sig[np.newaxis, np.newaxis, :], outtype='detector')[0]
    # decsig = dcm.transform(sig[np.newaxis, np.newaxis, :])[0]
    fig = plt.figure(1, figsize=(6.4, 4.8 * 1.5))
    viz_decomposer(fig, dcm, sig, time)
    fig.suptitle('Random signal')
    print('The pre-trained decomposer is not for (full-band) random signal, so it may not work well.')

    # =======================================
    print('\nVisualize pre-trained decomposer with an actual EEG signal')
    # =======================================
    dataset = moabb.datasets.BNCI2014_001()
    sbj = dataset.subject_list[0]
    print('Dataset: {}, subject: {}'.format(dataset.code, sbj))
    paradigm = moabb.paradigms.MotorImagery(resample=sfreq, channels=['C3'], tmin=-1, tmax=dataset.interval[1])
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[sbj]) # X (signal): trial x channel x time-sample, y (label): trial
    sig = X[0, 0, :]
    fig = plt.figure(2, figsize=(6.4, 4.8 * 1.5))
    viz_decomposer(fig, dcm, sig, time)
    fig.suptitle('EEG signal from {}'.format(dataset.code))

    # =======================================
    print('\nClassification with a estimator with pre-trained decomposer')
    # =======================================
    dcm = DAnet(net=net, sfreq=sfreq, norm=True, outtype='decomposed', update_oc_by_fit=True)
    components = [
            CSP(n_components=8),
            LDA(), ]
    pipelines = {
            'without':make_pipeline(*components),
            'with':make_pipeline(dcm, *components), }
    evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=[dataset], overwrite=True)
    res = evaluation.process(pipelines)
    print('Without decomposition:\t{:.2%}'.format(res[res['pipeline'] == 'without']['score'].mean()))
    print('With decomposition:\t{:.2%}'.format(res[res['pipeline'] == 'with']['score'].mean()))
