This repo opens a demo code of the following article.
- H. Higashi, ``Single-channel electroencephalography decomposition by detector-atom network and its pre-trained model,'' arXiv:2408:02185.
- https://arxiv.org/abs/2408.02185

# Requirements

| Package | Version |
| ---- | ---- |
| python | 3.11.7 |
| numpy | 1.26.3 |
| pytorch | 2.1.2 |
| scikit-learn | 1.4.0 |
| matplotlib | 3.8.2 |
| moabb | 1.0.0 |
| mne | 1.6.1 |
| tqdm | 4.66.1 |

# Use our pre-trained model
See and execute
```
python danet.py
```

# Train a decomposer with your-own-dataset
See and execute
```
python train.py
```

# Train a decomposer for a shared atom with your-own-dataset
See and execute
```
python train_shared_atoms.py
```

# Train a decomposer for time-locked components with your-own-dataset
See and execute
```
python train_time_locked.py
```

# Train a decomposer for noise reduction with your-own-dataset
See and execute
```
python train_noise_remover.py
```
