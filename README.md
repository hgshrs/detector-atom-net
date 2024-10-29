This repo opens a demo code of the following article.
- H. Higashi, ``Single-channel electroencephalography decomposition by detector-atom network and its pre-trained model,'' arXiv:2408:02185.
- https://arxiv.org/abs/2408.02185

# Requirements

| Package | Version |
| ---- | ---- |
| python | 3.8.18 |
| numpy | 1.24.3 |
| pytorch | 1.8.0 |
| scikit-learn | 1.3.2 |
| matplotlib | 1.24.3 |
| moabb | 1.0.0 |
| mne | 1.6.0 |
| tqdm | 4.65.0 |

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
