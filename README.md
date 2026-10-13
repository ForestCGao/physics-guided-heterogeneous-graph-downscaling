# HGT–ODE Radar Downscaling

Minimal research implementation of heterogeneous graph construction, Dense HGT
encoding, physics-guided ODE/GRU evolution, weighted HGT/ODE fusion, CNN decoding,
training, and evaluation. Each sample contains eight aligned frames for spatial
downscaling from 32×32 conditions to 128×128 radar reflectivity.

## Repository structure

- `models/`: Dense HGT, physics-guided ODE/GRU, weighted fusion, and CNN decoder.
- `graph/make_graph.py`: heterogeneous graph construction.
- `utils/`: dataset loading and composite loss.
- `train.py`, `evaluate.py`: training and evaluation entry points.
- `configs/default.yaml`: data paths and scientific settings.
- `LICENSE`, `THIRD_PARTY_LICENSES.md`: licenses and attribution.

## Requirements

Use Python 3.9 or newer with a compatible PyTorch installation:

```bash
python -m pip install -r requirements.txt
```

## Data format

Organize the data root into independent splits:

```text
data_root/
├── train/
├── val/
└── test/
```

`train/` is used for optimization, `val/` for validation and early stopping,
and `test/` for final evaluation. Each split contains `radar.npy`,
`conditions.npy`, and `graphs/`. Arrays are finite, aligned `float32` values;
the number of samples `N` may differ between splits:

```text
radar.npy       [N, 8, 128, 128]
conditions.npy  [N, 8, 32, 32, 5]
```

Condition channels:

0. Low-resolution radar reflectivity
1. 850 hPa zonal wind U
2. 850 hPa meridional wind V
3. 850 hPa relative humidity
4. 850 hPa temperature

Supply normalized arrays: radar reflectivity `z` in 0–70 dBZ uses `z / 35 - 1`;
condition channels encoded as `p` in 0–250 use `p / 125 - 1`. Obtain the
meteorological encoding ranges from your data provider. The loader consumes
these normalized values directly.

Dataset item `n` loads `radar[n]`, `conditions[n]`, and `graphs/{n:06d}.pt`;
the dataset length is `N`. The target has shape `[8,128,128]`. Training receives
the aligned conditions as `batch['input']` with shape `[8,5,32,32]`, with U/V
at channels 1/2 for the physics loss.

## Graph construction

Set `data.root` in `configs/default.yaml`. Run this Python example from the
repository directory for all three configured splits. Each sample produces one graph,
named `000000.pt`, `000001.pt`, and so on.

```python
from pathlib import Path
import numpy as np
import torch
from graph.make_graph import make_graph
from train import load_config

cfg = load_config('configs/default.yaml')
for name in ('train_split', 'val_split', 'test_split'):
    split = Path(cfg['data']['root']) / cfg['data'][name]
    x = np.load(split / cfg['data']['conditions'], mmap_mode='r', allow_pickle=False)
    out = split / cfg['data']['graphs']
    out.mkdir(parents=True, exist_ok=True)
    assert x.ndim == 5 and x.shape[1:] == (8, 32, 32, 5)
    for n in range(len(x)):
        graph = make_graph(x[n], **cfg['graph'])
        torch.save(graph, out / f'{n:06d}.pt')
```

## Training

```bash
python train.py --config configs/default.yaml --data_root /path/to/your/data
```

Defaults: hidden dimension 192, 3 HGT layers, 2 attention heads, Euler integration,
Adam, learning rate `1e-4`, and effective batch size 8 (one sample per loader batch
with 8 gradient-accumulation steps in a full single-process window).

Fusion is `H_HGT + residual_alpha * H_ODE_GRU`, with learned `residual_alpha`
initialized to `0.1`. `PixelDecoderHeadNoBatch02` produces `[8,128,128]` outputs.
The composite loss is `1.0 * L_tiered + 0.05 * L_physics + 0.2 * L_mask`.
Checkpoints are saved under `checkpoints/`; use `--resume /path/to/checkpoint.pt`
to resume training.

## Evaluation

Metrics: PSNR, SSIM, MAE, RMSE, CSI, HSS, and Bootstrap CSI.

```bash
python evaluate.py --config configs/default.yaml \
  --checkpoint ./checkpoints/latest.pt --data_root /path/to/your/data
```

The command evaluates the configured `test_split` by default and prints JSON.
Use `--split train`, `--split val`, or `--split test` to select a configured split.
Add `--device cpu` for CPU evaluation.

## Third-party code

The HGT encoder is adapted from [acbull/pyHGT](https://github.com/acbull/pyHGT)
(MIT). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the complete
upstream notice.
