from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


GRAPH_DTYPES = {
    'nf': torch.float32, 'nt': torch.long, 'ei': torch.long,
    'ety': torch.long, 'et': torch.long, 'pos': torch.float32,
    'times': torch.float32, 'offsets': torch.long,
}


class RadarDataset(Dataset):
    """Aligned radar, conditions, and one graph per 8-frame sample."""

    def __init__(self, root, split, settings, segment_len, grid_size):
        directory = Path(root) / split
        self.radar = np.load(directory / settings['radar'], mmap_mode='r', allow_pickle=False)
        self.conditions = np.load(directory / settings['conditions'], mmap_mode='r', allow_pickle=False)
        self.segment_len = segment_len
        self.grid_size = grid_size
        if self.radar.ndim != 4 or self.radar.shape[1:] != (segment_len, 4 * grid_size, 4 * grid_size):
            raise ValueError('Radar must have shape [N,T,4*S,4*S] (default [N,8,128,128]).')
        if self.conditions.shape != (len(self.radar), segment_len, grid_size, grid_size, 5):
            raise ValueError('Conditions must have shape [N,T,S,S,5] (default [N,8,32,32,5]).')
        self.graph_paths = sorted((directory / settings['graphs']).glob('*.pt'))
        if len(self.graph_paths) != len(self.radar) or not self.graph_paths:
            raise ValueError('Expected exactly N graphs, one per sample.')
        for idx, path in enumerate(self.graph_paths):
            if path.name != f'{idx:06d}.pt':
                raise ValueError(f'Expected graph {idx:06d}.pt, got {path.name}.')

    def __len__(self):
        return len(self.radar)

    def __getitem__(self, idx):
        target = np.asarray(self.radar[idx], dtype=np.float32)
        graph = torch.load(self.graph_paths[idx], map_location='cpu', weights_only=True)
        sample = {key: graph[key].to(dtype=dtype) for key, dtype in GRAPH_DTYPES.items()}
        if sample['nf'].ndim != 2 or sample['nf'].shape[1] != 5:
            raise ValueError('Graph nf must have shape [num_nodes,5].')
        if sample['times'].shape != (self.segment_len,) or sample['offsets'].shape != (self.segment_len + 1,):
            raise ValueError('Graph time/offset fields must match the configured segment length.')
        for frame in range(self.segment_len):
            a, b = sample['offsets'][frame:frame + 2].tolist()
            types = sample['nt'][a:b]
            if b - a < self.grid_size ** 2 or not torch.all(types[:self.grid_size ** 2] == 0) or not torch.all(types[self.grid_size ** 2:] == 1):
                raise ValueError('Every frame must store all row-major G nodes before its C nodes.')
        sample['target'] = torch.from_numpy(target.copy())
        conditions = np.asarray(
            self.conditions[idx], dtype=np.float32,
        )
        # The advection loss reads U/V at channels 1/2 of [T,5,S,S].
        sample['input'] = torch.from_numpy(conditions.copy()).permute(0, 3, 1, 2).contiguous()
        return sample


def collate_fn(batch):
    if len(batch) != 1:
        raise ValueError('The source model processes one heterogeneous graph sample at a time.')
    return batch[0]
