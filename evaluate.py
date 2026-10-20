import argparse
import json
import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

from train import build_model, load_config, make_loader, predict


def csi_from_counts(hit, miss, false_alarm):
    denominator = hit + miss + false_alarm
    if denominator == 0:
        return 1.0
    return hit / denominator


def hss_from_counts(hit, miss, false_alarm, correct_negative):
    numerator = 2.0 * (hit * correct_negative - miss * false_alarm)
    denominator = ((hit + miss) * (miss + correct_negative)
                   + (hit + false_alarm) * (false_alarm + correct_negative))
    if denominator == 0:
        return 1.0
    return numerator / denominator


def contingency_counts(pred, target, threshold):
    p, y = pred > threshold, target > threshold
    return np.array([
        np.logical_and(p, y).sum(), np.logical_and(~p, y).sum(),
        np.logical_and(p, ~y).sum(), np.logical_and(~p, ~y).sum(),
    ], dtype=np.int64)


def bootstrap_csi(counts, samples=1000, seed=42, percentiles=(2.5, 97.5)):
    """Resample whole graph segments and recompute CSI from summed counts.

    Adapted from the existing 8-frame bootstrap analysis. `counts` has shape
    [num_segments, num_thresholds, 4]; every segment is one sampling unit.
    The RNG, shared indices across thresholds and percentile CI are preserved.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[2] != 4 or len(counts) == 0 or samples <= 0:
        raise ValueError('Expected nonempty [N,K,4] counts and positive bootstrap_samples.')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(counts), size=(samples, len(counts)), dtype=np.int64)
    results = []
    for threshold_idx in range(counts.shape[1]):
        sampled = counts[:, threshold_idx, :][indices].sum(axis=1).astype(np.float64)
        hit, miss, false_alarm = sampled[:, 0], sampled[:, 1], sampled[:, 2]
        denominator = hit + miss + false_alarm
        # Preserve the source's denominator-zero value while avoiding a NumPy warning.
        values = np.divide(hit, denominator, out=np.ones_like(hit), where=denominator != 0)
        low, high = np.percentile(values, percentiles)
        totals = counts[:, threshold_idx, :].sum(axis=0).astype(np.float64)
        results.append(dict(estimate=float(csi_from_counts(*totals[:3])),
                            ci_low=float(low), ci_high=float(high)))
    return results


class MetricAccumulator:
    """Original global errors/counts and mean of per-segment frame SSIM."""

    def __init__(self, settings):
        self.settings = settings
        self.absolute_error = self.squared_error = self.ssim_sum = self.psnr_sum = 0.0
        self.num_pixels = self.num_samples = 0
        self.counts = []

    @torch.no_grad()
    def update(self, pred, target):
        pred = torch.as_tensor(pred).detach().float().cpu()
        target = torch.as_tensor(target).detach().float().cpu()
        if pred.shape != target.shape or pred.ndim != 3 or not pred.numel():
            raise ValueError('Prediction and target must have matching [T,H,W] shapes.')
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError('Metrics require finite prediction/target values.')
        diff = pred - target
        self.absolute_error += diff.abs().sum().item()
        self.squared_error += (diff ** 2).sum().item()
        self.num_pixels += diff.numel()
        self.num_samples += 1
        mse = ((target * 0.5 - pred * 0.5) ** 2).mean()
        self.psnr_sum += (-10.0 * torch.log10(mse + self.settings['psnr_epsilon'])).item()
        p, y = pred.numpy(), target.numpy()
        self.ssim_sum += float(np.mean([
            structural_similarity(y[t], p[t], data_range=self.settings['data_range'])
            for t in range(len(p))
        ]))
        self.counts.append(np.stack([contingency_counts(p, y, thr) for thr in self.settings['thresholds']]))

    def finalize(self):
        if not self.num_samples:
            raise ValueError('No evaluation samples were supplied.')
        counts = np.stack(self.counts)
        totals = counts.sum(axis=0).astype(np.float64)
        intervals = bootstrap_csi(
            counts, self.settings['bootstrap_samples'], self.settings['bootstrap_seed'],
            self.settings['bootstrap_percentiles'],
        )
        results = dict(
            PSNR=self.psnr_sum / self.num_samples,
            SSIM=self.ssim_sum / self.num_samples,
            MAE=self.absolute_error / self.num_pixels,
            RMSE=math.sqrt(self.squared_error / self.num_pixels),
        )
        results['CSI'], results['HSS'], results['Bootstrap CSI'] = {}, {}, {}
        for idx, thr in enumerate(self.settings['thresholds']):
            results['CSI'][str(thr)] = float(csi_from_counts(*totals[idx, :3]))
            results['HSS'][str(thr)] = float(hss_from_counts(*totals[idx]))
            results['Bootstrap CSI'][str(thr)] = intervals[idx]
        return results


@torch.no_grad()
def evaluate(model, loader, device, settings):
    model.eval()
    accumulator = MetricAccumulator(settings)
    for sample in loader:
        sample = {key: value.to(device) for key, value in sample.items()}
        accumulator.update(predict(model, sample), sample['target'])
    return accumulator.finalize()


def main():
    parser = argparse.ArgumentParser(description='Evaluate the seven paper metrics.')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                        help='Data split to evaluate (default: test).')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.data_root:
        config['data']['root'] = args.data_root
    device = torch.device(args.device or config['evaluation']['device'])
    model = build_model(config['model']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=True)
    loader = make_loader(config, config['data'][f'{args.split}_split'], False)
    results = evaluate(model, loader, device, config['evaluation'])
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
