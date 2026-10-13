import torch
from torch import nn
from torch.nn import functional as F


def _extract_wind_from_batch(batch, u_channel_idx=1, v_channel_idx=2):
    """Select U/V from the single-segment input [T,5,h,w]."""
    x = batch['input']
    if x.dim() != 4 or x.shape[1] != 5:
        raise ValueError("batch['input'] must have shape [T,5,h,w].")
    return x[:, [u_channel_idx, v_channel_idx], :, :].unsqueeze(0)

def _ensure_bthw(x):
    """
    Convert input tensor to [B, T, H, W].
    Supported:
        [T, H, W]
        [B, T, H, W]
    """
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f'Expected tensor with shape [T,H,W] or [B,T,H,W], got {x.shape}')

class TieredWeightedL1Loss(nn.Module):
    """
    Tiered weighted L1 for intensity-level-aware regression.

    For 0-70 dBZ normalized to [-1, 1]:
        20 dBZ -> -0.4286
        30 dBZ -> -0.1429
        40 dBZ ->  0.1429
        50 dBZ ->  0.4286
        60 dBZ ->  0.7143

    Default weight rule:
        target <= 20 dBZ: weight = 1.0
        target >  20 dBZ: weight = 1.5
        target >  30 dBZ: weight = 2.5
        target >  40 dBZ: weight = 4.0
        target >  50 dBZ: weight = 6.0
        target >  60 dBZ: weight = 8.0
    """

    def __init__(self, thresholds=(-0.4286, -0.1429, 0.1429, 0.4286, 0.7143), weights=(1.5, 2.5, 4.0, 6.0, 8.0), base_weight=1.0):
        super().__init__()
        assert len(thresholds) == len(weights), 'thresholds and weights must have the same length.'
        self.thresholds = [float(x) for x in thresholds]
        self.weights = [float(x) for x in weights]
        self.base_weight = float(base_weight)

    def forward(self, pred, target):
        pred = _ensure_bthw(pred)
        target = _ensure_bthw(target)
        weight_map = torch.ones_like(target) * self.base_weight
        for (threshold, weight) in zip(self.thresholds, self.weights):
            mask = target > threshold
            weight_map = weight_map.masked_fill(mask, float(weight))
        return torch.mean(weight_map * torch.abs(pred - target))

class PhysicsAdvectionResidualLoss(nn.Module):
    """
    Advection residual consistency loss.

    R(A) = dA/dt + vx * dA/dx + vy * dA/dy

    Loss = |R(pred) - R(target)|_1
    """

    def __init__(self, u_channel_idx=1, v_channel_idx=2, wind_scale=1.0, dt=1.0, dx=1.0, dy=1.0):
        super().__init__()
        self.u_channel_idx = int(u_channel_idx)
        self.v_channel_idx = int(v_channel_idx)
        self.wind_scale = float(wind_scale)
        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)

    def _residual(self, A, wind_lr):
        """
        A:
            [B, T, H, W]
        wind_lr:
            [B, T, 2, h, w]

        Returns:
            residual [B, T-1, H-2, W-2]
        """
        (B, T, H, W) = A.shape
        wind_hr = F.interpolate(wind_lr.reshape(B * T, 2, wind_lr.shape[-2], wind_lr.shape[-1]), size=(H, W), mode='bilinear', align_corners=False).reshape(B, T, 2, H, W)
        vx = wind_hr[:, :-1, 0, 1:-1, 1:-1] * self.wind_scale
        vy = wind_hr[:, :-1, 1, 1:-1, 1:-1] * self.wind_scale
        dA_dt = (A[:, 1:, 1:-1, 1:-1] - A[:, :-1, 1:-1, 1:-1]) / self.dt
        A_now = A[:, :-1]
        dA_dx = (A_now[..., 1:-1, 2:] - A_now[..., 1:-1, :-2]) / (2.0 * self.dx)
        dA_dy = (A_now[..., 2:, 1:-1] - A_now[..., :-2, 1:-1]) / (2.0 * self.dy)
        residual = dA_dt + vx * dA_dx + vy * dA_dy
        return residual

    def forward(self, pred, target, batch):
        pred = _ensure_bthw(pred)
        target = _ensure_bthw(target)
        wind_lr = _extract_wind_from_batch(batch, u_channel_idx=self.u_channel_idx, v_channel_idx=self.v_channel_idx).to(device=pred.device, dtype=pred.dtype)
        if wind_lr.shape[1] != pred.shape[1]:
            raise ValueError(f'Wind sequence length {wind_lr.shape[1]} does not match pred length {pred.shape[1]}.')
        pred_residual = self._residual(pred, wind_lr)
        target_residual = self._residual(target, wind_lr)
        return F.l1_loss(pred_residual, target_residual)

class MultiThresholdMaskBCELoss(nn.Module):
    """
    Multi-threshold boundary loss.

    For each threshold tau_k:
        M_k = 1(target > tau_k)
        z_k = (pred - tau_k) / temperature
        BCEWithLogits(z_k, M_k)

    This encourages clear boundaries among different intensity levels.
    """

    def __init__(self, thresholds=(-0.4286, -0.1429, 0.1429, 0.4286, 0.7143), temperature=0.05, threshold_weights=(1.0, 1.2, 1.5, 2.0, 2.5)):
        super().__init__()
        assert len(thresholds) == len(threshold_weights), 'thresholds and threshold_weights must have the same length.'
        self.thresholds = [float(x) for x in thresholds]
        self.temperature = float(temperature)
        self.threshold_weights = [float(x) for x in threshold_weights]

    def forward(self, pred, target):
        pred = _ensure_bthw(pred)
        target = _ensure_bthw(target)
        total = pred.new_tensor(0.0)
        weight_sum = 0.0
        for (threshold, weight) in zip(self.thresholds, self.threshold_weights):
            target_mask = (target > threshold).float()
            logits = (pred - threshold) / self.temperature
            loss = F.binary_cross_entropy_with_logits(logits, target_mask, reduction='mean')
            total = total + float(weight) * loss
            weight_sum += float(weight)
        return total / max(weight_sum, 1e-06)

class CompositeRadarLoss(nn.Module):
    """The three active paper terms, without historical ablation losses."""

    def __init__(self, weighted_l1_weight, physics_weight, rain_mask_weight,
                 tiered, physics, mask):
        super().__init__()
        self.weighted_l1_weight = float(weighted_l1_weight)
        self.physics_weight = float(physics_weight)
        self.rain_mask_weight = float(rain_mask_weight)
        self.weighted_l1 = TieredWeightedL1Loss(**tiered)
        self.physics = PhysicsAdvectionResidualLoss(**physics)
        self.rain_mask = MultiThresholdMaskBCELoss(**mask)

    def forward(self, pred, target, batch):
        total = pred.new_tensor(0.0)
        total = total + self.weighted_l1_weight * self.weighted_l1(pred, target)
        total = total + self.physics_weight * self.physics(pred, target, batch=batch)
        total = total + self.rain_mask_weight * self.rain_mask(pred, target)
        return total
