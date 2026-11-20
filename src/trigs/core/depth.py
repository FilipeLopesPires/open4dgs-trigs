"""Depth supervision from measured (e.g. Azure Kinect ToF) depth maps.

The renderer produces, per pixel, the alpha-weighted depth sum
``D = sum_i w_i z_i`` and the accumulated opacity ``A = sum_i w_i`` (gsplat
``render_mode="RGB+D"``), with ``z`` the camera-space Z (same convention as
the target: Z along the optical axis, not ray length).

The target is usually coarser than the render (half-resolution Kinect depth
for full-resolution colour). It is never upsampled: ``D`` and ``A`` are area
pooled onto the target grid separately and divided afterwards, so every
target pixel is compared with the opacity-weighted mean surface depth over
exactly the render pixels it covers::

    ED = avgpool(D) / avgpool(A)

Pixels are supervised only where the target is valid, the (optional)
foreground mask is set, the pooled opacity is meaningful, and the target is
locally smooth (depth discontinuities and holes are eroded away: ToF mixes
foreground and background there).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

DEPTH_LOSS_TYPES = ("huber_log", "l1_log", "l1_inv")


def _hw1_to_nchw(x: Tensor) -> Tensor:
    """[H, W] or [H, W, 1] -> [1, 1, H, W] float."""
    if x.ndim == 3:
        x = x[..., 0]
    return x.float().reshape(1, 1, int(x.shape[0]), int(x.shape[1]))


def pool_to(x: Tensor, hw: Tuple[int, int]) -> Tensor:
    """Area-average ``x`` ([H, W] or [H, W, 1]) onto an ``hw`` grid -> [h, w].

    Integer factors use avg_pool2d (exact block means); other ratios use
    adaptive average pooling, which is the same area-weighted mean with
    non-uniform bins. Identity when the sizes already match.
    """
    h, w = int(hw[0]), int(hw[1])
    t = _hw1_to_nchw(x)
    H, W = int(t.shape[-2]), int(t.shape[-1])
    if (H, W) == (h, w):
        return t[0, 0]
    if H % h == 0 and W % w == 0 and H // h == W // w:
        k = H // h
        return F.avg_pool2d(t, kernel_size=k, stride=k)[0, 0]
    return F.adaptive_avg_pool2d(t, (h, w))[0, 0]


@torch.no_grad()
def depth_target_valid(
    target: Tensor,
    *,
    edge_px: int = 1,
    edge_rel_jump: float = 0.05,
) -> Tensor:
    """Boolean [h, w]: target valid and not within ``edge_px`` of a hole or a
    relative depth jump larger than ``edge_rel_jump`` (in log-depth, i.e.
    max/min over the window > exp(edge_rel_jump))."""
    d = target.float()
    if d.ndim == 3:
        d = d[..., 0]
    valid = torch.isfinite(d) & (d > 0)
    k = int(edge_px)
    if k <= 0:
        return valid
    win = 2 * k + 1
    v = valid.float()[None, None]
    # Erode the valid set: any invalid neighbour within the window kills it.
    eroded = -F.max_pool2d(-v, win, stride=1, padding=k)
    keep = eroded[0, 0] > 0.5
    if edge_rel_jump > 0:
        ld = torch.where(valid, d.clamp_min(1e-6).log(), torch.zeros_like(d))
        hi = torch.where(valid, ld, torch.full_like(ld, -1e9))[None, None]
        lo = torch.where(valid, ld, torch.full_like(ld, 1e9))[None, None]
        local_max = F.max_pool2d(hi, win, stride=1, padding=k)[0, 0]
        local_min = -F.max_pool2d(-lo, win, stride=1, padding=k)[0, 0]
        keep = keep & ((local_max - local_min) <= float(edge_rel_jump))
    return valid & keep


def depth_residual_loss(
    pred: Tensor, target: Tensor, loss_type: str = "huber_log", huber_delta: float = 0.02,
) -> Tensor:
    """Per-pixel loss between positive depths ``pred`` and ``target`` (same shape).

    huber_log: Huber (smooth-L1, quadratic below ``huber_delta``) on
        log(pred) - log(target). A relative error, so it is unit-free and
        weights near and far surfaces alike; quadratic in the ToF noise band,
        linear (robust) beyond it.
    l1_log: |log(pred) - log(target)|.
    l1_inv: |1/pred - 1/target| * target = |target/pred - 1| (relative error
        on inverse depth).
    """
    p = pred.clamp_min(1e-4)
    t = target.clamp_min(1e-4)
    if loss_type == "huber_log":
        r = p.log() - t.log()
        return F.smooth_l1_loss(r, torch.zeros_like(r), reduction="none",
                                beta=max(float(huber_delta), 1e-8))
    if loss_type == "l1_log":
        return (p.log() - t.log()).abs()
    if loss_type == "l1_inv":
        return (t / p - 1.0).abs()
    raise ValueError(f"depth_loss_type must be one of {DEPTH_LOSS_TYPES}, got {loss_type!r}")


def depth_supervision(
    depth_acc: Tensor,
    alpha: Tensor,
    target: Tensor,
    *,
    mask: Optional[Tensor] = None,
    loss_type: str = "huber_log",
    huber_delta: float = 0.02,
    mask_min: float = 0.5,
    alpha_min: float = 0.5,
    edge_px: int = 1,
    edge_rel_jump: float = 0.05,
    target_scale: float = 1.0,
    with_stats: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Depth loss for one rendered view.

    Args:
        depth_acc: [H, W(, 1)] accumulated depth sum_i w_i z_i (render units).
        alpha: [H, W(, 1)] accumulated opacity.
        target: [h, w] measured depth in metres; 0/non-finite = invalid.
        mask: optional foreground mask [H', W'(, 1)] in [0, 1] (any size; it
            is area-pooled onto the target grid).
        target_scale: render (world) units per target unit (metres).

    Returns ``(loss, stats)``. The loss is the per-pixel loss summed over the
    supervised pixels and divided by the number of target pixels (frame
    normalisation, like the trainer's alpha terms, so lambda means the same
    for tight and wide framings). ``stats`` (no grad, only when
    ``with_stats``) holds ``abs_err`` (target units) and ``ratio``
    (render/target) over the supervised pixels plus pixel counts.
    """
    tgt = target.float()
    if tgt.ndim == 3:
        tgt = tgt[..., 0]
    tgt = tgt.to(depth_acc.device)
    h, w = int(tgt.shape[0]), int(tgt.shape[1])

    d_low = pool_to(depth_acc, (h, w))
    a_low = pool_to(alpha, (h, w))

    weight = depth_target_valid(tgt, edge_px=edge_px, edge_rel_jump=edge_rel_jump)
    if mask is not None:
        m = mask.float().to(depth_acc.device)
        if m.numel() and float(m.max()) > 1.5:
            m = m / 255.0
        weight = weight & (pool_to(m, (h, w)) >= float(mask_min))
    n_target = int(weight.sum())
    covered = weight & (a_low.detach() >= float(alpha_min))

    tgt_world = tgt * float(target_scale)
    n_px = float(h * w)
    if bool(covered.any()):
        ed = d_low[covered] / a_low[covered].clamp_min(1e-6)
        per_px = depth_residual_loss(ed, tgt_world[covered], loss_type, huber_delta)
        loss = per_px.sum() / n_px
    else:
        ed = d_low.new_zeros((0,))
        loss = (d_low.sum() + a_low.sum()) * 0.0

    stats: Dict[str, Tensor] = {}
    if with_stats:
        with torch.no_grad():
            t_sel = tgt_world[covered]
            e = ed.detach()
            stats = {
                "abs_err": (e - t_sel).abs() / max(float(target_scale), 1e-12),
                "ratio": e / t_sel.clamp_min(1e-6),
                "n_target": torch.tensor(n_target),
                "n_covered": torch.tensor(int(covered.sum())),
            }
    return loss, stats
