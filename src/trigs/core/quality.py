"""Small, pure helpers behind the opt-in quality options of the trainer.

Kept separate from trainer.py so they can be unit-tested without a renderer:

* mask confidence ("unknown" pixels) carried by grey+alpha mask PNGs,
* the depth-loss weight schedule,
* the depth-distortion (ray compactness) term from rendered depth moments,
* the opacity-binarisation term,
* per-camera appearance (gain/bias) that leaves a composite plate untouched,
* the eroded-interior mask used by the eval metrics.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def split_mask_confidence(mask: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
    """Split a mask tensor into (mask, confidence), both float [H, W, 1] in [0, 1].

    Masks are 8-bit (0..255) or float (0..1), [H, W] or [H, W, C]. Channel 0 is
    the mask. A second channel, when present, is the per-pixel confidence of
    the mask: 1 = known, 0 = unknown (ignored by the mask losses), from the
    alpha channel of a grey+alpha ("LA") mask PNG. Plain masks return None for
    the confidence, i.e. every pixel is known.
    """
    m = mask.float()
    if m.numel() and float(m.max()) > 1.0:
        m = m / 255.0
    if m.ndim == 2:
        m = m[..., None]
    conf = m[..., 1:2] if m.shape[-1] >= 2 else None
    return m[..., :1].clamp(0.0, 1.0), (None if conf is None else conf.clamp(0.0, 1.0))


def resize_hw1(x: Tensor, hw: Tuple[int, int]) -> Tensor:
    """Bilinear resize of an [H, W, C] tensor to ``hw`` (identity if equal)."""
    if tuple(x.shape[:2]) == tuple(hw):
        return x
    return F.interpolate(
        x.permute(2, 0, 1).unsqueeze(0), size=tuple(int(v) for v in hw),
        mode="bilinear", align_corners=False,
    )[0].permute(1, 2, 0)


def depth_lambda_scale(
    global_step: int, decay_start_step: int, decay_steps: int, final_scale: float,
) -> float:
    """Multiplier on lambda_depth at ``global_step``.

    1 until ``decay_start_step``, then log-linear down to ``final_scale`` over
    ``decay_steps`` steps (a step change when decay_steps <= 0), constant after.
    ``final_scale`` = 1 (the default) keeps the weight constant.
    """
    final = float(final_scale)
    if final == 1.0 or int(global_step) < int(decay_start_step):
        return 1.0
    if final <= 0.0:
        raise ValueError(f"depth_loss_final_scale must be > 0, got {final}")
    n = int(decay_steps)
    if n <= 0:
        return final
    p = min(max((int(global_step) - int(decay_start_step)) / float(n), 0.0), 1.0)
    return float(math.exp(p * math.log(final)))


def depth_distortion_loss(
    alpha: Tensor, m1: Tensor, m2: Tensor, *, alpha_min: float = 0.5,
    weight: Optional[Tensor] = None,
) -> Tensor:
    """Spread of the blending weights along each ray (2DGS-style distortion).

    With per-pixel blending weights w_i and splat depths z_i, the renders give
    A = sum w_i, M1 = sum w_i z_i and M2 = sum w_i z_i^2, and

        sum_ij w_i w_j (z_i - z_j)^2 / 2 = A M2 - M1^2.

    It is zero when everything a pixel sees sits at one depth and grows with a
    translucent stack spread along the ray. Divided by the (detached) squared
    mean depth so it is unit-free (relative depth spread), summed over pixels
    with A >= ``alpha_min`` (optionally times ``weight`` [H, W, 1]) and
    normalised by the frame size like the alpha terms. Inputs [H, W] or
    [H, W, 1].
    """
    def hw1(x: Tensor) -> Tensor:
        return x if x.ndim == 3 else x[..., None]

    a, s1, s2 = hw1(alpha).float(), hw1(m1).float(), hw1(m2).float()
    spread = (a * s2 - s1 * s1).clamp_min(0.0)
    with torch.no_grad():
        zbar = s1 / a.clamp_min(1e-6)
        sel = (a >= float(alpha_min)) & (zbar > 1e-6)
        w = sel.float() / zbar.clamp_min(1e-6).pow(2)
        if weight is not None:
            w = w * hw1(weight).float()
    n_px = float(a.shape[0] * a.shape[1])
    return (spread * w).sum() / max(n_px, 1.0)


def opacity_binarization_loss(opacity_logits: Tensor) -> Tensor:
    """mean a (1 - a) over the given rows, a = sigmoid(logit).

    Zero at a = 0 or 1, largest at 0.5; its gradient pushes opacities below
    0.5 towards 0 and above 0.5 towards 1, so a translucent stack has to
    resolve into fewer, more opaque splats. Empty input -> 0.
    """
    if opacity_logits.numel() == 0:
        return opacity_logits.sum() * 0.0
    a = torch.sigmoid(opacity_logits.reshape(-1))
    return (a * (1.0 - a)).mean()


def apply_camera_appearance(
    pred: Tensor, gain: Tensor, bias: Tensor, alpha: Optional[Tensor] = None,
) -> Tensor:
    """Per-camera colour correction of a render.

    ``pred`` is the premultiplied render [H, W, 3]. Without ``alpha`` the
    correction is gain * pred + bias over the whole frame (unmasked training,
    where the scene includes the background). With ``alpha`` [H, W, 1] the
    bias is premultiplied too (gain * pred + bias * alpha), so a composite
    plate where alpha = 0 is left exactly as it is: the bias corrects the
    subject's colour instead of being pulled to zero by (or tinting) the plate.
    """
    if alpha is None:
        return pred * gain + bias
    a = alpha if alpha.ndim == pred.ndim else alpha[..., None]
    return pred * gain + bias * a.to(pred.dtype)


def erode_mask(mask01: Tensor, erode_px: int) -> Tensor:
    """Binary erosion of an [H, W, 1] {0, 1} mask by a (2k+1)^2 square."""
    k = int(erode_px)
    if k <= 0:
        return mask01
    inv = (1.0 - mask01.float()).permute(2, 0, 1).unsqueeze(0)
    grown = F.max_pool2d(inv, 2 * k + 1, stride=1, padding=k)
    return (1.0 - grown)[0].permute(1, 2, 0)
