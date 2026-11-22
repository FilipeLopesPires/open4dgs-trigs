"""Opt-in quality options: mask confidence, eval metrics, camera appearance,
depth schedule/offsets, depth distortion, opacity binarisation, SH degree."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch
from PIL import Image

from trigs.core.quality import (
    apply_camera_appearance,
    depth_distortion_loss,
    depth_lambda_scale,
    erode_mask,
    opacity_binarization_loss,
    split_mask_confidence,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_split_mask_confidence_plain_and_la() -> None:
    plain = torch.full((4, 5, 1), 255, dtype=torch.uint8)
    m, conf = split_mask_confidence(plain)
    assert conf is None and m.shape == (4, 5, 1) and float(m.min()) == 1.0
    m2, conf2 = split_mask_confidence(torch.full((4, 5), 128, dtype=torch.uint8))
    assert conf2 is None and m2.shape == (4, 5, 1)
    la = torch.zeros((4, 5, 2), dtype=torch.uint8)
    la[..., 0] = 200
    la[:, :2, 1] = 255
    m3, conf3 = split_mask_confidence(la)
    assert m3.shape == (4, 5, 1) and torch.allclose(m3, torch.full_like(m3, 200 / 255))
    assert conf3 is not None and float(conf3[:, :2].min()) == 1.0 and float(conf3[:, 2:].max()) == 0.0


def test_depth_lambda_scale_default_is_constant_and_decay_is_log_linear() -> None:
    assert depth_lambda_scale(10_000, 0, 0, 1.0) == 1.0
    assert depth_lambda_scale(10, 100, 50, 0.1) == 1.0
    assert depth_lambda_scale(125, 100, 50, 0.01) == pytest.approx(0.1)
    assert depth_lambda_scale(150, 100, 50, 0.01) == pytest.approx(0.01)
    assert depth_lambda_scale(10_000, 100, 50, 0.15) == pytest.approx(0.15)
    assert depth_lambda_scale(100, 100, 0, 0.2) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        depth_lambda_scale(200, 100, 50, 0.0)


def test_depth_distortion_zero_for_one_surface_and_matches_pairwise_sum() -> None:
    w = torch.tensor([0.6, 0.3])
    z = torch.tensor([2.0, 2.0])
    a, m1, m2 = w.sum(), (w * z).sum(), (w * z * z).sum()
    one = depth_distortion_loss(a.reshape(1, 1), m1.reshape(1, 1), m2.reshape(1, 1))
    assert float(one) == pytest.approx(0.0, abs=1e-7)
    z = torch.tensor([2.0, 2.2])
    a, m1, m2 = w.sum(), (w * z).sum(), (w * z * z).sum()
    got = float(depth_distortion_loss(a.reshape(1, 1), m1.reshape(1, 1), m2.reshape(1, 1)))
    pair = 0.5 * sum(float(w[i] * w[j] * (z[i] - z[j]) ** 2) for i in range(2) for j in range(2))
    zbar = float(m1 / a)
    assert got == pytest.approx(pair / zbar ** 2, rel=1e-4)
    # Below alpha_min nothing counts.
    low = depth_distortion_loss(torch.tensor([[0.2]]), torch.tensor([[0.4]]), torch.tensor([[0.9]]))
    assert float(low) == 0.0


def test_depth_distortion_gradient_pulls_depths_together() -> None:
    w = torch.tensor([0.5, 0.5])
    z = torch.tensor([2.0, 2.4], requires_grad=True)
    a = w.sum()
    loss = depth_distortion_loss(a.reshape(1, 1), (w * z).sum().reshape(1, 1), (w * z * z).sum().reshape(1, 1))
    loss.backward()
    assert float(z.grad[0]) < 0 < float(z.grad[1])   # near one moves back, far one forward


def test_opacity_binarization_pushes_to_extremes() -> None:
    logits = torch.tensor([-3.0, -0.5, 0.5, 3.0], requires_grad=True)
    opacity_binarization_loss(logits).backward()
    g = logits.grad
    assert float(g[0]) > 0 and float(g[1]) > 0      # descent lowers the low ones
    assert float(g[2]) < 0 and float(g[3]) < 0      # and raises the high ones
    assert float(opacity_binarization_loss(torch.zeros(0))) == 0.0


def test_camera_appearance_premultiplied_bias_leaves_plate_alone() -> None:
    pred = torch.full((2, 2, 3), 0.4)
    alpha = torch.tensor([[[1.0], [0.0]], [[0.5], [0.0]]])
    gain, bias = torch.tensor([1.1, 1.0, 0.9]), torch.tensor([0.05, 0.0, -0.05])
    out = apply_camera_appearance(pred * alpha, gain, bias, alpha=alpha)
    assert torch.allclose(out[0, 1], torch.zeros(3))            # plate pixel untouched
    assert torch.allclose(out[0, 0], 0.4 * gain + bias)
    full = apply_camera_appearance(pred, gain, bias)
    assert torch.allclose(full[0, 1], 0.4 * gain + bias)


def test_erode_mask() -> None:
    m = torch.zeros((9, 9, 1))
    m[2:7, 2:7] = 1.0
    e = erode_mask(m, 1)
    assert float(e.sum()) == 9.0 and float(e[4, 4]) == 1.0 and float(e[2, 2]) == 0.0


# ---------------------------------------------------------------------------
# COLMAP CLI loaders
# ---------------------------------------------------------------------------

def test_load_mask_with_confidence_reads_la_and_plain(tmp_path) -> None:
    from trigs.cli.train_from_colmap import load_mask_with_confidence

    grey = np.full((6, 8), 200, np.uint8)
    alpha = np.zeros((6, 8), np.uint8)
    alpha[:, :4] = 255
    Image.fromarray(np.stack([grey, alpha], -1), mode="LA").save(tmp_path / "la.png")
    Image.fromarray(grey, mode="L").save(tmp_path / "l.png")
    la = load_mask_with_confidence(tmp_path / "la.png", (6, 8))
    assert la.shape == (6, 8, 2) and int(la[0, 0, 0]) == 200
    assert int(la[0, 0, 1]) == 255 and int(la[0, 7, 1]) == 0
    plain = load_mask_with_confidence(tmp_path / "l.png", (12, 16))
    assert plain.shape == (12, 16, 1) and int(plain[5, 5, 0]) == 200


def test_depth_offsets_applied_to_valid_depth_only(tmp_path) -> None:
    from trigs.cli.train_from_colmap import _LazyGTDepths, load_depth_offsets

    (tmp_path / "offs.json").write_text(json.dumps({"offsets_m": {"0002": -0.0125, "9999": 1.0}}))
    offs = load_depth_offsets(tmp_path / "offs.json", ["0001", "0002"])
    assert offs == [0.0, -0.0125]
    for cam in ("0001", "0002"):
        (tmp_path / "d" / cam).mkdir(parents=True)
        d = np.full((4, 6), 2000, np.uint16)
        d[0, 0] = 0
        Image.fromarray(d).save(tmp_path / "d" / cam / "000001.png")
    loader = _LazyGTDepths(tmp_path / "d", ["0001", "0002"], source_start_frame=1,
                           target_hw_by_pos=[(8, 12), (8, 12)], offsets_m_by_pos=offs)
    d1, d2 = loader[0][0], loader[0][1]
    assert float(d1[1, 1]) == pytest.approx(2.0)
    assert float(d2[1, 1]) == pytest.approx(1.9875, abs=1e-6)
    assert float(d2[0, 0]) == 0.0


def test_sh_degree_arg_reaches_npz_builder() -> None:
    import inspect
    from trigs.cli import train_from_colmap

    assert train_from_colmap.TrigsColmapArgs.__dataclass_fields__["sh_degree"].default == 3
    src = inspect.getsource(train_from_colmap.main)
    assert src.count("max_sh_degree=int(args.sh_degree)") == 2


# ---------------------------------------------------------------------------
# Trainer: photometric losses with unknown pixels, eval metrics
# ---------------------------------------------------------------------------

def _cfg(**kw):
    from trigs.core.trainer import TrigsTrainingConfig
    base = dict(verbose=False, masked_rgb_loss=True, masked_rgb_composite=True, ssim_lambda=0.0)
    base.update(kw)
    return TrigsTrainingConfig(**base)


def test_unknown_pixels_do_not_enter_mask_losses() -> None:
    from trigs.core.trainer import TrigsTrainer

    H, W = 8, 8
    alpha = torch.full((1, H, W, 1), 0.7)
    colors = torch.full((1, H, W, 3), 0.3)
    gt = torch.full((H, W, 3), 0.5)
    mask = torch.zeros((H, W, 2), dtype=torch.uint8)       # all background
    mask[..., 1] = 255
    mask[:, : W // 2, 1] = 0                                 # left half unknown
    svc = TrigsTrainer(device="cpu")
    rgb_u, out_u, bce_u = svc._photometric_losses({}, colors, alpha, gt, {0: [mask]}, 0, 0, _cfg())
    plain = mask[..., :1].clone()
    rgb_p, out_p, bce_p = svc._photometric_losses({}, colors, alpha, gt, {0: [plain]}, 0, 0, _cfg())
    assert float(out_u) == pytest.approx(0.5 * float(out_p))
    assert float(bce_u) == pytest.approx(0.5 * float(bce_p), rel=1e-5)
    assert float(rgb_u) == pytest.approx(0.5 * float(rgb_p), rel=1e-5)
    cfg_keep = _cfg(mask_unknown_ignore_rgb=False)
    rgb_k, _, _ = svc._photometric_losses({}, colors, alpha, gt, {0: [mask]}, 0, 0, cfg_keep)
    assert float(rgb_k) == pytest.approx(float(rgb_p), rel=1e-5)


def test_composite_appearance_bias_does_not_touch_plate() -> None:
    import torch.nn as nn
    from trigs.core.trainer import TrigsTrainer

    H, W = 8, 8
    alpha = torch.zeros((1, H, W, 1))                       # empty render
    colors = torch.zeros((1, H, W, 3))
    gt = torch.full((H, W, 3), 0.5)
    mask = torch.zeros((H, W, 1), dtype=torch.uint8)        # all background -> GT = white plate
    params = {"cam_gain": nn.Parameter(torch.ones(1, 3)), "cam_bias": nn.Parameter(torch.full((1, 3), 0.1))}
    rgb, _, _ = TrigsTrainer(device="cpu")._photometric_losses(params, colors, alpha, gt, {0: [mask]}, 0, 0, _cfg())
    assert float(rgb) == pytest.approx(0.0, abs=1e-7)


def _eval_with_fake_render(monkeypatch, pred_premult, alpha, gt, mask, **cfg_kw):
    from trigs.core.trainer import TrigsTrainer

    def fake_render(self, params, cameras, cam_idx, t, sh_degree, config, step=None, **kw):
        return pred_premult[None], alpha[None], None, {}

    monkeypatch.setattr(TrigsTrainer, "_render", fake_render)
    return TrigsTrainer(device="cpu")._evaluate_psnr_subset(
        params={}, cameras=[None], eval_pairs=[(0, 0)], sh_degree=0, config=_cfg(**cfg_kw),
        gt_images_per_frame={0: [gt]}, gt_alphas_per_frame={0: [mask]},
        gt_times_per_frame=None, step=0,
    )


def test_masked_eval_uses_soft_mask_so_a_perfect_fit_scores_perfectly(monkeypatch) -> None:
    H, W = 16, 16
    gt = torch.rand((H, W, 3))
    m = torch.zeros((H, W, 1))
    m[4:12, 4:12] = 1.0
    m[4:12, 3] = 0.6       # soft edge columns
    m[4:12, 12] = 0.3
    # A model that reproduces the training target: premultiplied colour gt*m, opacity m.
    out = _eval_with_fake_render(monkeypatch, gt * m, m, (gt * 255).round().to(torch.uint8),
                                 (m * 255).round().to(torch.uint8), psnr_interior_erode_px=2)
    assert out["psnr/eval_masked"] > 45.0
    assert out["psnr/eval_full"] > 45.0
    assert out["psnr/eval_interior"] > 45.0


def test_interior_metric_ignores_the_matting_edge(monkeypatch) -> None:
    H, W = 24, 24
    gt = torch.full((H, W, 3), 0.5)
    m = torch.zeros((H, W, 1))
    m[4:20, 4:20] = 1.0
    pred = gt * m
    pred[4:20, 4:6] = 0.0                    # wrong colour on a 2-px edge band only
    out = _eval_with_fake_render(monkeypatch, pred, m, (gt * 255).round().to(torch.uint8),
                                 (m * 255).round().to(torch.uint8), psnr_interior_erode_px=3)
    assert out["psnr/eval_interior"] > 45.0
    assert out["psnr/eval_masked"] < 30.0


def test_eval_applies_learned_camera_gain(monkeypatch) -> None:
    import torch.nn as nn
    from trigs.core.trainer import TrigsTrainer

    H, W = 16, 16
    gt = torch.full((H, W, 3), 0.6)
    m = torch.ones((H, W, 1))
    pred = torch.full((H, W, 3), 0.5)        # renders 0.5, camera gain 1.2 -> 0.6

    def fake_render(self, params, cameras, cam_idx, t, sh_degree, config, step=None, **kw):
        return pred[None], m[None], None, {}

    monkeypatch.setattr(TrigsTrainer, "_render", fake_render)
    params = {"cam_gain": nn.Parameter(torch.full((1, 3), 1.2)), "cam_bias": nn.Parameter(torch.zeros(1, 3))}
    out = TrigsTrainer(device="cpu")._evaluate_psnr_subset(
        params=params, cameras=[None], eval_pairs=[(0, 0)], sh_degree=0, config=_cfg(),
        gt_images_per_frame={0: [(gt * 255).round().to(torch.uint8)]},
        gt_alphas_per_frame={0: [torch.full((H, W, 1), 255, dtype=torch.uint8)]},
        gt_times_per_frame=None, step=0,
    )
    assert out["psnr/eval_masked"] > 40.0


# ---------------------------------------------------------------------------
# Trainer loop wiring (CPU, gsplat stubbed)
# ---------------------------------------------------------------------------

_HW = 16


def _fake_render(self, params, cameras, cam_idx, t, sh_degree, config,
                 step=None, with_depth=False, with_moments=False):
    n = int(params["means"].shape[0])
    means_t, _, gamma = self._deform(params, t, config)
    means2d = means_t[None, :, :2]
    means2d.retain_grad()
    opa = torch.sigmoid(params["opacities"]).reshape(-1, 1)
    color = (opa * gamma * (params["sh0"] + params["scales"].exp())).mean(dim=0)
    color = color + 0.01 * means2d.mean()
    image = color.reshape(1, 1, 1, 3).expand(1, _HW, _HW, 3)
    alphas = torch.full((1, _HW, _HW, 1), 0.8)
    info = {
        "means2d": means2d, "radii": torch.ones(1, n, 2, dtype=torch.int32),
        "width": _HW, "height": _HW, "active_idx": torch.arange(n), "n_active": n, "n_total": n,
    }
    if with_depth:
        z = 2.0 + means_t[:, 2].mean()
        info["render_depth_acc"] = (alphas * z).reshape(1, _HW, _HW, 1)
    if with_moments:
        z = 2.0 + means_t[:, 2]
        w = opa.reshape(-1) / opa.sum() * 0.8
        m1 = (w * z).sum()
        m2 = (w * z * z).sum()
        info["render_depth_moments"] = torch.stack([m1, m2]).reshape(1, 1, 1, 2).expand(1, _HW, _HW, 2)
    return image, alphas, gamma.detach(), info


class _Writer:
    def __init__(self):
        self.scalars: dict[str, list[float]] = {}

    def add_scalar(self, k, v, step):
        self.scalars.setdefault(k, []).append(float(v))

    def __getattr__(self, name):
        return lambda *a, **k: None


def _run(monkeypatch, n_steps=6, la_masks=False, **cfg_kw):
    from tests.test_trainer import _make_params
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

    seen: list[bool] = []

    def recording(self, *a, **k):
        seen.append(bool(k.get("with_moments", False)))
        return _fake_render(self, *a, **k)

    monkeypatch.setattr(TrigsTrainer, "_render", recording)
    torch.manual_seed(0)
    params = _make_params(N=24)
    for p in params.values():
        p.requires_grad_(True)
    n_frames, n_cams = 3, 2
    gt = {f: [torch.full((_HW, _HW, 3), 90, dtype=torch.uint8)] * n_cams for f in range(n_frames)}
    mask = torch.full((_HW, _HW, 2 if la_masks else 1), 255, dtype=torch.uint8)
    if la_masks:
        mask[:4, :, 1] = 0
    masks = {f: [mask] * n_cams for f in range(n_frames)}
    depths = {f: [torch.full((_HW // 2, _HW // 2), 2.5)] * n_cams for f in range(n_frames)}
    base = dict(
        frame_sampling_fast_visits=0.0, verbose=False, n_steps=n_steps, fps=2.0, batch_cameras=2,
        relocate=False, ssim_lambda=0.0, depth_alpha_min=0.4, depth_edge_px=0, tb_log_every=1,
        tb_stats_every=0, masked_rgb_loss=True, masked_rgb_composite=True, lambda_depth=1.0,
    )
    base.update(cfg_kw)
    writer = _Writer()
    result = TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * n_cams, n_frames=n_frames,
        gt_images_per_frame=gt, gt_alphas_per_frame=masks, config=TrigsTrainingConfig(**base),
        writer=writer, gt_depths_per_frame=depths,
    )
    return seen, writer, result


def test_defaults_render_no_moments_and_log_no_new_terms(monkeypatch) -> None:
    seen, writer, _ = _run(monkeypatch)
    assert seen and not any(seen)
    assert "loss/depth_distortion" not in writer.scalars
    assert "loss/opacity_binary" not in writer.scalars
    assert set(writer.scalars["loss/lambda_depth"]) == {1.0}


def test_depth_schedule_decays_lambda(monkeypatch) -> None:
    _, writer, _ = _run(monkeypatch, n_steps=6, depth_loss_decay_start_step=2,
                        depth_loss_decay_steps=2, depth_loss_final_scale=0.1)
    lam = writer.scalars["loss/lambda_depth"]
    assert lam == pytest.approx([1.0, 1.0, 1.0, math.sqrt(0.1), 0.1, 0.1])


def test_distortion_and_binarization_start_late(monkeypatch) -> None:
    seen, writer, _ = _run(monkeypatch, n_steps=6, lambda_depth_distortion=10.0,
                           depth_distortion_start_step=3, lambda_opacity_binary=0.1,
                           opacity_binary_start_step=4)
    assert seen[:6] == [False] * 6 and all(seen[6:])
    assert len(writer.scalars["loss/depth_distortion"]) == 3
    assert len(writer.scalars["loss/opacity_binary"]) == 2
    assert all(v > 0 for v in writer.scalars["loss/opacity_binary"])


def test_loop_runs_with_la_masks_and_camera_appearance(monkeypatch) -> None:
    _, writer, result = _run(monkeypatch, la_masks=True, learn_camera_appearance=True)
    assert "camera/gain_0" in writer.scalars
    assert "cam_gain" in result.params
