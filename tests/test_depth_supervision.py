"""Depth supervision: the --depth-root loader and the depth loss."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from trigs.cli.train_from_colmap import TrigsColmapArgs, _LazyGTDepths
from trigs.core.depth import (
    DEPTH_LOSS_TYPES,
    depth_supervision,
    depth_target_valid,
    pool_to,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _depths(tmp_path: Path, *, hw=(8, 6), pattern="{frame:06d}.png", **kwargs):
    return _LazyGTDepths(
        depth_root=tmp_path,
        cam_folders_by_pos=["0001"],
        source_start_frame=1,
        target_hw_by_pos=[hw],
        depth_pattern=pattern,
        **kwargs,
    )


def _write_png(path: Path, arr_u16: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), arr_u16.astype(np.uint16))


def test_cli_defaults_leave_depth_off() -> None:
    fields = TrigsColmapArgs.__dataclass_fields__
    assert fields["depth_root"].default is None
    assert fields["depth_pattern"].default == "{frame:06d}.png"
    assert fields["depth_scale"].default == pytest.approx(0.001)
    from trigs.core.trainer import TrigsTrainingConfig
    assert TrigsTrainingConfig().lambda_depth == 0.0


def test_png_uint16_millimetres_to_metres_with_zero_invalid(tmp_path: Path) -> None:
    raw = np.full((4, 3), 2000, dtype=np.uint16)
    raw[0, 0] = 0
    raw[1, 2] = 1234
    _write_png(tmp_path / "0001" / "000001.png", raw)

    got = _depths(tmp_path)[0][0]
    assert got.dtype == torch.float32
    assert tuple(got.shape) == (4, 3)  # half-res map keeps its own resolution
    assert float(got[0, 0]) == 0.0
    assert float(got[1, 2]) == pytest.approx(1.234)
    assert float(got[3, 1]) == pytest.approx(2.0)


def test_png_respects_depth_scale(tmp_path: Path) -> None:
    _write_png(tmp_path / "0001" / "000001.png", np.full((4, 3), 500, dtype=np.uint16))
    got = _depths(tmp_path, depth_scale=0.002)[0][0]
    assert float(got.mean()) == pytest.approx(1.0)


def test_npy_metres_non_finite_and_negative_are_invalid(tmp_path: Path) -> None:
    d = np.full((4, 3), 1.5, dtype=np.float32)
    d[0, 0] = np.nan
    d[0, 1] = np.inf
    d[0, 2] = -1.0
    (tmp_path / "0001").mkdir()
    np.save(tmp_path / "0001" / "000002.npy", d)

    loader = _depths(tmp_path, pattern="{frame:06d}.npy", depth_scale=123.0)
    got = loader[1][0]  # frame_offset 1 -> source frame 2
    assert got[0].tolist() == [0.0, 0.0, 0.0]
    assert float(got[1:].min()) == pytest.approx(1.5)  # .npy ignores depth_scale


def test_png_pattern_falls_back_to_npy(tmp_path: Path) -> None:
    (tmp_path / "0001").mkdir()
    np.save(tmp_path / "0001" / "000001.npy", np.full((4, 3), 0.75, dtype=np.float32))
    got = _depths(tmp_path)[0][0]
    assert float(got.mean()) == pytest.approx(0.75)


def test_map_finer_than_render_is_nearest_subsampled(tmp_path: Path) -> None:
    raw = np.zeros((16, 12), dtype=np.uint16)
    raw[::2, ::2] = 1000  # a blend would produce 250-500 mm values
    _write_png(tmp_path / "0001" / "000001.png", raw)
    got = _depths(tmp_path, hw=(8, 6))[0][0]
    assert tuple(got.shape) == (8, 6)
    assert set(np.unique(got.numpy()).round(6).tolist()) <= {0.0, 1.0}


def test_aspect_ratio_mismatch_is_an_error(tmp_path: Path) -> None:
    _write_png(tmp_path / "0001" / "000001.png", np.ones((4, 4), dtype=np.uint16))
    with pytest.raises(RuntimeError, match="aspect ratio"):
        _depths(tmp_path, hw=(8, 6))[0][0]


def test_non_uint16_png_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "0001").mkdir()
    cv2.imwrite(str(tmp_path / "0001" / "000001.png"), np.ones((4, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="uint16"):
        _depths(tmp_path)[0][0]


def test_missing_depth_raises_or_returns_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"000001\.png"):
        _depths(tmp_path)[0][0]
    assert _depths(tmp_path, error_on_missing=False)[0][0] is None


def test_boundary_validation_reports_last_frame(tmp_path: Path) -> None:
    _write_png(tmp_path / "0001" / "000001.png", np.ones((4, 3), dtype=np.uint16))
    loader = _depths(tmp_path)
    loader.validate_boundaries(1)
    with pytest.raises(FileNotFoundError, match=r"000003\.png"):
        loader.validate_boundaries(3)


def test_distorted_cameras_are_refused(tmp_path: Path) -> None:
    fake_warp = (torch.zeros(4, 3, 2), (0, 0, 3, 4))
    with pytest.raises(ValueError, match="PINHOLE"):
        _depths(tmp_path, raw_warps_by_pos=[fake_warp])


def test_lru_cache_is_bounded(tmp_path: Path) -> None:
    for f in (1, 2, 3):
        _write_png(tmp_path / "0001" / f"{f:06d}.png", np.full((4, 3), f, dtype=np.uint16))
    loader = _depths(tmp_path, cache_size=2)
    for fo in (0, 1, 2):
        loader[fo][0]
    assert len(loader._cache) == 2


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _render_from_target(target: torch.Tensor, factor: int = 2, alpha: float = 1.0):
    """Accumulated depth/alpha a perfect render at ``factor``x the target's
    resolution would produce (holes rendered at 3 m)."""
    z = torch.where(target > 0, target, torch.full_like(target, 3.0))
    z_hi = z.repeat_interleave(factor, 0).repeat_interleave(factor, 1)
    a_hi = torch.full_like(z_hi, alpha)
    return (a_hi * z_hi)[..., None], a_hi[..., None]


def _smooth_target(h=12, w=10):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return 2.0 + 0.002 * yy.float() + 0.001 * xx.float()


@pytest.mark.parametrize("loss_type", DEPTH_LOSS_TYPES)
@pytest.mark.parametrize("alpha", [1.0, 0.8])
def test_loss_is_zero_when_rendered_depth_equals_target(loss_type: str, alpha: float) -> None:
    target = _smooth_target()
    target[3, 4] = 0.0  # a hole: excluded, not penalised
    d_acc, a = _render_from_target(target, alpha=alpha)
    loss, stats = depth_supervision(
        d_acc, a, target, loss_type=loss_type, with_stats=True,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(stats["abs_err"].max()) < 1e-5
    assert float(stats["ratio"].median()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("loss_type", DEPTH_LOSS_TYPES)
def test_loss_is_positive_and_differentiable_when_depth_is_off(loss_type: str) -> None:
    target = _smooth_target()
    d_acc, a = _render_from_target(target * 1.1)
    d_acc.requires_grad_(True)
    loss, _ = depth_supervision(d_acc, a, target, loss_type=loss_type)
    assert float(loss.detach()) > 0.0
    loss.backward()
    # Too far -> the gradient pulls the depth sum down.
    assert float(d_acc.grad.sum()) > 0.0


def test_mask_restricts_supervision() -> None:
    target = _smooth_target()
    d_acc, a = _render_from_target(target)
    # Corrupt the render in the left half only.
    d_acc = d_acc.clone()
    d_acc[:, :10] *= 1.5
    mask = torch.zeros(24, 20, 1)
    mask[:, 10:] = 1.0  # foreground = right half (render resolution)
    loss_masked, _ = depth_supervision(d_acc, a, target, mask=mask, edge_px=0)
    assert float(loss_masked) == pytest.approx(0.0, abs=1e-7)
    loss_inv, _ = depth_supervision(d_acc, a, target, mask=1.0 - mask, edge_px=0)
    assert float(loss_inv) > 0.0
    # uint8 masks (0/255) behave the same as float masks.
    loss_u8, _ = depth_supervision(
        d_acc, a, target, mask=(mask * 255).to(torch.uint8), edge_px=0,
    )
    assert float(loss_u8) == pytest.approx(0.0, abs=1e-7)


def test_low_alpha_pixels_are_not_compared() -> None:
    target = _smooth_target()
    d_acc, a = _render_from_target(target)
    d_acc, a = d_acc.clone(), a.clone()
    # A near-empty region rendered at a wildly wrong depth.
    a[:12] = 0.1
    d_acc[:12] = 0.1 * 9.0
    loss, stats = depth_supervision(
        d_acc, a, target, alpha_min=0.5, edge_px=0, with_stats=True,
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    assert int(stats["n_covered"]) == 6 * 10
    assert int(stats["n_target"]) == 12 * 10


def test_render_is_area_pooled_onto_the_target_grid() -> None:
    # One target pixel covered by a 2x2 render block: two pixels at 2 m with
    # alpha 1, two at 4 m with alpha 0.5 -> alpha-weighted mean 2.666 m.
    d_acc = torch.tensor([[2.0, 2.0], [2.0, 2.0]])[..., None]
    a = torch.tensor([[1.0, 1.0], [0.5, 0.5]])[..., None]
    d_acc[1] = 0.5 * 4.0
    target = torch.full((1, 1), 8.0 / 3.0)
    loss, stats = depth_supervision(d_acc, a, target, edge_px=0, alpha_min=0.5,
                                    with_stats=True)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    # Non-integer factors use adaptive (area) pooling.
    assert tuple(pool_to(torch.ones(7, 5), (3, 2)).shape) == (3, 2)


def test_depth_discontinuities_and_hole_borders_are_eroded() -> None:
    target = torch.full((9, 9), 2.0)
    target[:, 5:] = 3.0  # 50 % jump between columns 4 and 5
    target[0, 0] = 0.0   # hole in a corner
    valid = depth_target_valid(target, edge_px=1, edge_rel_jump=0.05)
    assert not valid[:, 4].any() and not valid[:, 5].any()
    assert valid[4, 2] and valid[4, 7]
    assert not valid[0, 0] and not valid[1, 1] and not valid[0, 1]
    # A smooth surface keeps everything except the image border (no
    # neighbours outside the frame is not a hole).
    smooth = depth_target_valid(_smooth_target(), edge_px=1)
    assert bool(smooth.all())
    assert bool(depth_target_valid(target, edge_px=0)[:, 4].all())


def test_target_scale_converts_metres_to_world_units() -> None:
    target = _smooth_target()
    d_acc, a = _render_from_target(target * 0.5)  # world units = 0.5 x metres
    loss, stats = depth_supervision(d_acc, a, target, target_scale=0.5, with_stats=True)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    # Errors are reported back in metres.
    d_off, _ = _render_from_target(target * 0.5 + 0.005)
    _, stats = depth_supervision(d_off, a, target, target_scale=0.5, with_stats=True)
    assert float(stats["abs_err"].median()) == pytest.approx(0.01, rel=1e-3)


def test_huber_log_is_quadratic_inside_delta_and_linear_outside() -> None:
    from trigs.core.depth import depth_residual_loss
    t = torch.tensor([2.0])
    small = float(depth_residual_loss(t * math.exp(0.01), t, "huber_log", 0.02))
    large = float(depth_residual_loss(t * math.exp(0.2), t, "huber_log", 0.02))
    assert small == pytest.approx(0.5 * 0.01 ** 2 / 0.02, rel=1e-4)
    assert large == pytest.approx(0.2 - 0.01, rel=1e-4)


# ---------------------------------------------------------------------------
# Trainer wiring (CPU, gsplat stubbed)
# ---------------------------------------------------------------------------

_HW = 16  # SSIM's 11-px window needs more than the 4x4 toy image of test_trainer


def _fake_render_depth(self, params, cameras, cam_idx, t, sh_degree, config,
                       step=None, with_depth=False):
    """Differentiable gsplat stand-in: real deform, toy image, toy depth."""
    n = int(params["means"].shape[0])
    means_t, _, gamma = self._deform(params, t, config)
    means2d = means_t[None, :, :2]
    means2d.retain_grad()
    opa = torch.sigmoid(params["opacities"]).reshape(-1, 1)
    color = (opa * gamma * (params["sh0"] + params["scales"].exp())).mean(dim=0)
    color = color + 0.01 * means2d.mean() + 0.1 * float(cam_idx)
    image = color.reshape(1, 1, 1, 3).expand(1, _HW, _HW, 3)
    alphas = torch.full((1, _HW, _HW, 1), 0.5)
    info = {
        "means2d": means2d,
        "radii": torch.ones(1, n, 2, dtype=torch.int32),
        "width": _HW, "height": _HW,
        "active_idx": torch.arange(n),
        "n_active": n, "n_total": n,
    }
    if with_depth:
        z = 2.0 + means_t[:, 2].mean()
        info["render_depth_acc"] = (alphas * z).reshape(1, _HW, _HW, 1)
    return image, alphas, gamma.detach(), info


def _run_loop(monkeypatch, *, lambda_depth: float, with_depths: bool, writer=None):
    from tests.test_trainer import _make_params
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

    seen: list[bool] = []

    def _recording(self, *args, **kwargs):
        seen.append(bool(kwargs.get("with_depth", False)))
        return _fake_render_depth(self, *args, **kwargs)

    monkeypatch.setattr(TrigsTrainer, "_render", _recording)
    torch.manual_seed(0)
    params = _make_params(N=24)
    for p in params.values():
        p.requires_grad_(True)
    n_frames, n_cams = 3, 2
    gt = {f: [torch.full((_HW, _HW, 3), 90, dtype=torch.uint8)] * n_cams for f in range(n_frames)}
    masks = {f: [torch.full((_HW, _HW), 255, dtype=torch.uint8)] * n_cams for f in range(n_frames)}
    depths = {f: [torch.full((_HW // 2, _HW // 2), 2.5)] * n_cams for f in range(n_frames)}
    config = TrigsTrainingConfig(
        frame_sampling_fast_visits=0.0, verbose=False, n_steps=4, fps=2.0,
        batch_cameras=2, relocate=False, lambda_depth=lambda_depth, ssim_lambda=0.0,
        depth_alpha_min=0.4, depth_edge_px=0, tb_log_every=1, tb_stats_every=0,
    )
    result = TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * n_cams,
        n_frames=n_frames, gt_images_per_frame=gt, gt_alphas_per_frame=masks,
        config=config, writer=writer,
        gt_depths_per_frame=depths if with_depths else None,
    )
    return seen, result


class _Writer:
    def __init__(self):
        self.scalars: dict[str, list[float]] = {}

    def add_scalar(self, k, v, step):
        self.scalars.setdefault(k, []).append(float(v))

    def __getattr__(self, name):
        return lambda *a, **k: None


def test_trainer_default_path_never_renders_depth(monkeypatch) -> None:
    seen, _ = _run_loop(monkeypatch, lambda_depth=0.0, with_depths=False)
    assert seen and not any(seen)


def test_trainer_logs_depth_at_lambda_zero_without_using_it(monkeypatch) -> None:
    writer = _Writer()
    seen, _ = _run_loop(monkeypatch, lambda_depth=0.0, with_depths=True, writer=writer)
    assert all(seen)
    assert len(writer.scalars["loss/depth"]) == 4
    assert all(v > 0 for v in writer.scalars["loss/depth"])
    assert "depth/current_median_abs_mm" in writer.scalars


def test_trainer_depth_loss_pulls_depth_towards_target(monkeypatch) -> None:
    writer = _Writer()
    _run_loop(monkeypatch, lambda_depth=10.0, with_depths=True, writer=writer)
    errs = writer.scalars["depth/current_median_abs_mm"]
    assert errs[-1] < errs[0]


def test_lambda_depth_without_targets_is_an_error(monkeypatch) -> None:
    with pytest.raises(ValueError, match="depth targets"):
        _run_loop(monkeypatch, lambda_depth=0.1, with_depths=False)
