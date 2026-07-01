#!/usr/bin/env python3
"""
fault_merge.py
==============
Merge nearby, geometrically aligned fault trace segments.

The important improvement in this version is configurable orientation.
Older versions normalized every segment West-to-East, which is fine for
mostly E-W faults but unreliable for mostly S-N faults. This version can
normalize along x, along y, skip normalization, or choose automatically
from the dataset.

Usage:
    python fault_merge.py
    python fault_merge.py --orientation y
    python fault_merge.py --orientation auto
    python fault_merge.py --test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:  # pragma: no cover - exercised only without scipy
    cKDTree = None


CONFIG = dict(
    # I/O
    data_dir=".",
    input_faults="faults.dat1",
    input_dim="dim.dat1",
    input_flen="flen.dat1",
    output_faults="faults_merged.dat",
    output_dim="dim_merged.dat",
    output_flen="flen_merged.dat",
    figure_out="fault_merge_result.png",

    # Merge criteria
    eps_km=5.0,
    max_angle_deg=20.0,
    min_pts=2,

    # Segment orientation before matching.
    #   "auto" = choose x or y from the dominant endpoint displacement
    #   "x"    = West-to-East, increasing x
    #   "y"    = South-to-North, increasing y
    #   "none" = keep input order
    orientation_axis="auto",

    # Behaviour
    max_passes=50,
    verbose=True,
    plot_steps=False,
    plot_overview=True,
)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def _endpoint_displacement(seg: np.ndarray) -> tuple[float, float]:
    """Return absolute endpoint displacement in x and y."""
    if len(seg) < 2:
        return 0.0, 0.0
    dx = float(abs(seg[-1, 0] - seg[0, 0]))
    dy = float(abs(seg[-1, 1] - seg[0, 1]))
    return dx, dy


def _resolve_orientation_axis(
    faults: list[np.ndarray],
    requested: str,
    *,
    verbose: bool = False,
) -> str:
    """
    Convert the user setting to one of: x, y, none.

    Auto mode chooses the axis with the larger total endpoint displacement.
    This handles mostly E-W and mostly S-N study regions without requiring
    users to edit code.
    """
    aliases = {
        "ew": "x",
        "we": "x",
        "west-east": "x",
        "east-west": "x",
        "x": "x",
        "sn": "y",
        "ns": "y",
        "south-north": "y",
        "north-south": "y",
        "y": "y",
        "none": "none",
        "input": "none",
    }

    value = str(requested).strip().lower()
    if value in aliases:
        axis = aliases[value]
    elif value == "auto":
        total_dx = 0.0
        total_dy = 0.0
        for fault in faults:
            dx, dy = _endpoint_displacement(fault)
            total_dx += dx
            total_dy += dy
        axis = "y" if total_dy > total_dx else "x"
        if verbose:
            label = "South-to-North" if axis == "y" else "West-to-East"
            print(
                "Auto orientation selected "
                f"{axis} ({label}); total |dx|={total_dx:.3f}, "
                f"total |dy|={total_dy:.3f}."
            )
    else:
        allowed = "auto, x, y, none, EW/WE, SN/NS"
        raise ValueError(f"Unknown orientation_axis={requested!r}. Use {allowed}.")

    if verbose and value != "auto":
        labels = {
            "x": "x / West-to-East",
            "y": "y / South-to-North",
            "none": "none / keep input point order",
        }
        print(f"Using orientation: {labels[axis]}.")
    return axis


def _orient_segment(seg: np.ndarray, axis: str) -> np.ndarray:
    """
    Return a copy of a segment with a stable start/end direction.

    For axis="x", segment[0] is the western tip.
    For axis="y", segment[0] is the southern tip.
    For axis="none", point order is left unchanged.
    """
    out = np.asarray(seg, dtype=float)
    if axis == "none" or len(out) < 2:
        return out.copy()

    if axis == "x":
        first = (out[0, 0], out[0, 1])
        last = (out[-1, 0], out[-1, 1])
    elif axis == "y":
        first = (out[0, 1], out[0, 0])
        last = (out[-1, 1], out[-1, 0])
    else:
        raise ValueError(f"Internal error: unknown axis {axis!r}")

    if first > last:
        return out[::-1].copy()
    return out.copy()


def _orient_faults(faults: list[np.ndarray], axis: str) -> list[np.ndarray]:
    """Normalize all fault traces using the selected orientation axis."""
    return [_orient_segment(seg, axis) for seg in faults]


def _angle_diff(a: float, b: float) -> float:
    """
    Smallest unsigned angular difference between two strikes, in degrees.

    Fault strike has 180 degree ambiguity, so the result is folded into
    the range 0..90.
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _seg_length(seg: np.ndarray) -> float:
    """Cumulative Euclidean length of a polyline."""
    d = np.diff(seg, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _local_strike(seg: np.ndarray, end: bool, n_pts: int = 3) -> float:
    """
    Estimate local strike at one end of a segment.

    end=False means the beginning of the segment.
    end=True means the end of the segment.
    """
    if end:
        pts = seg[-min(n_pts, len(seg)) :]
    else:
        pts = seg[: min(n_pts, len(seg))]

    if len(pts) < 2:
        return 0.0

    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return float(np.degrees(np.arctan2(dy, dx)))


class _RadiusIndex:
    """
    Radius-search wrapper.

    Uses scipy.spatial.cKDTree when SciPy is installed. Falls back to a
    vectorized NumPy scan for small datasets and test environments.
    """

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points) if cKDTree is not None else None

    def query_ball_point(self, point: np.ndarray, radius: float) -> list[int]:
        if self.tree is not None:
            return list(self.tree.query_ball_point(point, radius))

        delta = self.points - np.asarray(point, dtype=float)
        dist2 = np.einsum("ij,ij->i", delta, delta)
        return np.flatnonzero(dist2 <= radius * radius).tolist()


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------


def load_faults(cfg: dict) -> list[np.ndarray]:
    """Load faults as a list of (N_i, 2) arrays."""
    base = Path(cfg["data_dir"])
    pts = np.loadtxt(base / cfg["input_faults"])
    dims = np.loadtxt(base / cfg["input_dim"], dtype=int).ravel()

    faults = []
    idx = 0
    for dim in dims:
        faults.append(np.asarray(pts[idx : idx + dim], dtype=float).copy())
        idx += int(dim)

    if idx != len(pts):
        raise ValueError(
            "Input size mismatch: sum(dim.dat1) does not equal rows in faults.dat1."
        )

    if cfg["verbose"]:
        print(
            f"Loaded {len(faults):,} fault segments, "
            f"{sum(len(f) for f in faults):,} total points."
        )
    return faults


def save_faults(faults: list[np.ndarray], cfg: dict) -> None:
    """Write merged faults, dimensions, and lengths."""
    base = Path(cfg["data_dir"])
    dims = np.array([len(f) for f in faults], dtype=int)
    flens = np.array([_seg_length(f) for f in faults])
    pts = np.vstack(faults) if faults else np.empty((0, 2))

    np.savetxt(base / cfg["output_faults"], pts, fmt="%.6f")
    np.savetxt(base / cfg["output_dim"], dims, fmt="%d")
    np.savetxt(base / cfg["output_flen"], flens, fmt="%.6f")
    print(
        f"Saved: {cfg['output_faults']}, "
        f"{cfg['output_dim']}, {cfg['output_flen']}"
    )


# ---------------------------------------------------------------------
# Core merge logic
# ---------------------------------------------------------------------


def _build_endpoint_arrays(faults: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return beginning and ending endpoints as two (N, 2) arrays."""
    begins = np.vstack([fault[0] for fault in faults])
    ends = np.vstack([fault[-1] for fault in faults])
    return begins, ends


def _candidate_pairs(
    faults: list[np.ndarray],
    eps: float,
    max_angle: float,
    min_pts: int,
) -> list[tuple[float, int, int]]:
    """
    Find all merge-eligible pairs in one pass.

    Returns (distance, upstream_index, downstream_index), sorted by distance.
    The endpoint search uses two KD-trees:
      - one over all beginnings
      - one over all endings
    """
    n = len(faults)
    if n == 0:
        return []

    begins, ends = _build_endpoint_arrays(faults)
    begin_tree = _RadiusIndex(begins)
    end_tree = _RadiusIndex(ends)

    pairs = []
    seen: set[tuple[int, int]] = set()

    for i in range(n):
        fi = faults[i]
        if len(fi) < min_pts:
            continue

        # end of i -> beginning of j
        for j in begin_tree.query_ball_point(fi[-1], eps):
            if j == i or (i, j) in seen or len(faults[j]) < min_pts:
                continue

            fj = faults[j]
            dist = float(np.hypot(*(fi[-1] - fj[0])))
            strike_i = _local_strike(fi, end=True)
            strike_j = _local_strike(fj, end=False)

            if _angle_diff(strike_i, strike_j) < max_angle:
                pairs.append((dist, i, j))
                seen.add((i, j))
                seen.add((j, i))

        # end of j -> beginning of i
        for j in end_tree.query_ball_point(fi[0], eps):
            if j == i or (j, i) in seen or len(faults[j]) < min_pts:
                continue

            fj = faults[j]
            dist = float(np.hypot(*(fj[-1] - fi[0])))
            strike_j = _local_strike(fj, end=True)
            strike_i = _local_strike(fi, end=False)

            if _angle_diff(strike_j, strike_i) < max_angle:
                pairs.append((dist, j, i))
                seen.add((j, i))
                seen.add((i, j))

    pairs.sort(key=lambda item: item[0])
    return pairs


def _do_merge(
    upstream: np.ndarray,
    downstream: np.ndarray,
    dist: float,
    orientation_axis: str,
) -> np.ndarray:
    """Concatenate two ordered segments and normalize the result."""
    if dist == 0.0:
        merged = np.vstack([upstream, downstream[1:]])
    else:
        merged = np.vstack([upstream, downstream])
    return _orient_segment(merged, orientation_axis)


def merge_all(faults: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """
    Iteratively merge fault segments.

    Each pass:
      1. Normalize segment start/end order using x, y, none, or auto.
      2. Build KD-trees over segment endpoints.
      3. Find candidate tip-to-tip matches within eps_km.
      4. Filter by local strike alignment.
      5. Greedily merge non-conflicting pairs, closest first.
    """
    eps = float(cfg["eps_km"])
    max_angle = float(cfg["max_angle_deg"])
    min_pts = int(cfg["min_pts"])
    verbose = bool(cfg["verbose"])

    orientation_axis = _resolve_orientation_axis(
        faults,
        cfg.get("orientation_axis", "auto"),
        verbose=verbose,
    )
    faults = _orient_faults(faults, orientation_axis)

    n_orig = len(faults)
    total_merges = 0
    pass_no = 0

    for pass_no in range(1, int(cfg["max_passes"]) + 1):
        pairs = _candidate_pairs(faults, eps, max_angle, min_pts)
        if not pairs:
            break

        used = [False] * len(faults)
        merges_this = []

        for dist, i, j in pairs:
            if used[i] or used[j]:
                continue
            used[i] = True
            used[j] = True
            merges_this.append((dist, i, j))

        if not merges_this:
            break

        merged_indices = {idx for _, i, j in merges_this for idx in (i, j)}
        new_faults = [
            fault for idx, fault in enumerate(faults) if idx not in merged_indices
        ]

        for dist, i, j in merges_this:
            new_faults.append(_do_merge(faults[i], faults[j], dist, orientation_axis))

        total_merges += len(merges_this)
        faults = new_faults

        if verbose:
            print(
                f"  Pass {pass_no:2d}: {len(merges_this):4d} merges, "
                f"{len(faults):,} segments remaining"
            )

        if cfg["plot_steps"]:
            _plot_step(faults, pass_no)

    print(
        f"\nFinished. {n_orig:,} -> {len(faults):,} segments "
        f"({total_merges} total merges, {pass_no} passes)."
    )
    return faults


# ---------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------


def _plot_faults(ax, faults: list[np.ndarray], title: str, max_faults: int = 5000):
    """Draw fault traces; subsample if there are too many."""
    import matplotlib.cm as cm

    n = len(faults)
    step = max(1, n // max_faults)
    cmap = cm.get_cmap("tab20", 20)

    for k, fault in enumerate(faults[::step]):
        color = cmap(k % 20)
        ax.plot(fault[:, 0], fault[:, 1], color=color, lw=0.8, alpha=0.7)
        ax.plot(*fault[0], "o", color=color, ms=2, zorder=3)
        ax.plot(*fault[-1], "s", color=color, ms=2, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.2, alpha=0.5)


def plot_overview(
    faults_before: list[np.ndarray],
    faults_after: list[np.ndarray],
    cfg: dict,
) -> None:
    """Create before/after map views and a length histogram."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Fault Trace Merging Results", fontsize=14, fontweight="bold")

    _plot_faults(axes[0], faults_before, f"Before merge\n({len(faults_before):,})")
    _plot_faults(axes[1], faults_after, f"After merge\n({len(faults_after):,})")

    len_before = [_seg_length(f) for f in faults_before]
    len_after = [_seg_length(f) for f in faults_after]
    max_len = max(max(len_before, default=0.0), max(len_after, default=0.0))
    bins = np.linspace(0.0, max_len if max_len > 0 else 1.0, 60)

    axes[2].hist(len_before, bins=bins, alpha=0.6, label="Before", color="steelblue")
    axes[2].hist(len_after, bins=bins, alpha=0.6, label="After", color="tomato")
    axes[2].set_xlabel("Fault length")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Length distribution")
    axes[2].legend()
    axes[2].grid(True, lw=0.3, alpha=0.5)

    plt.tight_layout()
    out = Path(cfg["data_dir"]) / cfg["figure_out"]
    plt.savefig(out, dpi=150)
    print(f"Figure saved: {out}")
    plt.show()


def _plot_step(faults: list[np.ndarray], pass_no: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))
    _plot_faults(ax, faults, f"After pass {pass_no}: {len(faults)} segments")
    plt.tight_layout()
    plt.pause(0.5)
    plt.close(fig)


# ---------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------


def make_test_data(out_dir: str, orientation: str) -> list[np.ndarray]:
    """Create a small E-W or S-N synthetic dataset."""
    if orientation == "x":
        faults = [
            np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
            np.array([[7.0, 0.0], [8.0, 0.0], [9.0, 0.0]]),
            np.array([[0.0, 20.0], [3.0, 20.0], [6.0, 20.0]]),
            np.array([[0.0, 40.0], [3.0, 40.0]]),
            np.array([[15.0, 40.0], [18.0, 40.0]]),
            np.array([[0.0, 60.0], [4.0, 60.0]]),
            np.array([[6.0, 60.0], [8.0, 62.5], [10.0, 65.0]]),
            np.array([[0.0, 80.0], [2.0, 80.0], [4.0, 80.0], [8.0, 80.0]]),
            np.array([[5.0, 100.0], [3.0, 100.0], [1.0, 100.0]]),
            np.array([[6.0, 100.0], [8.0, 100.0], [10.0, 100.0]]),
        ]
    elif orientation == "y":
        faults = [
            np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]]),
            np.array([[0.0, 7.0], [0.0, 8.0], [0.0, 9.0]]),
            np.array([[20.0, 0.0], [20.0, 3.0], [20.0, 6.0]]),
            np.array([[40.0, 0.0], [40.0, 3.0]]),
            np.array([[40.0, 15.0], [40.0, 18.0]]),
            np.array([[60.0, 0.0], [60.0, 4.0]]),
            np.array([[60.0, 6.0], [62.5, 8.0], [65.0, 10.0]]),
            np.array([[80.0, 0.0], [80.0, 2.0], [80.0, 4.0], [80.0, 8.0]]),
            np.array([[100.0, 5.0], [100.0, 3.0], [100.0, 1.0]]),
            np.array([[100.0, 6.0], [100.0, 8.0], [100.0, 10.0]]),
        ]
    else:
        raise ValueError("orientation must be x or y")

    dims = np.array([len(f) for f in faults], dtype=int)
    flens = np.array([_seg_length(f) for f in faults])
    pts = np.vstack(faults)

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    np.savetxt(base / "faults.dat1", pts, fmt="%.6f")
    np.savetxt(base / "dim.dat1", dims, fmt="%d")
    np.savetxt(base / "flen.dat1", flens, fmt="%.6f")

    return faults


def run_test(cfg: dict) -> None:
    """Run synthetic E-W and S-N datasets."""
    base = Path(cfg["data_dir"]) / "test_output"
    cases = [
        ("ew_auto", "x", "auto"),
        ("sn_auto", "y", "auto"),
        ("sn_forced_y", "y", "y"),
    ]

    for case_name, data_orientation, requested_orientation in cases:
        print("=" * 60)
        print(f"Running synthetic test: {case_name}")
        test_dir = base / case_name
        make_test_data(str(test_dir), data_orientation)

        test_cfg = {
            **cfg,
            "data_dir": str(test_dir),
            "orientation_axis": requested_orientation,
            "figure_out": f"{case_name}.png",
            "plot_overview": False,
        }

        faults_orig = load_faults(test_cfg)
        faults_merged = merge_all(faults_orig, test_cfg)
        save_faults(faults_merged, test_cfg)
        print(f"Result: {len(faults_orig)} -> {len(faults_merged)} segments")
        print("Expected: 10 -> 8 segments\n")

        if len(faults_merged) != 8:
            raise RuntimeError(f"Synthetic test failed: {case_name}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge fault traces.")
    parser.add_argument("--test", action="store_true", help="Run synthetic tests.")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualisation.")
    parser.add_argument(
        "--orientation",
        choices=["auto", "x", "y", "none", "ew", "we", "sn", "ns"],
        help="Segment orientation rule. Overrides CONFIG['orientation_axis'].",
    )
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.no_plot:
        cfg["plot_overview"] = False
    if args.orientation:
        cfg["orientation_axis"] = args.orientation

    if args.test:
        run_test(cfg)
        return

    faults_orig = load_faults(cfg)
    faults_merged = merge_all(faults_orig, cfg)
    save_faults(faults_merged, cfg)

    if cfg["plot_overview"]:
        plot_overview(faults_orig, faults_merged, cfg)


if __name__ == "__main__":
    main()
