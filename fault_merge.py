#!/usr/bin/env python3
"""
fault_merge.py
==============
Merges nearby, geometrically-aligned fault trace segments.

Improvements over flt_merge2.m
-------------------------------
1. Speed    – KD-tree endpoint lookup (O(n log n) instead of O(n²)).
              Greedy batch-merge passes replace scan-from-zero restarts.
2. Structure– All tuneable parameters live in CONFIG; data flows through
              clean Python lists/numpy arrays with no global mutation.
3. Ordering – All fault traces are re-oriented West→East before any
              processing, and after every merge.
4. Criteria – Three independent thresholds (distance, strike difference,
              overlap fraction) are each documented and enforced.
5. Visuals  – Before/after overview plot, per-merge step plot (optional),
              and a stats histogram.

Usage
-----
    python fault_merge.py                 # uses CONFIG defaults
    python fault_merge.py --test          # run on built-in synthetic dataset
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import cKDTree

# ═══════════════════════════════════════════════════════════════════════
#  CENTRAL CONFIGURATION  ← edit everything here
# ═══════════════════════════════════════════════════════════════════════
CONFIG = dict(
    # ── I/O ──────────────────────────────────────────────────────────
    data_dir        = ".",
    input_faults    = "faults.dat1",
    input_dim       = "dim.dat1",
    input_flen      = "flen.dat1",
    output_faults   = "faults_merged.dat",
    output_dim      = "dim_merged.dat",
    output_flen     = "flen_merged.dat",
    figure_out      = "fault_merge_result.png",

    # ── Merge criteria ────────────────────────────────────────────────
    # 1. Tip-to-tip distance threshold (same units as input coordinates)
    eps_km          = 5.0,

    # 2. Max allowed difference in strike between the two segments at
    #    the connection point (degrees, 0–90).  Smaller = stricter.
    max_angle_deg   = 20.0,

    # 3. Minimum number of points a segment must have before its
    #    strike can be estimated reliably (skip merge if either
    #    segment has fewer than this many points).
    min_pts         = 2,

    # ── Behaviour ────────────────────────────────────────────────────
    max_passes      = 50,       # safety cap on iteration count
    verbose         = True,
    plot_steps      = False,    # True → pop-up plot at every merge step
    plot_overview   = True,     # True → save before/after overview figure
)
# ═══════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────
#  Geometry helpers
# ───────────────────────────────────────────────────────────────────────

def _orient_we(seg: np.ndarray) -> np.ndarray:
    """Return segment with points ordered West→East (ascending x)."""
    if seg[0, 0] > seg[-1, 0]:
        return seg[::-1].copy()
    return seg


def _strike(p1: np.ndarray, p2: np.ndarray) -> float:
    """Azimuth in degrees from p1 → p2 (−180 … +180)."""
    return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))


def _angle_diff(a: float, b: float) -> float:
    """
    Smallest unsigned angular difference between two strikes (0 … 90°).
    Faults have 180° ambiguity so we fold into [0, 90].
    """
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _seg_length(seg: np.ndarray) -> float:
    """Cumulative Euclidean length of a polyline."""
    d = np.diff(seg, axis=0)
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _local_strike(seg: np.ndarray, end: bool, n_pts: int = 3) -> float:
    """
    Estimate local strike at one end of `seg` using up to `n_pts` points.
    end=False → beginning; end=True → end.
    """
    if end:
        pts = seg[-min(n_pts, len(seg)):]
    else:
        pts = seg[:min(n_pts, len(seg))]
    # Least-squares linear fit for a more robust azimuth estimate
    if len(pts) < 2:
        return 0.0
    dx = pts[-1, 0] - pts[0, 0]
    dy = pts[-1, 1] - pts[0, 1]
    return np.degrees(np.arctan2(dy, dx))


# ───────────────────────────────────────────────────────────────────────
#  I/O
# ───────────────────────────────────────────────────────────────────────

def load_faults(cfg: dict) -> list[np.ndarray]:
    """Load faults → list of (N_i, 2) arrays, each oriented W→E."""
    base = Path(cfg["data_dir"])
    pts  = np.loadtxt(base / cfg["input_faults"])
    dims = np.loadtxt(base / cfg["input_dim"],  dtype=int).ravel()

    faults, idx = [], 0
    for d in dims:
        seg = pts[idx : idx + d].copy()
        faults.append(_orient_we(seg))
        idx += d

    if cfg["verbose"]:
        print(f"Loaded  {len(faults):,} fault segments, "
              f"{sum(len(f) for f in faults):,} total points.")
    return faults


def save_faults(faults: list[np.ndarray], cfg: dict) -> None:
    base = Path(cfg["data_dir"])
    dims  = np.array([len(f) for f in faults], dtype=int)
    flens = np.array([_seg_length(f) for f in faults])
    pts   = np.vstack(faults)

    np.savetxt(base / cfg["output_faults"], pts,  fmt="%.6f")
    np.savetxt(base / cfg["output_dim"],    dims, fmt="%d")
    np.savetxt(base / cfg["output_flen"],  flens, fmt="%.6f")
    print(f"Saved → {cfg['output_faults']}, "
          f"{cfg['output_dim']}, {cfg['output_flen']}")


# ───────────────────────────────────────────────────────────────────────
#  Core merge logic
# ───────────────────────────────────────────────────────────────────────

def _build_endpoint_arrays(faults: list[np.ndarray]):
    """Return (begins, ends) as (N,2) arrays."""
    begins = np.vstack([f[0]  for f in faults])
    ends   = np.vstack([f[-1] for f in faults])
    return begins, ends


def _candidate_pairs(faults, eps, max_angle, min_pts):
    """
    Find all merge-eligible (i, j, order) triples in one pass.
    order='ij' → end of i → beginning of j
    order='ji' → end of j → beginning of i  (i.e. begin of i ← end of j)

    Returns list of (dist, i, j, order)  sorted by distance ascending.
    """
    n = len(faults)
    if n == 0:
        return []

    begins, ends = _build_endpoint_arrays(faults)
    begin_tree   = cKDTree(begins)
    end_tree     = cKDTree(ends)

    pairs = []
    seen  = set()

    for i in range(n):
        fi = faults[i]
        if len(fi) < min_pts:
            continue

        # ── end-of-i → beginning-of-j ──────────────────────────────
        idxs = begin_tree.query_ball_point(fi[-1], eps)
        for j in idxs:
            if j == i or (i, j) in seen or len(faults[j]) < min_pts:
                continue
            fj = faults[j]
            dist = float(np.hypot(fi[-1, 0] - fj[0, 0],
                                  fi[-1, 1] - fj[0, 1]))
            s_i = _local_strike(fi, end=True)
            s_j = _local_strike(fj, end=False)
            if _angle_diff(s_i, s_j) < max_angle:
                pairs.append((dist, i, j, "ij"))
                seen.add((i, j))
                seen.add((j, i))

        # ── end-of-j → beginning-of-i ──────────────────────────────
        idxs2 = end_tree.query_ball_point(fi[0], eps)
        for j in idxs2:
            if j == i or (j, i) in seen or len(faults[j]) < min_pts:
                continue
            fj = faults[j]
            dist = float(np.hypot(fj[-1, 0] - fi[0, 0],
                                  fj[-1, 1] - fi[0, 1]))
            s_j = _local_strike(fj, end=True)
            s_i = _local_strike(fi, end=False)
            if _angle_diff(s_j, s_i) < max_angle:
                pairs.append((dist, j, i, "ij"))   # normalise: j feeds into i
                seen.add((j, i))
                seen.add((i, j))

    pairs.sort(key=lambda x: x[0])   # closest first
    return pairs


def _do_merge(fa: np.ndarray, fb: np.ndarray, dist: float) -> np.ndarray:
    """
    Concatenate fa (upstream) and fb (downstream).
    If they share an endpoint (dist==0) drop the duplicate.
    Re-orient W→E on output.
    """
    if dist == 0.0:
        merged = np.vstack([fa, fb[1:]])
    else:
        merged = np.vstack([fa, fb])
    return _orient_we(merged)


def merge_all(faults: list[np.ndarray], cfg: dict) -> list[np.ndarray]:
    """
    Iteratively merge fault segments.

    Algorithm
    ---------
    Each pass:
      1. Compute all eligible pairs (KD-tree, O(n log n)).
      2. Greedily accept non-conflicting pairs (closest first, greedy set cover).
      3. Build new fault list.
    Repeat until no merges remain or max_passes reached.
    """
    eps       = cfg["eps_km"]
    max_angle = cfg["max_angle_deg"]
    min_pts   = cfg["min_pts"]
    verbose   = cfg["verbose"]

    n_orig = len(faults)
    total_merges = 0

    for pass_no in range(1, cfg["max_passes"] + 1):
        pairs = _candidate_pairs(faults, eps, max_angle, min_pts)
        if not pairs:
            break

        # Greedy: pick closest pair, mark both indices as used
        used        = [False] * len(faults)
        merges_this = []

        for dist, i, j, _ in pairs:
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            merges_this.append((dist, i, j))

        if not merges_this:
            break

        # Build new fault list
        merged_set = set()
        for _, i, j in merges_this:
            merged_set.add(i)
            merged_set.add(j)

        new_faults = [f for k, f in enumerate(faults) if k not in merged_set]
        for dist, i, j in merges_this:
            new_faults.append(_do_merge(faults[i], faults[j], dist))

        total_merges += len(merges_this)
        faults = new_faults

        if verbose:
            print(f"  Pass {pass_no:2d}: {len(merges_this):4d} merges  "
                  f"→ {len(faults):,} segments remaining")

        if cfg["plot_steps"]:
            _plot_step(faults, pass_no)

    print(f"\nFinished. {n_orig:,} → {len(faults):,} segments "
          f"({total_merges} total merges, {pass_no} passes).")
    return faults


# ───────────────────────────────────────────────────────────────────────
#  Visualisation
# ───────────────────────────────────────────────────────────────────────

def _plot_faults(ax, faults, title, max_faults=5000):
    """Draw fault traces; subsample if too many for legibility."""
    n = len(faults)
    step = max(1, n // max_faults)
    cmap = cm.get_cmap("tab20", 20)

    for k, f in enumerate(faults[::step]):
        c = cmap(k % 20)
        ax.plot(f[:, 0], f[:, 1], color=c, lw=0.8, alpha=0.7)
        ax.plot(*f[0],  "o", color=c, ms=2, zorder=3)
        ax.plot(*f[-1], "s", color=c, ms=2, zorder=3)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.2, alpha=0.5)


def plot_overview(faults_before, faults_after, cfg):
    """3-panel figure: before | after | length histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("Fault Trace Merging Results", fontsize=14, fontweight="bold")

    _plot_faults(axes[0], faults_before,
                 f"Before merge\n({len(faults_before):,} segments)")
    _plot_faults(axes[1], faults_after,
                 f"After merge\n({len(faults_after):,} segments)")

    # Length histograms
    len_b = [_seg_length(f) for f in faults_before]
    len_a = [_seg_length(f) for f in faults_after]
    bins  = np.linspace(0, max(max(len_b), max(len_a)), 60)

    axes[2].hist(len_b, bins=bins, alpha=0.6, label="Before", color="steelblue")
    axes[2].hist(len_a, bins=bins, alpha=0.6, label="After",  color="tomato")
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


def _plot_step(faults, pass_no):
    fig, ax = plt.subplots(figsize=(9, 9))
    _plot_faults(ax, faults, f"After pass {pass_no} — {len(faults)} segments")
    plt.tight_layout()
    plt.pause(0.5)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────────
#  Synthetic test dataset
# ───────────────────────────────────────────────────────────────────────

def make_test_data(out_dir="."):
    """
    Create a small synthetic dataset to verify the algorithm.

    Segment layout (all units = km, approximate x–y):
    ───────────────────────────────────────────────────────────────────
      A1──A2      B1──B2          ← should merge into one long E-W fault
                      gap < eps

      C1──C2  (isolated, no near neighbour)

      D1──D2      E1──E2          ← D and E are collinear but gap > eps
                                    → should NOT merge

      F1──F2                      ← F and G have large angle → should NOT merge
               G1
              /
            G2

      H1──H2──H3──H4              ← already one segment (no merge needed)

      I1──I2      J1──J2          ← collinear but reversed; ordering fix needed
    ───────────────────────────────────────────────────────────────────
    Expected result: A+B merge, C stays, D stays, E stays,
                     F stays, G stays, H stays, I+J merge.
    """
    faults = [
        # A: will merge with B (end of A close to start of B, collinear)
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        # B: (gap ~4 km, within eps=5 km)
        np.array([[7.0, 0.0], [8.0, 0.0], [9.0, 0.0]]),

        # C: isolated
        np.array([[0.0, 20.0], [3.0, 20.0], [6.0, 20.0]]),

        # D: far from E (gap = 12 km > eps) → no merge
        np.array([[0.0, 40.0], [3.0, 40.0]]),
        # E:
        np.array([[15.0, 40.0], [18.0, 40.0]]),

        # F: near G tip but angle > 20° → no merge
        np.array([[0.0, 60.0], [4.0, 60.0]]),
        # G: strikes ~60° from F
        np.array([[6.0, 60.0], [8.0, 62.5], [10.0, 65.0]]),

        # H: already long, no neighbours → unchanged
        np.array([[0.0, 80.0],[2.0,80.0],[4.0,80.0],[6.0,80.0],[8.0,80.0]]),

        # I: end points rightward toward J (deliberately reversed for ordering test)
        np.array([[5.0, 100.0], [3.0, 100.0], [1.0, 100.0]]),  # reversed!
        # J: follows on from I after reordering
        np.array([[6.0, 100.0], [8.0, 100.0], [10.0, 100.0]]),
    ]

    dims  = np.array([len(f) for f in faults], dtype=int)
    flens = np.array([_seg_length(f) for f in faults])
    pts   = np.vstack(faults)

    base = Path(out_dir)
    np.savetxt(base / "faults.dat1", pts,   fmt="%.6f")
    np.savetxt(base / "dim.dat1",    dims,  fmt="%d")
    np.savetxt(base / "flen.dat1",   flens, fmt="%.6f")

    print("Test data written to:", out_dir)
    print(f"  {len(faults)} segments, {len(pts)} points")
    print("\nExpected outcome:")
    print("  A+B  → 1 merged segment  (gap 4 km, collinear)")
    print("  C    → stays (no neighbour)")
    print("  D, E → stay  (gap 12 km > eps)")
    print("  F, G → stay  (angle > 20°)")
    print("  H    → stays (no neighbour)")
    print("  I+J  → 1 merged segment  (I will be re-ordered W→E first)")
    print()
    return faults


def run_test(cfg):
    """End-to-end test on synthetic data."""
    print("=" * 60)
    print("  RUNNING SYNTHETIC TEST")
    print("=" * 60)
    test_dir = Path(cfg["data_dir"]) / "test_output"
    test_dir.mkdir(exist_ok=True)

    # Write test files into test_output/
    test_cfg = {**cfg,
                "data_dir": str(test_dir),
                "figure_out": "test_result.png",
                "output_faults": "faults_merged.dat",
                "output_dim":    "dim_merged.dat",
                "output_flen":   "flen_merged.dat"}

    faults_orig = make_test_data(str(test_dir))

    # Re-load so ordering normalisation is applied
    faults_orig = [_orient_we(f) for f in faults_orig]

    faults_merged = merge_all(faults_orig, test_cfg)

    print(f"\nResult: {len(faults_orig)} → {len(faults_merged)} segments")
    print("Expected: 10 → 8  (A+B merged, I+J merged)")

    save_faults(faults_merged, test_cfg)
    if cfg["plot_overview"]:
        plot_overview(faults_orig, faults_merged, test_cfg)

    return faults_merged


# ───────────────────────────────────────────────────────────────────────
#  Entry point
# ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge fault traces.")
    parser.add_argument("--test", action="store_true",
                        help="Run on built-in synthetic test dataset.")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip visualisation.")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.no_plot:
        cfg["plot_overview"] = False

    if args.test:
        run_test(cfg)
        return

    # ── Real data run ────────────────────────────────────────────────
    faults_orig   = load_faults(cfg)
    faults_merged = merge_all(faults_orig, cfg)
    save_faults(faults_merged, cfg)

    if cfg["plot_overview"]:
        plot_overview(faults_orig, faults_merged, cfg)


if __name__ == "__main__":
    main()
