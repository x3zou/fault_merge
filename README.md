# fault_merge

A Python tool for merging nearby, geometrically aligned fault trace segments. Designed as a fast, well-structured replacement for legacy MATLAB fault-merging workflows.

---

## Overview

Fault trace datasets commonly contain many short, fragmented segments that represent portions of the same physical fault. `fault_merge.py` iteratively merges these fragments using three independent geometric criteria:

1. **Tip-to-tip distance** — endpoints must be within `eps_km` of each other
2. **Strike alignment** — the local strike difference at the connection point must be below `max_angle_deg`
3. **Minimum segment size** — segments with fewer than `min_pts` points are skipped to avoid unreliable strike estimates

Merging is repeated in passes until no further candidates remain.

---

## Input Files

The tool expects three plain-text files:

| File | Description |
|---|---|
| `faults.dat1` | Two-column `(x, y)` coordinates of all fault trace points, stacked continuously |
| `dim.dat1` | Integer per row — number of points belonging to each fault segment |
| `flen.dat1` | Length of each fault segment (same units as coordinates) |

The total number of rows in `faults.dat1` must equal the sum of all values in `dim.dat1`.

---

## Output Files

| File | Description |
|---|---|
| `faults_merged.dat` | Merged fault point coordinates |
| `dim_merged.dat` | Point counts per merged segment |
| `flen_merged.dat` | Lengths of merged segments |
| `fault_merge_result.png` | Before/after map views and length distribution histogram |

---

## Installation

**Python ≥ 3.10** is required.

```bash
pip install numpy scipy matplotlib
```

No additional installation is needed — `fault_merge.py` is a single self-contained script.

---

## Usage

### Run on real data

```bash
python fault_merge.py
```

Input and output paths are read from the `CONFIG` dictionary at the top of the script.

### Run the built-in synthetic test

```bash
python fault_merge.py --test
```

This creates a small 10-segment dataset in `test_output/`, runs the merge, prints expected vs. actual results, and saves a diagnostic figure.

### Skip figure generation

```bash
python fault_merge.py --no-plot
```

---

## Configuration

All parameters are in the `CONFIG` dictionary at the top of `fault_merge.py`. No command-line flags are needed for tuning:

```python
CONFIG = dict(
    # I/O
    data_dir        = ".",
    input_faults    = "faults.dat1",
    input_dim       = "dim.dat1",
    input_flen      = "flen.dat1",
    output_faults   = "faults_merged.dat",
    output_dim      = "dim_merged.dat",
    output_flen     = "flen_merged.dat",
    figure_out      = "fault_merge_result.png",

    # Merge criteria
    eps_km          = 5.0,      # max tip-to-tip distance to consider merging
    max_angle_deg   = 20.0,     # max strike difference at connection (degrees)
    min_pts         = 2,        # min points required for strike estimation

    # Behaviour
    max_passes      = 50,       # safety cap on iteration count
    verbose         = True,
    plot_steps      = False,    # pop-up plot after every merge pass
    plot_overview   = True,     # save before/after summary figure
)
```

---

## Algorithm

```
For each pass:
  1. Extract begin/end endpoints of all segments
  2. Build KD-trees over endpoints (O(n log n))
  3. Query each segment's tip for candidates within eps_km
  4. Filter candidates by strike alignment (local strike estimated
     from the nearest 3 points at each tip)
  5. Greedily accept non-conflicting pairs, closest-first
  6. Concatenate accepted pairs into merged segments
  7. Re-orient all segments West→East
Repeat until no candidates remain.
```

The KD-tree lookup makes each pass scale as **O(n log n)** rather than the O(n²) scan used in the original MATLAB code. On a dataset of ~34,000 segments, the full merge completes in under 20 seconds across ~9 passes.

---

## Synthetic Test Dataset

Running `--test` exercises all merge scenarios:

| Segments | Gap | Angle | Expected result |
|---|---|---|---|
| A + B | 4 km (< eps) | 0° | ✅ Merged |
| D + E | 12 km (> eps) | 0° | ❌ Not merged |
| F + G | 2 km (< eps) | ~60° (> max) | ❌ Not merged |
| I + J | 1 km (< eps) | 0° (I reversed) | ✅ Merged after reordering |
| C, H  | no neighbour | — | ❌ Unchanged |

Expected output: 10 → 8 segments.

---

## Key Improvements Over Original MATLAB Code

| Issue | Solution |
|---|---|
| Slow O(n²) scan with restart-from-zero | KD-tree endpoint lookup; greedy batch merging per pass |
| Scattered global variables | Single `CONFIG` dict; data flows through clean function arguments |
| Inconsistent point ordering | All segments re-oriented West→East on load and after every merge |
| No visualisation | 3-panel figure: before/after maps + length histogram; optional per-pass plots |
| Ad-hoc angle criterion | Three named, independently tunable thresholds with documented geometry |

---

## License

MIT
