# fault_merge

`fault_merge.py` merges nearby, geometrically aligned fault trace segments.

The main improvement in this version is configurable segment orientation. Older versions normalized every segment West-to-East. That works well for mostly East-West fault systems, but can fail for mostly South-North systems because "beginning" and "ending" stop meaning the same thing across segments.

This version supports both.

## Install

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

SciPy is used for fast KD-tree lookup on large datasets. If SciPy is not installed, the script falls back to a slower NumPy radius search so small tests can still run.

## Input files

The script expects three plain-text files in `CONFIG["data_dir"]`.

| File | Meaning |
| --- | --- |
| `faults.dat1` | Two-column `x y` coordinates for all fault trace points |
| `dim.dat1` | Number of points in each fault segment |
| `flen.dat1` | Length of each fault segment |

The number of rows in `faults.dat1` must equal the sum of `dim.dat1`.

## Run

```bash
python fault_merge.py
```

To skip figure generation:

```bash
python fault_merge.py --no-plot
```

To run the built-in synthetic checks:

```bash
python fault_merge.py --test
```

## Orientation options

Set this in the `CONFIG` dictionary:

```python
orientation_axis = "auto"
```

Or override it from the command line:

```bash
python fault_merge.py --orientation auto
python fault_merge.py --orientation x
python fault_merge.py --orientation y
python fault_merge.py --orientation none
```

Options:

| Option | Meaning | Best for |
| --- | --- | --- |
| `auto` | Pick `x` or `y` from the dataset's dominant endpoint displacement | Most use cases |
| `x` | Normalize each segment West-to-East, increasing x | Mostly E-W faults |
| `y` | Normalize each segment South-to-North, increasing y | Mostly S-N faults |
| `none` | Keep input point order | Data that is already carefully ordered |

Aliases are accepted: `ew`, `we`, `sn`, and `ns`.

## Why orientation matters

The merge algorithm connects segment tips. It mostly wants this pattern:

```text
end of segment A -> beginning of segment B
```

If all segments are consistently ordered, this is simple. If some segments are stored in the opposite direction, the algorithm can miss valid merges or test the wrong tips.

For mostly E-W faults, ordering by increasing x is natural:

```text
west tip -> east tip
```

For mostly S-N faults, ordering by increasing y is natural:

```text
south tip -> north tip
```

`orientation_axis = "auto"` chooses between those two rules based on the total endpoint displacement of the dataset.

## Algorithm

For each pass:

1. Normalize each segment using the chosen orientation rule.
2. Extract beginning and ending endpoints.
3. Build two KD-trees:
   - one over all beginnings
   - one over all endings
4. Query nearby opposite tips within `eps_km`.
5. Filter candidates by local strike alignment.
6. Greedily accept non-conflicting pairs, closest first.
7. Concatenate accepted pairs and normalize the merged segments again.

The KD-tree step avoids comparing every endpoint to every other endpoint. Each pass scales much better than a full pairwise scan.

## Output files

| File | Meaning |
| --- | --- |
| `faults_merged.dat` | Merged fault point coordinates |
| `dim_merged.dat` | Point counts per merged segment |
| `flen_merged.dat` | Lengths of merged segments |
| `fault_merge_result.png` | Before/after map views and length histogram |

## Configuration

Most tuning is done in the `CONFIG` dictionary at the top of `fault_merge.py`.

Important fields:

```python
eps_km = 5.0
max_angle_deg = 20.0
min_pts = 2
orientation_axis = "auto"
max_passes = 50
```

## License

MIT
