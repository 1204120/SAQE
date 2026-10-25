"""Two-sided paired bootstrap tests over queries (10,000 resamples, seed 42) and Holm correction.

For an average over several datasets, two tests are used:
  pooled      all queries of the datasets are resampled together (pass a single array);
  stratified  queries are resampled within each dataset and the statistic is the unweighted average of the
              per-dataset mean differences (pass one array per dataset).
The 95% interval is the percentile interval of the resampled statistic, and
p = (#{|stat* - obs| >= |obs|} + 1) / (B + 1).
"""
from typing import Dict, Sequence

import numpy as np

B_BOOT, SEED = 10000, 42


def _resample(diffs, B=B_BOOT, seed=SEED, chunk=500):
    rng = np.random.default_rng(seed)
    out = np.zeros(B)
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        acc = np.zeros(m)
        for d in diffs:
            acc += d[rng.integers(0, len(d), size=(m, len(d)))].mean(axis=1)
        out[s:s + m] = acc / len(diffs)
    return out


def bootstrap_test(diffs: Sequence[Sequence[float]]) -> Dict[str, float]:
    """diffs: one array of per-query differences (pooled test) or one array per dataset (stratified test)."""
    diffs = [np.asarray(d, dtype=np.float64) for d in diffs]
    obs = float(np.mean([d.mean() for d in diffs]))
    bs = _resample(diffs)
    p = (np.sum(np.abs(bs - obs) >= abs(obs)) + 1) / (len(bs) + 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": obs, "lo": float(lo), "hi": float(hi), "p": float(p), "n": int(sum(len(d) for d in diffs))}


def holm(ps: Dict[str, float]) -> Dict[str, float]:
    """Holm-adjusted p-values (monotone) for a family of tests {name: p}."""
    items = sorted(ps.items(), key=lambda x: x[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out
