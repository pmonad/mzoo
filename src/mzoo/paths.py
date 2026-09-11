"""Run directories: <root>/000N-<proj>/yymmdd-000N-<exp>/000N-<run>/"""

import re
from datetime import date
from pathlib import Path

ROOT = "/data/pmonad/mzoo"


def _next(parent, pattern):
    nums = [int(m[1]) for p in parent.glob("*") if (m := re.fullmatch(pattern, p.name))]
    return max(nums, default=0) + 1


def run_dir(proj, exp, run, root=ROOT):
    """Reuse proj by name; exp is new unless it names an existing exp dir; run is always new."""
    root = Path(root)
    proj_dir = next(root.glob(f"[0-9][0-9][0-9][0-9]-{proj}"), None)
    proj_dir = proj_dir or root / f"{_next(root, r'(\d{4})-.+'):04d}-{proj}"
    exp_dir = proj_dir / exp
    if not exp_dir.is_dir():
        exp_dir = proj_dir / f"{date.today():%y%m%d}-{_next(proj_dir, r'\d{6}-(\d{4})-.+'):04d}-{exp}"
    out = exp_dir / f"{_next(exp_dir, r'(\d{4})-.+'):04d}-{run}"
    out.mkdir(parents=True)
    return out
