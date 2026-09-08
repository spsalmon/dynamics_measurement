#!/usr/bin/env python3
"""Batch-fix NIS-exported z-stacks.

Input  : (2, Z, 2, Y, X)  -- acquisition, z, channel-within-acquisition, y, x
Output : (Z, 4, Y, X)     -- with merged channels reordered [1, 0, 3, 2]

Merged channel index is acquisition * 2 + channel, i.e. the naive merge order is
(a0c0, a0c1, a1c0, a1c1). CHANNEL_ORDER is applied on top of that.

Usage:
    python fix_stacks.py IN_DIR OUT_DIR -j 8
    python fix_stacks.py IN_DIR OUT_DIR -j 8 --compression zstd
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from joblib import Parallel, delayed

CHANNEL_ORDER = [1, 0, 3, 2]

# (acquisition, channel) tuple feeding each output channel
_SRC = [(k // 2, k % 2) for k in CHANNEL_ORDER]


def transform(arr: np.ndarray) -> np.ndarray:
    """(A, Z, C, Y, X) -> (Z, A*C, Y, X), reordered. One allocation, no temporaries."""
    if arr.ndim != 5 or arr.shape[0] != 2 or arr.shape[2] != 2:
        raise ValueError(f"unexpected shape {arr.shape}")
    z, y, x = arr.shape[1], arr.shape[3], arr.shape[4]
    out = np.empty((z, len(_SRC), y, x), dtype=arr.dtype)
    for i, (a, c) in enumerate(_SRC):
        out[:, i] = arr[a, :, c]
    return out


def convert_one(src: Path, in_dir: Path, out_dir: Path,
                overwrite: bool, compression: str | None = "zlib") -> tuple[str, str]:
    """Worker. Takes paths, not arrays -- nothing large crosses the process boundary."""
    rel = src.relative_to(in_dir)
    dst = out_dir / rel
    if dst.exists() and not overwrite:
        return str(rel), "skip"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    try:
        out = transform(tifffile.imread(src))
        # imagej=True gives a hyperstack ImageJ/Fiji reads directly, but ImageJ
        # cannot read compressed TIFFs -- fall back to OME when compressing.
        if compression is None:
            tifffile.imwrite(tmp, out, imagej=True, metadata={"axes": "ZCYX"})
        else:
            tifffile.imwrite(tmp, out, ome=True, metadata={"axes": "ZCYX"},
                             compression=compression)
        tmp.replace(dst)  # atomic: a killed job never leaves a half-written file
        return str(rel), "ok"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return str(rel), f"FAIL {type(exc).__name__}: {exc}"


def make_parallel(n_jobs: int) -> Parallel:
    """Stream results as they finish where joblib supports it (>=1.4 unordered,
    >=1.3 ordered), otherwise fall back to collecting a list at the end."""
    kwargs = dict(n_jobs=n_jobs, backend="loky", batch_size=1, verbose=0)
    for mode in ("generator_unordered", "generator"):
        try:
            return Parallel(return_as=mode, **kwargs)
        except (TypeError, ValueError):
            pass
    return Parallel(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    in_dir = Path("/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/raw_stacks")
    out_dir = Path("/mnt/towbin.data/shared/spsalmon/20260807_134551_682_ZIVA_60x_col10_reporter/fixed_stacks")
    ap.add_argument("-j", "--workers", type=int, default=8)
    ap.add_argument("--pattern", default="*.ome.tiff*")
    ap.add_argument("--compression", default="zlib",
                    help="e.g. zstd, zlib. Default: zlib.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--failures", type=Path, default=Path("failures.txt"))
    args = ap.parse_args()

    files = sorted(p for p in in_dir.rglob(args.pattern) if p.is_file())
    if not files:
        print(f"no files matching {args.pattern!r} under {in_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures: list[str] = []
    t0 = time.monotonic()

    tasks = (delayed(convert_one)(f, in_dir, out_dir,
                                  args.overwrite, args.compression) for f in files)

    for n, (rel, status) in enumerate(make_parallel(args.workers)(tasks), 1):
        if status.startswith("FAIL"):
            counts["fail"] += 1
            failures.append(f"{rel}\t{status}")
        else:
            counts[status] += 1
        if n % 25 == 0 or n == len(files):
            rate = n / (time.monotonic() - t0)
            eta = (len(files) - n) / rate / 60
            print(f"\r{n}/{len(files)}  {rate:.1f} file/s  ETA {eta:.1f} min  "
                  f"ok={counts['ok']} skip={counts['skip']} fail={counts['fail']}",
                  end="", file=sys.stderr, flush=True)

    print(file=sys.stderr)
    if failures:
        args.failures.write_text("\n".join(failures) + "\n")
        print(f"{len(failures)} failures written to {args.failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())