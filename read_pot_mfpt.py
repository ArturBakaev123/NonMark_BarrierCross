#!/usr/bin/env python
# coding: utf-8

# In[1]:

#!/usr/bin/env python3
"""
Analyze first-passage data for a double-well potential over
tau_m, tau_o, and tau_phi.

Expected input file naming (adjust the prefix if needed):
  <prefix>hist_unnorm i_paral={i}, j_trajpatch={j}, tau_m={tm:.3f}, tau_phi={tp:.3f}, tau_o={to:.3f}.txt
  <prefix>counts      i_paral={i}, j_trajpatch={j}, tau_m={tm:.3f}, tau_phi={tp:.3f}, tau_o={to:.3f}.txt
  <prefix>fptsum      i_paral={i}, j_trajpatch={j}, tau_m={tm:.3f}, tau_phi={tp:.3f}, tau_o={to:.3f}.txt

Example usage:
  python read_pot_mfpt.py \
  --pot-dir data/pots \
  --fpt-dir data/fpts \
  --out-dir results \
  --edges-file data/edges.txt \
  --tau-m 0.01 1.0 \
  --tau-o 0.1 0.6 1.0 1.7 2.7 5.0 \
  --tau-phi 0.01 0.1 1.0 10.0 100.0 \
  --N-fac 4e8 \
  --patch-traj 100
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze first-passage and free-energy data over tau_m, tau_o, tau_phi."
    )

    parser.add_argument("--pot-dir", type=Path, required=True,
                        help="Directory containing histogram/potential files.")
    parser.add_argument("--fpt-dir", type=Path, required=True,
                        help="Directory containing counts and fptsum files.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory.")
    parser.add_argument("--edges-file", type=Path, required=True,
                        help="Text file containing histogram bin edges.")

    parser.add_argument("--tau-m", type=float, nargs="+", required=True,
                        help="List of tau_m / tau_D values.")
    parser.add_argument("--tau-o", type=float, nargs="+", required=True,
                        help="List of tau_o / tau_D values.")
    parser.add_argument("--tau-phi", type=float, nargs="+", required=True,
                        help="List of tau_phi / tau_D values.")

    parser.add_argument("--file-prefix", default="",
                        help="Optional filename prefix, e.g. a run ID.")

    parser.add_argument("--parallel-index", type=int, default=0,
                        help="Value of i_paral in the input filenames.")

    parser.add_argument("--tau-D", type=float, default=1.0,
                        help="Diffusion timescale tau_D.")
    parser.add_argument("--U0", type=float, default=3.0,
                        help="Barrier height parameter U0.")
    parser.add_argument("--L", type=float, default=1.0,
                        help="Half well separation L.")

    parser.add_argument("--dt-fac", type=float, default=0.01,
                        help="Time-step factor used when computing dt.")
    parser.add_argument("--stride", type=int, default=5,
                        help="Trajectory stride used when computing MFPT.")

    parser.add_argument("--N-fac", type=float, default=4e8,
                        help="Global trajectory-length factor.")
    parser.add_argument("--patch-traj", type=int, default=100,
                        help="Global number of trajectory patches per parameter set.")

    parser.add_argument("--copy-script", action="store_true",
                        help="Copy this script into the output directory for provenance.")

    return parser.parse_args()


def filename(prefix: str, stem: str, i_paral: int, j: int,
             tau_m: float, tau_phi: float, tau_o: float) -> str:
    return (
        f"{prefix}{stem} i_paral={i_paral}, j_trajpatch={j}, "
        f"tau_m={tau_m:.3f}, tau_phi={tau_phi:.3f}, tau_o={tau_o:.3f}.txt"
    )


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_script:
        shutil.copy(sys.argv[0], args.out_dir / Path(sys.argv[0]).name)

    tau_ms = np.asarray(args.tau_m, dtype=float)
    tau_os = np.asarray(args.tau_o, dtype=float)
    tau_phis = np.asarray(args.tau_phi, dtype=float)

    n_m = len(tau_ms)
    n_o = len(tau_os)
    n_p = len(tau_phis)

    edges = np.loadtxt(args.edges_file)
    n_bins = len(edges) - 1

    pos = 0.5 * (edges[1:] + edges[:-1])
    x = np.linspace(np.min(pos), np.max(pos), n_bins)

    # Original expression, kept explicit for clarity.
    energy_real = args.U0 * ((x / args.L) ** 2 - 1.0) ** 2

    hist_all = np.full((n_m, n_o, n_p, n_bins), np.nan, dtype=float)
    fe_all = np.full((n_m, n_o, n_p, n_bins), np.nan, dtype=float)
    R_all = np.full((n_m, n_o, n_p), np.nan, dtype=float)
    mfpt_all = np.full((n_m, n_o, n_p), np.nan, dtype=float)

    for im, tau_m in enumerate(tau_ms):
        for io, tau_o in enumerate(tau_os):
            for ip, tau_phi in enumerate(tau_phis):
                hist_sum = np.zeros(n_bins, dtype=float)
                fpts_tot = np.zeros((2, 2), dtype=float)
                coun_tot = np.zeros((2, 2), dtype=float)

                n_used = 0

                for j in range(args.patch_traj):
                    hist_file = args.pot_dir / filename(
                        args.file_prefix, "hist_unnorm", args.parallel_index, j,
                        tau_m, tau_phi, tau_o
                    )
                    counts_file = args.fpt_dir / filename(
                        args.file_prefix, "counts", args.parallel_index, j,
                        tau_m, tau_phi, tau_o
                    )
                    fptsum_file = args.fpt_dir / filename(
                        args.file_prefix, "fptsum", args.parallel_index, j,
                        tau_m, tau_phi, tau_o
                    )

                    if not (hist_file.exists() and counts_file.exists() and fptsum_file.exists()):
                        warnings.warn(
                            f"Missing files for j={j}, tau_m={tau_m}, "
                            f"tau_phi={tau_phi}, tau_o={tau_o}; skipping patch."
                        )
                        continue

                    hist_sum += np.loadtxt(hist_file)
                    coun_tot += np.loadtxt(counts_file)
                    fpts_tot += np.loadtxt(fptsum_file)
                    n_used += 1

                if n_used == 0:
                    warnings.warn(
                        f"No data found for tau_m={tau_m}, tau_phi={tau_phi}, "
                        f"tau_o={tau_o}; leaving entry as NaN."
                    )
                    continue

                dt = min(tau_m, args.tau_D, tau_phi, tau_o) * args.dt_fac

                denom = coun_tot[0, 0] + coun_tot[1, 1]
                if denom > 0:
                    mfpt_all[im, io, ip] = (
                        (fpts_tot[0, 0] + fpts_tot[1, 1]) / denom
                        * dt
                        * args.stride
                    )

                if hist_sum.sum() > 0:
                    hist_norm = hist_sum / (
                        np.diff(edges) * np.sum(hist_sum)
                    )
                    hist_all[im, io, ip] = hist_norm

                    with np.errstate(divide="ignore", invalid="ignore"):
                        fe = -np.log(hist_norm)

                    if np.all(np.isfinite(fe)):
                        fe -= np.min(fe)
                        fe_all[im, io, ip] = fe
                        R_all[im, io, ip] = np.sqrt(
                            np.sum((fe - energy_real) ** 2) / len(energy_real)
                        )

                print(
                    f"Processed tau_m={tau_m:.3f}, tau_o={tau_o:.3f}, "
                    f"tau_phi={tau_phi:.3f} using {n_used} patches."
                )

    np.save(args.out_dir / "mfpt.npy", mfpt_all)
    np.save(args.out_dir / "hist.npy", hist_all)
    np.save(args.out_dir / "free_energy.npy", fe_all)
    np.save(args.out_dir / "R.npy", R_all)

    np.savetxt(args.out_dir / "tau_m.txt", tau_ms)
    np.savetxt(args.out_dir / "tau_o.txt", tau_os)
    np.savetxt(args.out_dir / "tau_phi.txt", tau_phis)
    np.savetxt(args.out_dir / "bin_edges.txt", edges)
    np.savetxt(args.out_dir / "energy_real.txt", energy_real)

    print(f"Saved results to {args.out_dir}")


if __name__ == "__main__":
    main()