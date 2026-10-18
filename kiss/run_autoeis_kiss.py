#!/usr/bin/env python3
"""Run the AutoEIS pipeline on kiss EISF spectra.

Takes a single EISF 1.1.0 file or a tar archive of such files (see
kiss2autoeis.py for the format) and runs the full AutoEIS workflow on
each spectra:

    GEP circuit generation -> heuristic filter -> Bayesian inference
    -> fitness metrics (WAIC / R^2 / MAPE) -> ranking by WAIC

Outputs (in -o OUTDIR):
    spectra/<name>.csv        ranked circuits per spectra (all fitness columns)
    spectra/<name>_best.json  parameter estimates for the best-ranked circuit
    summary.csv               one row per spectra: pool sizes, best circuit,
                              top-1/top-3 hit vs ground-truth class, runtime

Usage:
    run_autoeis_kiss.py FILE.csv -o OUTDIR
    run_autoeis_kiss.py ARCHIVE.tar -o OUTDIR --limit 10
    run_autoeis_kiss.py ARCHIVE.tar -o OUTDIR --classes r-p,rp-p --iters 100

Environment (needed by the Julia GEP backend):
    PYTHON_JULIAPKG_EXE=/path/to/julia
    JULIA_DEPOT_PATH=/path/to/julia/depot

Notes:
    - Impedance is used as-is (raw ohms); only the frequency is converted
      from rad/s to Hz.
    - WAIC values are only comparable within one spectra, never across
      spectra with different impedance scales.
    - If the parallel pools cannot start (e.g., /dev/shm unavailable),
      AutoEIS falls back to serial execution automatically.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from kiss2autoeis import parse_eisf_file, parse_eisf_tar  # noqa: E402

from autoeis import core  # noqa: E402

AE_TO_KISS = {"R": "r", "C": "c", "L": "l", "P": "p"}


# ---------------------------------------------------------------------------
# Topology canonicalization (for comparing predicted vs ground-truth class)
# ---------------------------------------------------------------------------

def _split_top(s: str, delim: str, open_ch: str, close_ch: str) -> list[str]:
    """Split on ``delim`` only at bracket depth 0."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch == delim and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
    out.append(cur)
    return [x for x in out if x]


def _canon(struct) -> str:
    """Render a parsed structure; series/parallel children are sorted."""
    if isinstance(struct, str):
        return struct
    op, *children = struct
    return op + "".join(_canon(c) for c in sorted(children, key=_canon))


def canonical_kiss(s: str) -> str:
    """Canonical form of a kiss model string (params already stripped).

    kiss grammar: series = '-', parallel = juxtaposition, '(' groups a series.
    """

    def parse_circuit(s):
        return ("s", *[parse_node(n) for n in _split_top(s, "-", "(", ")")])

    def parse_node(n):
        # juxtaposition = parallel; a '( ... )' group is a single element
        parts, i = [], 0
        while i < len(n):
            if n[i] == "(":
                depth, j = 0, i
                while j < len(n):
                    if n[j] == "(":
                        depth += 1
                    elif n[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                parts.append(parse_circuit(n[i + 1 : j]))
                i = j + 1
            else:
                parts.append(n[i])
                i += 1
        if len(parts) == 1:
            return parts[0]
        return ("p", *parts)

    return _canon(parse_circuit(s.strip()))


def canonical_autoeis(circuit: str) -> str:
    """Canonical form of an AutoEIS circuit string.

    AutoEIS grammar: series = '-', parallel = '[...]' with ',' branches.
    """
    s = re.sub(r"([A-Za-z])\d+", r"\1", circuit)

    def parse_circuit(s):
        return ("s", *[parse_branch(b) for b in _split_top(s, "-", "[", "]")])

    def parse_branch(b):
        if b.startswith("[") and b.endswith("]"):
            parts = [parse_branch(p) for p in _split_top(b[1:-1], ",", "[", "]")]
            if len(parts) == 1:
                return parts[0]
            return ("p", *parts)
        if "-" in b:
            return parse_circuit(b)  # series chain inside a parallel branch
        return AE_TO_KISS.get(b, b.lower())

    return _canon(parse_circuit(s))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    freq: np.ndarray,
    Z: np.ndarray,
    iters: int,
    tol: float,
    terminals: str,
    complexity: int,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    seed: int | None,
    method: str,
    parallel: bool,
) -> dict:
    """Runs the AutoEIS workflow on one spectra.

    Returns a dict with pool sizes, the ranked circuits dataframe (or None),
    and the per-stage runtime in seconds.
    """
    out = {
        "n_candidates": 0,
        "n_plausible": 0,
        "n_converged": 0,
        "ranked": None,
        "best": None,
        "t_gep": 0.0,
        "t_mcmc": 0.0,
    }

    t0 = time.time()
    circuits = core.generate_equivalent_circuits(
        freq, Z, iters=iters, tol=tol, terminals=terminals,
        complexity=complexity, seed=seed, parallel=parallel,
    )
    out["t_gep"] = time.time() - t0
    out["n_candidates"] = len(circuits)
    if len(circuits) == 0:
        print("    GEP: empty candidate pool")
        return out

    circuits_f = core.filter_implausible_circuits(circuits)
    out["n_plausible"] = len(circuits_f)
    if len(circuits_f) == 0:
        print("    GEP: no plausible circuits after filtering")
        return out

    t0 = time.time()
    results = core.perform_bayesian_inference(
        circuits_f, freq, Z,
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        seed=seed, method=method, progress_bar=False, parallel=parallel,
    )
    out["t_mcmc"] = time.time() - t0

    conv = [r.converged for r in results]
    circuits_f = circuits_f[conv].reset_index(drop=True)
    results = [r for r, c in zip(results, conv) if c]
    out["n_converged"] = len(results)
    if len(results) == 0:
        print("    MCMC: no converged circuits")
        return out
    circuits_f["InferenceResult"] = results

    circuits_f = core.compute_fitness_metrics(circuits_f, freq, Z)
    if "WAIC (real)" in circuits_f.columns:
        w1, w2 = "WAIC (real)", "WAIC (imag)"
    else:
        w1, w2 = "WAIC (mag)", "WAIC (phase)"
    circuits_f["WAIC (sum)"] = circuits_f[w1] + circuits_f[w2]
    circuits_f = circuits_f.sort_values("WAIC (sum)").reset_index(drop=True)

    out["ranked"] = circuits_f
    out["best"] = circuits_f.iloc[0]
    return out


def circuit_json(result, waic_sum: float) -> dict:
    """Parameter estimates (mean/std/median/5-95%) for one InferenceResult."""
    params = {}
    for var in result.variables:
        arr = np.asarray(result.samples[var]).reshape(-1)
        params[var] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    return {
        "circuit": result.circuit,
        "converged": result.converged,
        "divergences": int(result.num_divergences),
        "WAIC (sum)": float(waic_sum),
        "parameters": params,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("source", help="EISF file or tar archive of EISF files")
    ap.add_argument("-o", "--outdir", required=True, help="output directory")
    # GEP
    ap.add_argument("--iters", type=int, default=100, help="GEP iterations (default 100)")
    ap.add_argument("--tol", type=float, default=1e-2, help="GEP convergence tol (default 1e-2)")
    ap.add_argument("--terminals", default="RLP", help="GEP element terminals (default RLP)")
    ap.add_argument("--complexity", type=int, default=12, help="GEP search space complexity (default 12)")
    # MCMC
    ap.add_argument("--warmup", type=int, default=2500, help="MCMC warmup samples (default 2500)")
    ap.add_argument("--samples", type=int, default=1000, help="MCMC samples (default 1000)")
    ap.add_argument("--chains", type=int, default=1, help="MCMC chains (default 1)")
    ap.add_argument(
        "--method", default="bode",
        choices=["bode", "nyquist", "magnitude", "chi-squared", "phase"],
        help="inference objective (default bode)",
    )
    # selection / misc
    ap.add_argument("--seed", type=int, default=None, help="random seed (default None)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N spectra")
    ap.add_argument("--classes", default=None, help="comma-separated classes to include")
    ap.add_argument(
        "--no-parallel", action="store_true",
        help="force serial execution (default: parallel with serial fallback)",
    )
    args = ap.parse_args(argv)

    source = Path(args.source)
    outdir = Path(args.outdir)
    specdir = outdir / "spectra"
    specdir.mkdir(parents=True, exist_ok=True)

    if source.suffix == ".tar":
        parsed = parse_eisf_tar(source)
        entries = list(parsed.items())
    else:
        entries = [(source.stem, parse_eisf_file(source))]

    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",")}
        entries = [(n, s) for n, s in entries if s.class_string in wanted]
    if args.limit is not None:
        entries = entries[: args.limit]
    print(f"processing {len(entries)} spectra -> {outdir}")

    parallel = not args.no_parallel
    summary_rows = []
    for i, (name, s) in enumerate(entries, 1):
        base = Path(name).stem
        freq, Z = s.to_autoeis()
        print(f"[{i}/{len(entries)}] {base} (class {s.class_string}, {len(freq)} pts)")

        t0 = time.time()
        res = run_pipeline(
            freq, Z,
            iters=args.iters, tol=args.tol, terminals=args.terminals,
            complexity=args.complexity,
            num_warmup=args.warmup, num_samples=args.samples,
            num_chains=args.chains, seed=args.seed, method=args.method,
            parallel=parallel,
        )
        runtime = time.time() - t0

        row = {
            "name": base,
            "class": s.class_string,
            "n_points": len(freq),
            "fmin_Hz": f"{freq.min():.6g}",
            "fmax_Hz": f"{freq.max():.6g}",
            "n_candidates": res["n_candidates"],
            "n_plausible": res["n_plausible"],
            "n_converged": res["n_converged"],
            "top1_circuit": "",
            "top1_waic": "",
            "top2_circuit": "",
            "top2_waic": "",
            "top3_circuit": "",
            "top3_waic": "",
            "top1_canonical": "",
            "gt_canonical": canonical_kiss(s.class_string),
            "top1": False,
            "top3": False,
            "t_gep_s": f"{res['t_gep']:.1f}",
            "t_mcmc_s": f"{res['t_mcmc']:.1f}",
            "runtime_s": f"{runtime:.1f}",
        }

        if res["ranked"] is not None:
            best = res["best"]
            top3_df = res["ranked"].head(3)
            for k in range(len(top3_df)):
                row[f"top{k + 1}_circuit"] = top3_df.iloc[k].circuitstring
                row[f"top{k + 1}_waic"] = f"{top3_df.iloc[k]['WAIC (sum)']:.4g}"
            row["top1_canonical"] = canonical_autoeis(row["top1_circuit"])
            row["top1"] = row["top1_canonical"] == row["gt_canonical"]
            top3_canon = [canonical_autoeis(c) for c in top3_df.circuitstring]
            row["top3"] = row["gt_canonical"] in top3_canon

            res["ranked"].drop(
                columns=["InferenceResult", "Z_pred"]
            ).to_csv(specdir / f"{base}.csv", index=False)
            top3_json = {
                f"top{k + 1}": circuit_json(
                    top3_df.iloc[k]["InferenceResult"],
                    top3_df.iloc[k]["WAIC (sum)"],
                )
                for k in range(len(top3_df))
            }
            with open(specdir / f"{base}_best.json", "w") as f:
                json.dump(top3_json, f, indent=2)
            print(
                f"    best: {best.circuitstring}  WAIC {best['WAIC (sum)']:.1f}  "
                f"top1={row['top1']} top3={row['top3']}  ({runtime:.0f}s)"
            )
        else:
            print(f"    no ranked circuits  ({runtime:.0f}s)")

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    n = len(summary)
    n_ranked = int((summary["n_converged"] > 0).sum())
    n1 = int(summary["top1"].sum())
    n3 = int(summary["top3"].sum())
    print(f"\n=== summary: {n} spectra, {n_ranked} with ranked circuits ===")
    print(f"top-1 accuracy: {n1}/{n} = {n1 / n:.0%}")
    print(f"top-3 accuracy: {n3}/{n} = {n3 / n:.0%}")
    print(f"summary written to {outdir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
