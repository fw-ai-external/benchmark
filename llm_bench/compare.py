"""Compare two sets of llm_bench summaries (A = without a change, B = with it) from their --summary-file CSVs.

    python -m llm_bench.compare --a A1.csv [A2.csv ...] --b B1.csv [B2.csv ...] [--min-noise 0.03] \\
        [--json out.json] [--markdown out.md]

Every row of a side's files is one repeat. The two sides are only comparable if they saw the same workload: the
workload files' sha256, the options that change the traffic (replayed pod, retiming, arrival process, -t, corpus,
tokenizer, output mode, chat mode ...), the request seed, the number of warm-up requests, the target rate and the
llm_bench version must match (across the repeats of a side too); otherwise this refuses (exit 2) and says which
differ. Rows that are not measured runs (a --prefix-fill step) are ignored, and said so. Else it reports per-metric
deltas against a noise floor, max(the spread of the repeats on either side, --min-noise): a delta inside it reads
"same", unless the floor is over twice --min-noise, when the repeats are too noisy to say ("inconclusive"). A metric
that is 0 on A and not on B reads "different". A run that failed its delivery check or its --max-fail-ratio is
flagged in the output (it is still in the averages). Exit 0 compared, 2 refused, 1 error.

For a trustworthy comparison, run at least 3 repeats per side, alternate A and B (don't run all of one side first),
and don't run them alongside other load; P99 is the noisiest figure.

Run B with --warmup-requests N, N being the "Warmup Requests" A reported, so both get the same warm-up and the same
measured requests.
"""

import argparse
import csv
import json
import math
import sys
from typing import Any, Optional

# Summary columns (llm_bench's pretty names) that make up a run's workload identity: (column, label)
IDENTITY = [
    ("Workload Sha256", "workload file sha256"),
    ("Workload Options Sha256", "workload options"),
    ("Request Seed", "request seed"),
    ("Warmup Requests", "warm-up requests"),
    ("Target Qps", "rate (target QPS)"),
    ("Llm Bench Version", "llm_bench version"),
]

# (column, higher is better, kind): kind "rel" compares as a relative change, "abs" as a difference (fractions)
METRICS = [
    ("Qps", True, "rel"),
    ("Delivered Qps", True, "rel"),
    ("P50 Time To First Token", False, "rel"),
    ("P90 Time To First Token", False, "rel"),
    ("P99 Time To First Token", False, "rel"),
    ("Latency Per Token", False, "rel"),
    ("P50 Total Latency", False, "rel"),
    ("P90 Total Latency", False, "rel"),
    ("P99 Total Latency", False, "rel"),
    ("Measured Hit Rate", True, "abs"),
    ("Fail Ratio", False, "abs"),
    ("Requests Failed", False, "rel"),
]


def _not_measured(row: dict[str, str]) -> Optional[str]:
    """Why a summary row isn't a measured run (a --prefix-fill step, a --prefix-reuse-probe), else None."""
    if row.get("Mode") == "prefix_fill":
        return "a --prefix-fill step"
    if row.get("Prefix Reuse Probe Points"):
        return "a --prefix-reuse-probe step"
    return None


def read_rows(paths: list[str], ignored: Optional[list[str]] = None) -> list[dict[str, str]]:
    """The measured rows of the summary CSVs; the others are left out and described in `ignored`."""
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 1):
                why = _not_measured(row)
                if why is None:
                    rows.append(row)
                elif ignored is not None:
                    ignored.append(f"{path} row {n}: {why}")
    if not rows:
        raise ValueError(f"no measured summary rows in {', '.join(paths)}")
    return rows


def _same(a: str, b: str) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except ValueError:
        return a == b


def _option_differences(a: dict[str, str], b: dict[str, str]) -> str:
    """Which workload options differ, from the summaries' Workload Options JSON (empty when it isn't there)."""
    try:
        va, vb = json.loads(a.get("Workload Options", "")), json.loads(b.get("Workload Options", ""))
    except ValueError:
        return ""
    return ", ".join(
        f"{k}: A {va.get(k)!r} vs B {vb.get(k)!r}" for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)
    )


def identity_differences(a: list[dict[str, str]], b: list[dict[str, str]]) -> list[str]:
    """Why the two sides' workloads differ (and a side's repeats too), empty when they are the same."""
    problems = []
    for column, label in IDENTITY:
        values = {side: [r.get(column, "") for r in rows] for side, rows in (("A", a), ("B", b))}
        for side, found in values.items():
            if not all(_same(found[0], v) for v in found):
                problems.append(f"{label} differs between the {side} repeats: {sorted(set(found))}")
        if not _same(values["A"][0], values["B"][0]):
            detail = _option_differences(a[0], b[0]) if column == "Workload Options Sha256" else ""
            problems.append(
                f"{label}: A {values['A'][0] or '(none)'} vs B {values['B'][0] or '(none)'}"
                + (f" ({detail})" if detail else "")
            )
        elif not values["A"][0] and column in ("Workload Sha256", "Workload Options Sha256", "Llm Bench Version"):
            problems.append(f"{label} is missing from the summaries: not written by a warm-up-aware llm_bench")
    return problems


def _numbers(rows: list[dict[str, str]], column: str) -> list[float]:
    out = []
    for row in rows:
        try:
            value = float(row.get(column, ""))
        except ValueError:
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def _spread(values: list[float], kind: str) -> float:
    """How far a side's repeats scatter: (max - min) / mean, or max - min for fractions; 0 with one repeat."""
    if len(values) < 2:
        return 0.0
    width = max(values) - min(values)
    mean = sum(values) / len(values)
    return width if kind == "abs" else (width / abs(mean) if mean else 0.0)


def compare(a: list[dict[str, str]], b: list[dict[str, str]], min_noise: float) -> list[dict[str, Any]]:
    results = []
    for column, higher_is_better, kind in METRICS:
        va, vb = _numbers(a, column), _numbers(b, column)
        if not va or not vb:
            continue
        mean_a, mean_b = sum(va) / len(va), sum(vb) / len(vb)
        noise = max(_spread(va, kind), _spread(vb, kind), min_noise)
        if kind == "abs":
            delta = mean_b - mean_a
        elif mean_a:
            delta = (mean_b - mean_a) / abs(mean_a)
        elif mean_b:
            kind, delta, noise = "abs", mean_b - mean_a, 0.0  # a relative change from 0 means nothing: the difference
        else:
            delta = 0.0
        if kind == "abs" and noise == 0.0 and delta:
            verdict = "different"
        elif abs(delta) <= noise:
            verdict = "same" if noise <= 2 * min_noise else "inconclusive (noisy repeats)"
        else:
            verdict = "better" if (delta > 0) == higher_is_better else "worse"
        results.append(
            {
                "metric": column,
                "a": mean_a,
                "b": mean_b,
                "delta": delta,
                "delta_kind": kind,
                "noise": noise,
                "verdict": verdict,
                "repeats_a": len(va),
                "repeats_b": len(vb),
            }
        )
    return results


def _format_delta(result: dict[str, Any]) -> str:
    if result["delta"] is None:
        return "n/a"
    return f"{result['delta']:+.4f}" if result["delta_kind"] == "abs" else f"{100 * result['delta']:+.1f}%"


def flagged_runs(side: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Runs that did not deliver their traffic or failed too many requests: their figures are not a clean data point."""
    flagged = []
    for n, row in enumerate(rows, 1):
        reasons = []
        if row.get("Delivery Check") == "fail":
            reasons.append(f"delivery check failed ({row.get('Delivery Failures', '')})")
        if row.get("Failed") == "1":
            reasons.append(f"exceeded --max-fail-ratio (Fail Ratio {row.get('Fail Ratio', '')})")
        if reasons:
            flagged.append({"side": side, "run": n, "reasons": reasons})
    return flagged


def _flag_lines(flagged: list[dict[str, Any]]) -> list[str]:
    return [f"!!! {f['side']} run {f['run']}: {'; '.join(f['reasons'])} (it is in the figures above)" for f in flagged]


def markdown(results: list[dict[str, Any]], a_runs: int, b_runs: int, flagged: Optional[list] = None) -> str:
    lines = [
        f"A: {a_runs} run(s), B: {b_runs} run(s)",
        "",
        "| metric | A | B | delta | noise | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        noise = f"{r['noise']:.4f}" if r["delta_kind"] == "abs" else f"{100 * r['noise']:.1f}%"
        lines.append(f"| {r['metric']} | {r['a']:.4g} | {r['b']:.4g} | {_format_delta(r)} | {noise} | {r['verdict']} |")
    return "\n".join(lines + [""] + _flag_lines(flagged or [])) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True, metavar="SUMMARY", help="A's summary CSV(s).")
    parser.add_argument("--b", nargs="+", required=True, metavar="SUMMARY", help="B's summary CSV(s).")
    parser.add_argument("--min-noise", type=float, default=0.03, help="Least noise floor (default 0.03).")
    parser.add_argument("--json", default=None, help="Write the comparison here as JSON.")
    parser.add_argument("--markdown", default=None, help="Write the comparison here as a markdown table.")
    args = parser.parse_args(argv)
    try:
        ignored: list[str] = []
        a, b = read_rows(args.a, ignored), read_rows(args.b, ignored)
        for note in ignored:
            print(f"compare: ignoring {note} (not a measured run)", file=sys.stderr)
        problems = identity_differences(a, b)
        if problems:
            print("Refusing to compare: the two sides differ in their workload identity:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        results = compare(a, b, args.min_noise)
        flagged = flagged_runs("A", a) + flagged_runs("B", b)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {
                        "a_runs": len(a),
                        "b_runs": len(b),
                        "metrics": results,
                        "flagged_runs": flagged,
                        "ignored": ignored,
                    },
                    f,
                    indent=2,
                )
        table = markdown(results, len(a), len(b), flagged)
        if args.markdown:
            with open(args.markdown, "w") as f:
                f.write(table)
        print(table)
        return 0
    except (OSError, ValueError) as e:
        print(f"compare: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
