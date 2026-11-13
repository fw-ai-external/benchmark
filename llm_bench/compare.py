"""Compare two sets of llm_bench summaries (A = without a change, B = with it) from their --summary-file CSVs.

    python -m llm_bench.compare --a A1.csv [A2.csv ...] --b B1.csv [B2.csv ...] [--min-noise 0.03] \\
        [--min-repeats 3] [--json out.json] [--markdown out.md]

The runs are paired: run i uses its own request seed s_i on both sides, A_i warms up to the hit-rate target (and prints
its N_i), B_i is given --request-seed s_i --warmup-requests N_i, and the runs go A1, B1, A2, B2 ... A repeat with
the same seed would resend the first run's exact prompts into the cache the first run left and read as a better hit
rate and TTFT. The rows of --a and of --b (every row is a run, files read in order) are paired in order. Within a pair
the request seed and the warm-up requests must match; across all the runs the workload files' sha256, the options
that change the traffic (replayed pod, retiming, arrival process, -t, corpus, tokenizer, output mode, chat mode ...),
the target rate and the llm_bench version must match; and the seeds must differ from pair to pair. Otherwise this
refuses (exit 2) and says which differ. Rows that are not measured runs (a --prefix-fill step) are ignored, and said so.

Else each metric's verdict comes from the paired relative deltas (A_i vs B_i): their mean against a noise floor of
max(--min-noise, the spread (max - min) of the per-pair deltas). Inside the floor it reads "same", unless the floor is
over twice --min-noise, when the pairs are too noisy to say ("inconclusive (noisy pairs)"). --min-repeats (3) counts
pairs; with fewer every verdict reads "inconclusive (too few repeats)", the deltas still reported. A metric that is 0
on A and not on B reads "different". Each metric's floor and where it came from, and every pair's delta, are reported.
A run that failed its delivery check or its --max-fail-ratio is flagged in the output (it is still in the figures).
Exit 0 compared, 2 refused, 1 error.

For a trustworthy comparison run at least 3 pairs, alternating A and B, not alongside other load; P99 is the noisiest
figure.
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


def _option_differences(a: dict[str, str], b: dict[str, str], ignore: tuple = ()) -> str:
    """Which workload options differ, from the summaries' Workload Options JSON (empty when it isn't there)."""
    try:
        va, vb = json.loads(a.get("Workload Options", "")), json.loads(b.get("Workload Options", ""))
    except ValueError:
        return ""
    return ", ".join(
        f"{k}: A {va.get(k)!r} vs B {vb.get(k)!r}"
        for k in sorted(set(va) | set(vb))
        if va.get(k) != vb.get(k) and k not in ignore
    )


# Identity that must match across every run of both sides, and the columns that must match within a pair
SHARED_IDENTITY = [i for i in IDENTITY if i[0] not in ("Request Seed", "Warmup Requests")]
PAIR_IDENTITY = [i for i in IDENTITY if i[0] in ("Request Seed", "Warmup Requests")]


def identity_differences(a: list[dict[str, str]], b: list[dict[str, str]]) -> list[str]:
    """Why the runs can't be compared, empty when they can: unequal counts (they are paired in order), a pair whose
    seed or warm-up requests differ, a seed repeated across pairs, or a workload identity that differs anywhere."""
    if len(a) != len(b):
        return [f"A has {len(a)} run(s) and B {len(b)}: runs are paired in order (A_i with B_i)"]
    problems = []
    for column, label in PAIR_IDENTITY:
        for i, (ra, rb) in enumerate(zip(a, b), 1):
            if not _same(ra.get(column, ""), rb.get(column, "")):
                problems.append(
                    f"pair {i}: {label}: A {ra.get(column) or '(none)'} vs B {rb.get(column) or '(none)'}"
                    + (" (give B_i the --warmup-requests N_i that A_i printed)" if column == "Warmup Requests" else "")
                )
    seeds = [r.get("Request Seed", "") for r in a]
    for seed in sorted({s for s in seeds if seeds.count(s) > 1}):
        runs = ", ".join(str(i) for i, s in enumerate(seeds, 1) if s == seed)
        problems.append(
            f"request seed {seed or '(none)'} is the same in pairs {runs}: a repeat with the same seed resends the "
            f"first run's prompts into the cache it left; give each pair its own --request-seed"
        )
    runs = a + b
    for column, label in SHARED_IDENTITY:
        found = [r.get(column, "") for r in runs]
        if not all(_same(found[0], v) for v in found):
            detail = ""
            if column == "Workload Options Sha256":
                other = next(r for r in runs if not _same(r.get(column, ""), found[0]))
                detail = _option_differences(runs[0], other, ("request_seed",))
            problems.append(f"{label} differs between runs: {sorted(set(found))}" + (f" ({detail})" if detail else ""))
        elif not found[0] and column in ("Workload Sha256", "Workload Options Sha256", "Llm Bench Version"):
            problems.append(f"{label} is missing from the summaries: not written by a warm-up-aware llm_bench")
    return problems


def _number(row: dict[str, str], column: str) -> Optional[float]:
    try:
        value = float(row.get(column, ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


MIN_REPEATS = 3  # a verdict needs this many pairs


def compare(
    a: list[dict[str, str]], b: list[dict[str, str]], min_noise: float, min_repeats: int = MIN_REPEATS
) -> list[dict[str, Any]]:
    """Per-metric verdicts from the pairs (a[i], b[i]); the rows are in pair order and as many on each side."""
    results = []
    for column, higher_is_better, kind in METRICS:
        pairs = [(_number(ra, column), _number(rb, column)) for ra, rb in zip(a, b)]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        if kind == "rel" and any(x == 0 and y != 0 for x, y in pairs):
            kind = "abs"  # a relative change from 0 means nothing: the difference
            absolute = True
        else:
            absolute = False
        deltas = [(y - x) if kind == "abs" else ((y - x) / abs(x) if x else 0.0) for x, y in pairs]
        delta = sum(deltas) / len(deltas)
        spread = max(deltas) - min(deltas) if len(deltas) > 1 else 0.0
        noise = 0.0 if absolute else max(spread, min_noise)
        if absolute:
            verdict = "different" if delta else "same"
        elif abs(delta) <= noise:
            verdict = "same" if noise <= 2 * min_noise else "inconclusive (noisy pairs)"
        else:
            verdict = "better" if (delta > 0) == higher_is_better else "worse"
        if len(pairs) < min_repeats:
            # Too few pairs can't tell a change from run-to-run noise, and "same" and "different" are claims too
            verdict = "inconclusive (too few repeats)"
        results.append(
            {
                "metric": column,
                "a": sum(x for x, _ in pairs) / len(pairs),
                "b": sum(y for _, y in pairs) / len(pairs),
                "delta": delta,
                "delta_kind": kind,
                "pair_deltas": deltas,
                "noise": noise,
                "noise_from": "pair spread" if spread > min_noise and not absolute else "--min-noise",
                "verdict": verdict,
                "pairs": len(pairs),
            }
        )
    return results


def _format_delta(result: dict[str, Any], delta: Optional[float] = None) -> str:
    delta = result["delta"] if delta is None else delta
    return f"{delta:+.4f}" if result["delta_kind"] == "abs" else f"{100 * delta:+.1f}%"


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
        f"A: {a_runs} run(s), B: {b_runs} run(s), paired in order",
        "",
        "| metric | A | B | mean delta | noise floor (from) | pairs | per-pair deltas | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        noise = f"{r['noise']:.4f}" if r["delta_kind"] == "abs" else f"{100 * r['noise']:.1f}%"
        per_pair = ", ".join(_format_delta(r, d) for d in r["pair_deltas"])
        lines.append(
            f"| {r['metric']} | {r['a']:.4g} | {r['b']:.4g} | {_format_delta(r)} | {noise} ({r['noise_from']}) "
            f"| {r['pairs']} | {per_pair} | {r['verdict']} |"
        )
    return "\n".join(lines + [""] + _flag_lines(flagged or [])) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True, metavar="SUMMARY", help="A's summary CSV(s).")
    parser.add_argument("--b", nargs="+", required=True, metavar="SUMMARY", help="B's summary CSV(s).")
    parser.add_argument("--min-noise", type=float, default=0.03, help="Least noise floor (default 0.03).")
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=MIN_REPEATS,
        help=f"Pairs needed for a verdict (default {MIN_REPEATS}); fewer reads inconclusive.",
    )
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
        results = compare(a, b, args.min_noise, args.min_repeats)
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
