"""Compare two sets of llm_bench summaries (A = without a change, B = with it) from their --summary-file CSVs.

    python -m llm_bench.compare --a A1.csv [A2.csv ...] --b B1.csv [B2.csv ...] [--min-noise 0.03] \\
        [--min-repeats 3] [--restarted-between-runs] [--json out.json] [--markdown out.md]
    python -m llm_bench.compare --recipe A1.csv [A2.csv ...] [--json recipe.json]

--recipe prints, for a baseline's runs, each pair's `--request-seed s_i --warmup-requests N_i` (what every later
variant run must use to be paired with it) and the workload identity it must match; it refuses a baseline with a
repeated or missing seed or count or with runs of different workloads. --restarted-between-runs turns the
same-endpoint warning into a note (below).

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
Failures are judged exactly, pair by pair, with no noise floor: any increase (or decrease) is reported. A run that
failed its delivery check or its --max-fail-ratio makes compare exit 3 and name the run; --allow-flagged compares
anyway (the run is flagged in the output and is still in the figures).
A pair whose baseline is 0 reads "different" for that metric if B is not 0. Identity (seeds, counts, shas, versions,
option hashes) is compared as exact strings, only the target rate with a tolerance.
A percentile's floor is at least one bin of the response-time rounding Locust applies (1 ms below 100 ms, 10 below
1,000, 100 below 10,000, 1,000 above) over the value: 1,500 vs 1,400 ms is one bin, not a change. If A and B were sent
to the same host (the summaries' Target Host) it warns in every output: B's requests are A's, and they hit the cache
A filled unless the server was restarted between runs, so it is not a valid A/A. Exit 0 compared, 2 refused, 3 flagged runs (without --allow-flagged), 1 error.

For a trustworthy comparison run at least 3 pairs, alternating A and B, not alongside other load; P99 is the noisiest
figure.
"""

import argparse
import csv
import json
import math
import re
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


def _same(a: str, b: str, column: str = "") -> bool:
    """Whether two identity values agree: exactly, as strings (a seed, a count, a sha, a version, an options hash:
    isclose would call seeds 3000000000 and 3000000002 equal). Only the target rate, a float, takes a tolerance."""
    if column == "Target Qps":
        try:
            return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except ValueError:
            pass
    return a.strip() == b.strip()


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
            if not _same(ra.get(column, ""), rb.get(column, ""), column):
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
        if not all(_same(found[0], v, column) for v in found):
            detail = ""
            if column == "Workload Options Sha256":
                other = next(r for r in runs if not _same(r.get(column, ""), found[0], column))
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


def bin_width_ms(value_ms: float) -> float:
    """The width of the bin Locust stores a response time of `value_ms` in: it keeps response times rounded, exact below
    100 ms, to 10 below 1,000, to 100 below 10,000 and to 1,000 above (stats.py, StatsEntry._log_response_time), so a
    percentile is only known to a bin."""
    if value_ms < 100:
        return 1.0
    if value_ms < 1000:
        return 10.0
    return 100.0 if value_ms < 10000 else 1000.0


def _is_percentile(column: str) -> bool:
    return bool(re.match(r"P\d", column))


FAILURE_METRICS = ("Fail Ratio", "Requests Failed")  # one rule: any increase is reported, no noise floor


def _failure_result(column: str, pairs: list[tuple[float, float]], kind: str) -> dict[str, Any]:
    """Failures are not judged against a noise floor: a run that failed more requests did, whatever --min-noise says.
    Each pair is judged on its own (a pair at 0 is no special case); B up in some pairs and down in others is mixed."""
    deltas = [y - x for x, y in pairs]
    up, down = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    if up and down:
        verdict = f"mixed (failures up in {up} pair(s), down in {down})"
    else:
        verdict = "worse" if up else "better" if down else "same"
    return {
        "metric": column,
        "a": sum(x for x, _ in pairs) / len(pairs),
        "b": sum(y for _, y in pairs) / len(pairs),
        "delta": sum(deltas) / len(deltas),
        "delta_kind": "abs",
        "pair_deltas": deltas,
        "noise": 0.0,
        "noise_from": "none (failures are exact)",
        "verdict": verdict,
        "pairs": len(pairs),
    }


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
        if column in FAILURE_METRICS:
            results.append(_failure_result(column, pairs, kind))
            continue
        # A pair whose baseline is 0 has no relative change: it is "different" if B is not 0, and the others are
        # judged as usual (it is not a reason to judge the whole metric by a raw difference)
        from_zero = [i for i, (x, y) in enumerate(pairs) if kind == "rel" and x == 0 and y != 0]
        deltas: list[Optional[float]] = [
            (y - x) if kind == "abs" else (None if (x == 0 and y != 0) else ((y - x) / abs(x) if x else 0.0))
            for x, y in pairs
        ]
        valid = [d for d in deltas if d is not None]
        delta = sum(valid) / len(valid) if valid else None
        spread = max(valid) - min(valid) if len(valid) > 1 else 0.0
        noise = max(spread, min_noise)
        # A percentile is only known to the bin Locust rounded its response times into (1,500 vs 1,400 ms is one bin),
        # so its floor is at least a bin's width over the value
        statistical, bin_floor, bin_ms = noise, 0.0, 0.0
        mean_a, mean_b = sum(x for x, _ in pairs) / len(pairs), sum(y for _, y in pairs) / len(pairs)
        if _is_percentile(column):
            bin_ms = max(bin_width_ms(mean_a), bin_width_ms(mean_b))
            bin_floor = bin_ms / max(min(mean_a, mean_b), 1.0)
            noise = max(noise, bin_floor)
        if from_zero:
            verdict = "different"
        elif abs(delta) <= noise:
            # (the bin floor is quantization, not noise: it widens "same" but never makes the pairs "noisy")
            verdict = "same" if statistical <= 2 * min_noise else "inconclusive (noisy pairs)"
        else:
            verdict = "better" if (delta > 0) == higher_is_better else "worse"
        if len(pairs) < min_repeats:
            # Too few pairs can't tell a change from run-to-run noise, and "same" and "different" are claims too
            verdict = "inconclusive (too few repeats)"
        results.append(
            {
                "metric": column,
                "a": mean_a,
                "b": mean_b,
                "delta": delta,
                "delta_kind": kind,
                "pair_deltas": deltas,
                "noise": noise,
                "noise_from": (
                    f"percentile bin {bin_ms:g} ms"
                    if bin_floor and bin_floor > statistical
                    else "pair spread" if spread > min_noise else "--min-noise"
                ),
                "verdict": verdict,
                "pairs": len(pairs),
                "from_zero_pairs": [i + 1 for i in from_zero],
            }
        )
    return results


def _format_delta(result: dict[str, Any], delta: Optional[float] = None, pair: bool = False) -> str:
    delta = delta if pair else result["delta"]
    if delta is None:
        return "n/a (A=0)"
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


EXIT_FLAGGED = 3  # runs that failed their delivery check or --max-fail-ratio, without --allow-flagged

SAME_ENDPOINT_WARNING = (
    "same endpoint: B reuses A's cache unless the server was restarted between runs; not a valid A/A"
)


def _host(row: dict[str, str]) -> str:
    """The run's target host, normalized (scheme and case ignored, no trailing slash); empty when not recorded."""
    host = row.get("Target Host", "").strip().lower()
    return re.sub(r"^https?://", "", host).rstrip("/")


SAME_ENDPOINT_NOTE = "same endpoint; the operator states the server was restarted before every run"


def warnings_for(
    a: list[dict[str, str]], b: list[dict[str, str]], restarted: bool = False
) -> tuple[list[str], list[str]]:
    """(warnings, notes). Both sides sent to one host: B_i's requests (A_i's exact ones, by design) hit the cache A_i
    filled, a warning, unless the operator states (--restarted-between-runs) the server was restarted before every
    run, which makes it a note."""
    shared = sorted(({_host(r) for r in a} & {_host(r) for r in b}) - {""})
    if not shared:
        return [], []
    where = f" ({', '.join(shared)})"
    return ([], [SAME_ENDPOINT_NOTE + where]) if restarted else ([SAME_ENDPOINT_WARNING + where], [])


def recipe(rows: list[dict[str, str]]) -> dict[str, Any]:
    """What a later variant run must use to be paired with these baseline runs: per pair its `--request-seed s_i
    --warmup-requests N_i`, and the workload identity it must match. Raises ValueError when the rows are not a valid
    baseline (a seed or warm-up count missing, a repeated seed, runs of different workloads)."""
    problems = []
    for i, row in enumerate(rows, 1):
        for column, label in PAIR_IDENTITY:
            if not row.get(column, ""):
                problems.append(f"run {i}: {label} is missing from its summary")
    problems += identity_differences(rows, rows)
    if problems:
        raise ValueError("not a valid baseline:\n  - " + "\n  - ".join(dict.fromkeys(problems)))
    pairs = [
        {
            "pair": i,
            "request_seed": row["Request Seed"],
            "warmup_requests": row["Warmup Requests"],
            "flags": f"--request-seed {row['Request Seed']} --warmup-requests {row['Warmup Requests']}",
        }
        for i, row in enumerate(rows, 1)
    ]
    return {"pairs": pairs, "identity": {column: rows[0].get(column, "") for column, _ in SHARED_IDENTITY}}


def recipe_text(result: dict[str, Any]) -> str:
    identity = result["identity"]
    lines = [f"pair {p['pair']}: {p['flags']}" for p in result["pairs"]]
    lines.append(
        "must match: " + ", ".join(f"{label} {identity[column] or '(none)'}" for column, label in SHARED_IDENTITY)
    )
    return "\n".join(lines)


def _flag_lines(flagged: list[dict[str, Any]]) -> list[str]:
    return [f"!!! {f['side']} run {f['run']}: {'; '.join(f['reasons'])} (it is in the figures above)" for f in flagged]


def markdown(
    results: list[dict[str, Any]],
    a_runs: int,
    b_runs: int,
    flagged: Optional[list] = None,
    warnings: Optional[list] = None,
    notes: Optional[list] = None,
) -> str:
    lines = (
        [f"!!! WARNING: {w}" for w in warnings or []]
        + [f"NOTE: {n}" for n in notes or []]
        + [
            f"A: {a_runs} run(s), B: {b_runs} run(s), paired in order",
            "",
            "| metric | A | B | mean delta | noise floor (from) | pairs | per-pair deltas | verdict |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for r in results:
        noise = f"{r['noise']:.4f}" if r["delta_kind"] == "abs" else f"{100 * r['noise']:.1f}%"
        per_pair = ", ".join(_format_delta(r, d, pair=True) for d in r["pair_deltas"])
        lines.append(
            f"| {r['metric']} | {r['a']:.4g} | {r['b']:.4g} | {_format_delta(r)} | {noise} ({r['noise_from']}) "
            f"| {r['pairs']} | {per_pair} | {r['verdict']} |"
        )
    return "\n".join(lines + [""] + _flag_lines(flagged or [])) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", metavar="SUMMARY", help="A's summary CSV(s).")
    parser.add_argument("--b", nargs="+", metavar="SUMMARY", help="B's summary CSV(s).")
    parser.add_argument(
        "--recipe",
        nargs="+",
        metavar="SUMMARY",
        help="Instead of comparing: from a baseline's summary CSV(s), print for each pair the flags a later variant "
        "run must use (--request-seed s_i --warmup-requests N_i) and the workload identity it must match; refuses a "
        "file that is not a valid baseline. With --json, machine-readable.",
    )
    parser.add_argument(
        "--allow-flagged",
        action="store_true",
        help="Compare even when runs failed their delivery check or --max-fail-ratio (they are flagged in the "
        "output); without it such runs make compare exit 3 and name them.",
    )
    parser.add_argument(
        "--restarted-between-runs",
        action="store_true",
        help="You state the server was restarted before every run: A and B sharing a host is then a note, not a "
        "warning.",
    )
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
    if args.recipe is None and not (args.a and args.b):
        parser.error("give --a and --b to compare, or --recipe for a baseline's flags")
    if args.recipe is not None and (args.a or args.b):
        parser.error("--recipe goes alone")
    try:
        ignored: list[str] = []
        if args.recipe is not None:
            rows = read_rows(args.recipe, ignored)
            for note in ignored:
                print(f"compare: ignoring {note} (not a measured run)", file=sys.stderr)
            try:
                result = recipe(rows)
            except ValueError as e:
                print(f"compare: {e}", file=sys.stderr)
                return 2
            if args.json:
                with open(args.json, "w") as f:
                    json.dump(result, f, indent=2)
            print(recipe_text(result))
            return 0
        a, b = read_rows(args.a, ignored), read_rows(args.b, ignored)
        for note in ignored:
            print(f"compare: ignoring {note} (not a measured run)", file=sys.stderr)
        problems = identity_differences(a, b)
        if problems:
            print("Refusing to compare: the two sides differ in their workload identity:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2
        flagged = flagged_runs("A", a) + flagged_runs("B", b)
        if flagged and not args.allow_flagged:
            print(
                "compare: not comparing: these runs failed their delivery check or --max-fail-ratio, so their "
                "figures are not a clean data point (--allow-flagged compares anyway):",
                file=sys.stderr,
            )
            for f in flagged:
                print(f"  - {f['side']} run {f['run']}: {'; '.join(f['reasons'])}", file=sys.stderr)
            if args.json:
                with open(args.json, "w") as out:
                    json.dump({"compared": False, "flagged_runs": flagged, "ignored": ignored}, out, indent=2)
            return EXIT_FLAGGED
        results = compare(a, b, args.min_noise, args.min_repeats)
        warnings, notes = warnings_for(a, b, args.restarted_between_runs)
        for warning in warnings:
            print(f"compare: !!! WARNING: {warning}", file=sys.stderr)
        for note in notes:
            print(f"compare: NOTE: {note}", file=sys.stderr)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(
                    {
                        "a_runs": len(a),
                        "b_runs": len(b),
                        "metrics": results,
                        "flagged_runs": flagged,
                        "warnings": warnings,
                        "notes": notes,
                        "ignored": ignored,
                    },
                    f,
                    indent=2,
                )
        table = markdown(results, len(a), len(b), flagged, warnings, notes)
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
