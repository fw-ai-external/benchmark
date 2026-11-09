from __future__ import annotations

import argparse
from collections.abc import Iterator
from types import SimpleNamespace

import gevent
import pytest
from locust.argument_parser import get_parser
from locust.exception import StopUser
from locust.stats import RequestStats

import load_test as bench
from load_test import CohortTrace, DecodeCohorts, FireworksProvider, cohort_window, text_summary_entries


def options(*flags: str) -> argparse.Namespace:
    return get_parser().parse_args(
        [
            "--headless",
            "--provider",
            "fireworks",
            "--model",
            "test",
            "--no-chat",
            "-u",
            "2",
            "--cohorts",
            "2",
            "--max-tokens",
            "10",
            *flags,
        ]
    )


def trace(request_id: str, offset: float = 0) -> CohortTrace:
    return CohortTrace(request_id, [(offset + i, i) for i in range(1, 11)])


def test_grouped_token_window_uses_exact_open_left_counts() -> None:
    # Receipt times are for whole groups; never interpolate tokens within a group.
    a = CohortTrace("a", [(1, 2), (3, 5), (5, 8), (7, 10)])
    b = CohortTrace("b", [(2, 1), (4, 4), (6, 10)])
    result = cohort_window([a, b], minimum_fraction=0.6)
    assert (result["start"], result["end"]) == (2, 6)
    assert result["requests"] == [
        {"id": "a", "n_start": 2, "n_end": 8, "total": 10},
        {"id": "b", "n_start": 1, "n_end": 10, "total": 10},
    ]
    assert result["cohort_window_tokens"] == 15
    assert result["cohort_decode_lpt"] == pytest.approx(1000 * 2 * 4 / 15)


@pytest.mark.parametrize("moonshot", [False, True])
def test_ids_include_empty_text_and_ignore_usage_only_frames(moonshot: bool) -> None:
    t = CohortTrace()
    choice = (
        {"internal_content": [{"token_ids": [12, 13]}]}
        if moonshot
        else {"delta": {"content": "", "internal_content": {"token_ids": [12, 13]}}}
    )
    t.observe({"id": "r", "choices": [choice]}, timestamp=1)
    t.observe({"id": "r", "choices": [], "usage": {"completion_tokens": 2}}, timestamp=100)
    assert t.progress == [(1, 2)]  # EOF/usage latency is excluded.
    t.validate(completion_tokens=2, expected=2)
    with pytest.raises(ValueError, match="total"):
        t.validate(completion_tokens=3, expected=3)
    with pytest.raises(ValueError, match="total"):
        t.validate(completion_tokens=2.0, expected=2)


def test_missing_ids_and_insufficient_overlap_are_rejected() -> None:
    t = CohortTrace()
    t.observe({"id": "r", "choices": [{"text": "ten tokens"}], "usage": {"completion_tokens": 10}}, 1)
    with pytest.raises(ValueError, match="token IDs"):
        t.validate(completion_tokens=10, expected=10)
    with pytest.raises(ValueError, match="coverage"):
        cohort_window([trace("a"), trace("b", offset=7)], minimum_fraction=0.85)
    with pytest.raises(ValueError, match="positive common"):
        cohort_window([trace("a"), trace("b", offset=10)], minimum_fraction=0.01)
    with pytest.raises(ValueError, match="duplicate"):
        cohort_window([trace("a"), trace("a")], minimum_fraction=0.85)


@pytest.mark.parametrize(
    "flags",
    [
        ("--qps", "1"),
        ("--worker",),
        ("--no-stream",),
        ("--cancel-rate", ".5"),
        ("-u", "0"),
        ("--cohort-timeout", "nan"),
    ],
)
def test_incompatible_modes_fail_preflight(flags: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        bench.validate_cohort_options(options(*flags))


def test_payload_and_summary_opt_in_leave_legacy_metric_unchanged() -> None:
    off = options("--cohorts", "0")
    on = options()
    assert (
        "include_internal_content"
        not in FireworksProvider("test", off).format_payload("p", 10, None)["stream_options"]
    )
    assert (
        FireworksProvider("test", on).format_payload("p", 10, None)["stream_options"]["include_internal_content"]
        is True
    )
    stats = RequestStats()
    stats.log_request("METRIC", "latency_per_token", 20, 0)
    entries, _ = text_summary_entries(stats, stream=True)
    assert "cohort_decode_lpt" not in entries
    stats.log_request("METRIC", "cohort_decode_lpt", 7, 0)
    stats.log_request("METRIC", "cohort_decode_lpt", 9, 0)
    entries, percentiles = text_summary_entries(stats, stream=True)
    assert entries["latency_per_token"] == 20
    assert entries["cohort_decode_lpt"] == 8
    assert "cohort_decode_lpt" in percentiles


def test_barriers_drain_all_requests_and_exclude_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    samples, quits, starts, ends = [], [], [], []
    env = SimpleNamespace(parsed_options=options(), runner=SimpleNamespace(quit=lambda: quits.append(True)))
    run = DecodeCohorts(env)
    monkeypatch.setattr(bench, "add_custom_metric", lambda name, value: samples.append((name, value)))

    def user(index: int) -> None:
        for wave in range(3):
            run.begin()
            starts.append((wave, index, len(ends)))
            if index == 1:
                gevent.sleep(0.005)
            ends.append((wave, index))
            run.finish(trace(f"{wave}-{index}"))

    jobs = [gevent.spawn(user, index) for index in range(2)]
    gevent.joinall(jobs, timeout=1, raise_error=True)
    assert all(job.ready() for job in jobs)
    gevent.sleep(0)
    assert len(quits) == 1 and run.measured == 2 and run.completed == 3
    assert len([s for s in samples if s[0] == "cohort_decode_lpt"]) == 2
    assert len(starts) == 6 and all(drained >= wave * 2 for wave, _, drained in starts)
    with pytest.raises(StopUser):
        run.begin()
    assert not run.error


def test_failure_unblocks_peers_and_stops_without_a_metric() -> None:
    env = SimpleNamespace(parsed_options=options(), runner=SimpleNamespace(quit=lambda: None))
    run = DecodeCohorts(env)
    a = gevent.spawn(run.begin)
    gevent.sleep(0)
    with pytest.raises(StopUser):
        run.finish(None)
    gevent.joinall([a], timeout=1)
    assert a.ready() and isinstance(a.exception, StopUser)
    assert run.error and run.measured == 0


@pytest.mark.parametrize("problem", [None, "missing_ids", "early_eos", "missing_done", "transport"])
def test_streaming_request_integration(monkeypatch: pytest.MonkeyPatch, problem: str | None) -> None:
    opts = options()
    user = bench.LLMUser.__new__(bench.LLMUser)
    user.environment = SimpleNamespace(parsed_options=opts)
    user.provider_formatter = FireworksProvider("test", opts)
    user.stream, user.lora_adapters, user.first_done = True, (), True
    user.prompt_tokenizer_tokens, user._designed_cached_tokens = 1, None
    payload = user.provider_formatter.format_payload("prompt", 10, None)
    failures = []

    class Response:
        status_code, headers = 200, {}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def failure(self, error: object) -> None:
            failures.append(str(error))

        def success(self) -> None:
            return None

        def iter_lines(self, **kwargs: object) -> Iterator[bytes]:
            for _ in range(4 if problem == "early_eos" else 5):
                choice = {"text": "x", "internal_content": [{"token_ids": [10, 11]}]}
                if problem == "missing_ids":
                    choice.pop("internal_content")
                yield b"data: " + bench.orjson.dumps({"id": "r", "choices": [choice]})
            if problem == "transport":
                raise bench.requests.exceptions.ConnectionError("interrupted stream")
            yield b"data: " + bench.orjson.dumps({"id": "r", "choices": [], "usage": {"completion_tokens": 10}})
            if problem != "missing_done":
                yield b"data: [DONE]"

    user.client = SimpleNamespace(post=lambda *args, **kwargs: Response())
    monkeypatch.setattr(bench, "add_custom_metric", lambda *args, **kwargs: None)
    result = user._do_generate_text(prepared=(payload, {}, 10, 1, None, "{}"), cohort_trace=CohortTrace())
    if problem is None:
        assert result.request_id == "r" and result.progress[-1][1] == 10 and not failures
    else:
        assert result is None and len(failures) == 1
