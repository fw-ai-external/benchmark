"""Representative warm-up: the shape-mix and replay warm-ups against the mock server (CPU, localhost), the A/B
guarantees (same warm-up, same measured requests), plan resolution, the delivery check and llm_bench.compare."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import compare
import load_test
from load_test import (
    HitRateWindow,
    LeadIns,
    MeasuredReuse,
    ReplayPacer,
    SendTimeline,
    UnscoredSender,
    Warmup,
    WarmupPlan,
    delivery_entries,
    load_replay_schedule,
    parse_shape_mix,
    run_replay_warmup,
    run_shape_mix_warmup,
    warmup_plan,
)
from mock_server import PrefixCacheEngine, make_handler
from test_shape_mix import LIMERICKS, MODEL, WordTokenizer, _common_prefix, _dataset, _options

# One class, one pool prefix: after the first request every prompt hits the shared 200 of 300 tokens
MIX = {
    "seed": 11,
    "classes": [
        {
            "name": "chat",
            "weight": 1.0,
            "prompt_tokens": 300,
            "cached_prompt_tokens": 200,
            "generated_tokens": 4,
            "prefix_pool_size": 1,
        }
    ],
}
REPLAY_MIX = {
    "seed": 5,
    "classes": [
        {
            "name": "chat",
            "weight": 1.0,
            "prompt_tokens": 900,
            "cached_prompt_tokens": 400,
            "generated_tokens": 4,
            "prefix_pool_size": 2,
        }
    ],
}


@pytest.fixture
def mock_server(tmp_path):
    """A mock engine (prefix cache over WordTokenizer ids) on a free port; yields (host, request log path)."""
    started = []

    def start():
        log_path = tmp_path / f"requests{len(started)}.log"
        log = open(log_path, "a")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PrefixCacheEngine(WordTokenizer()), 0.0, 0.0, log))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return f"http://127.0.0.1:{server.server_port}", log_path

    yield start
    for server in started:
        server.shutdown()


def _run_options(mix: dict, **overrides) -> argparse.Namespace:
    values = dict(
        provider="openai",
        model=MODEL,
        api_key=None,
        header=None,
        shape_mix=json.dumps(mix),
        request_seed=None,
        qps=40.0,
        qps_distribution="constant",
        qps_cv2=None,
        qps_modulation_window=None,
        qps_dispersion=None,
        shape_mix_output_mode="exact",
        request_timeout=10.0,
        warmup_hit_rate=None,
        warmup_requests=None,
        warmup_tolerance=0.0,
        warmup_window_s=0.3,
        warmup_cap_s=None,
        no_warmup=False,
        prefix_fill=False,
        prefix_fill_concurrency=4,
        replay_schedule=None,
    )
    values.update(overrides)
    return _options(False, **values)


@pytest.fixture(autouse=True)
def _reset_state():
    Warmup.mode, Warmup.reason, Warmup.requests, Warmup.prompt_tokens = "none", "", 0, 0
    Warmup.hit_rate = Warmup.target = None
    for name in ("rows", "sent", "failed", "skipped", "dropped"):
        setattr(LeadIns, name, 0)
    MeasuredReuse.reset()
    yield


@pytest.fixture
def sent_log(monkeypatch):
    """Every request the unscored sender posts, in order, as (schedule row, one_token, prompt); the posts go through."""
    log = []
    post = UnscoredSender.post

    def recording(self, request, one_token=False):
        log.append((request.schedule_row, one_token, request.prompt, time.monotonic()))
        return post(self, request, one_token)

    monkeypatch.setattr(UnscoredSender, "post", recording)
    return log


def _shape_mix_run(host: str, plan: WarmupPlan, seed_options=None):
    options = _run_options(MIX, **(seed_options or {}))
    ds = _dataset(MIX, tokenizer=WordTokenizer())
    ds.build_prefixes()
    run_shape_mix_warmup(SimpleNamespace(host=host), options, ds, plan)
    return ds


def test_a_target_run_reports_its_count_and_b_with_that_count_sends_the_same_warmup(mock_server, sent_log) -> None:
    host_a, log_a = mock_server()
    plan_a = WarmupPlan("target", hit_rate=0.5, tolerance=0.0, window_s=0.3, cap_s=30.0, source="cli")
    ds_a = _shape_mix_run(host_a, plan_a)
    n = Warmup.requests
    assert Warmup.mode == "target" and Warmup.reason == "target" and n >= 10  # 40 QPS for at least 0.3 s
    assert Warmup.hit_rate >= 0.5 and Warmup.target == 0.5 and Warmup.prompt_tokens > 0
    assert len(log_a.read_text().splitlines()) == n  # the server saw exactly the reported number
    warm_a = [prompt for _, _, prompt, _ in sent_log]
    measured_a = [next(ds_a).prompt for _ in range(8)]
    sent_log.clear()

    host_b, log_b = mock_server()
    ds_b = _shape_mix_run(host_b, WarmupPlan("requests", requests=n, window_s=0.3, source="cli"))
    assert Warmup.mode == "requests" and Warmup.reason == "requests" and Warmup.requests == n
    warm_b = [prompt for _, _, prompt, _ in sent_log]
    assert warm_b == warm_a and len(log_b.read_text().splitlines()) == n  # identical warm-up requests
    assert [next(ds_b).prompt for _ in range(8)] == measured_a  # and identical measured requests


def test_measured_requests_do_not_depend_on_the_warmup_length(mock_server, sent_log) -> None:
    streams = {}
    for count in (0, 3, 9):
        host, _ = mock_server()
        ds = _shape_mix_run(host, WarmupPlan("requests", requests=count, window_s=0.3))
        streams[count] = [next(ds).prompt for _ in range(6)]
    plain = _dataset(MIX, tokenizer=WordTokenizer())
    plain.build_prefixes()
    assert streams[0] == streams[3] == streams[9] == [next(plain).prompt for _ in range(6)]
    # the warm-up's own stream is a different stream: its requests aren't the measured ones
    warm = _dataset(MIX, tokenizer=WordTokenizer())
    warm.build_prefixes()
    assert [warm.warmup_request().prompt for _ in range(6)] != streams[0]


def test_a_longer_warmup_extends_the_shorter_one(mock_server, sent_log) -> None:
    prompts = {}
    for count in (3, 7):
        host, _ = mock_server()
        _shape_mix_run(host, WarmupPlan("requests", requests=count, window_s=0.3))
        prompts[count] = [p for _, _, p, _ in sent_log]
        sent_log.clear()
    assert prompts[7][:3] == prompts[3] and len(prompts[7]) == 7


def test_a_target_run_stops_at_the_cap_when_the_target_is_out_of_reach(mock_server, sent_log, caplog) -> None:
    host, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.999, tolerance=0.0, window_s=0.2, cap_s=0.7, source="cli")
    with caplog.at_level("WARNING", logger="load_test"):
        _shape_mix_run(host, plan)
    assert Warmup.reason == "cap" and Warmup.requests >= 10
    assert 0.7 <= Warmup.duration_s < 3.0
    assert "Warm-up stopped on cap" in caplog.text  # a warning, not a failure


def test_warmup_flags_without_qps_or_shape_mix_are_refused_and_no_flags_is_no_warmup() -> None:
    assert warmup_plan(_run_options(MIX, qps=None)) == WarmupPlan()  # nothing asked, nothing warmed up
    assert warmup_plan(argparse.Namespace()) == WarmupPlan()
    with pytest.raises(ValueError, match="need --qps"):
        warmup_plan(_run_options(MIX, qps=None, warmup_requests=5))
    with pytest.raises(ValueError, match="need --shape-mix"):
        warmup_plan(argparse.Namespace(warmup_requests=5))
    with pytest.raises(ValueError, match="mutually exclusive"):
        warmup_plan(_run_options(MIX, warmup_requests=5, warmup_hit_rate=0.5, warmup_cap_s=3))
    with pytest.raises(ValueError, match="--warmup-cap-s"):
        warmup_plan(_run_options(MIX, warmup_hit_rate=0.5))
    with pytest.raises(ValueError, match="prefix-fill"):
        warmup_plan(_run_options(MIX, warmup_requests=5, prefix_fill=True))
    plan = warmup_plan(_run_options(MIX, warmup_hit_rate=0.5, warmup_cap_s=60.0, warmup_tolerance=0.02))
    assert (plan.mode, plan.hit_rate, plan.cap_s, plan.tolerance) == ("target", 0.5, 60.0, 0.02)


def test_the_workload_files_warmup_block_is_the_default_and_the_cli_overrides_it() -> None:
    mix = dict(MIX, warmup={"hit_rate": 0.62, "source": "prod 2026-11-01..07, replica r3"})
    plan = warmup_plan(_run_options(mix, warmup_cap_s=90.0))
    assert (plan.mode, plan.hit_rate, plan.source) == ("target", 0.62, "prod 2026-11-01..07, replica r3")
    with pytest.raises(ValueError, match="--warmup-cap-s"):
        warmup_plan(_run_options(mix))  # a target needs its cap
    assert warmup_plan(_run_options(mix, no_warmup=True)) == WarmupPlan()
    assert warmup_plan(_run_options(mix, warmup_hit_rate=0.4, warmup_cap_s=9.0)).hit_rate == 0.4
    assert warmup_plan(_run_options(mix, warmup_requests=7)).mode == "requests"
    # the block is for a run at a rate: without --qps it is left alone, not an error (old command lines keep working)
    assert warmup_plan(_run_options(mix, qps=None)) == WarmupPlan()
    assert parse_shape_mix(json.dumps(mix)).warmup == {"hit_rate": 0.62, "source": "prod 2026-11-01..07, replica r3"}
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        parse_shape_mix(json.dumps(dict(MIX, warmup={"hit_rate": 1.4})))
    assert parse_shape_mix(json.dumps(MIX)).warmup is None
    assert parse_shape_mix(json.dumps(dict(MIX, warmup={"source": "unknown"}))).warmup == {"source": "unknown"}


def test_hit_rate_window_measures_by_completion_time() -> None:
    window = HitRateWindow(10.0)
    assert window.rate(0.0) is None
    window.add(1.0, 100, 0)
    window.add(5.0, 100, 100)
    window.add(6.0, 100, None)  # no cached count reported: left out
    assert window.rate(6.0) == pytest.approx(0.5) and window.unreported == 1
    assert window.rate(12.0) == pytest.approx(1.0)  # the first response has left the window


# --- replay ---------------------------------------------------------------------------------------------------------


def _row(t, session, length, cached, **flags) -> dict:
    return {
        "time_s": t,
        "class": "chat",
        "session_id": session,
        "input_length": length,
        "cached_length": cached,
        "output_length": 3,
        "prefix_index": 0,
        **flags,
    }


# Warm-up rows of four sessions from -1 s, lead-ins among them, and a scored follow-up of session s1 at time 0:
#   row 0 w s1 | row 1 w s2 | row 2 lead-in s3 | row 3 w s3 | row 4 w s4 | row 5 w s1 (longer) | row 6 lead-in s1 | 7 scored
REPLAY_ROWS = [
    _row(-1.0, "s1", 700, 400, warmup=True),
    _row(-0.8, "s2", 700, 400, warmup=True),
    _row(-0.65, "s3", 800, 400, lead_in=True),
    _row(-0.6, "s3", 700, 400, warmup=True),
    _row(-0.4, "s4", 700, 400, warmup=True),
    _row(-0.3, "s1", 1000, 700, warmup=True),
    _row(-0.1, "s1", 900, 400, lead_in=True),
    _row(0.0, "s1", 1300, 900),
]


def _replay_run(tmp_path, host, plan, rows=None, **overrides):
    path = tmp_path / "schedule.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (rows or REPLAY_ROWS)) + "\n")
    options = _run_options(
        REPLAY_MIX, replay_schedule=str(path), replay_pods=1, replay_pod_index=0, qps=None, **overrides
    )
    options.replay_retime = options.replay_pod = options.replay_schedule_meta = None
    options.replay_retime_min_parent_gap = "off"
    options.replay_template_subtraction = False
    options.replay_pool_prefix = "max"
    ds = _dataset(REPLAY_MIX, tokenizer=WordTokenizer(), split_restore_classes=False)
    ds.build_prefixes()
    names = ("chat",)
    entries = load_replay_schedule(str(path), names)
    ds.prepare_replay(entries, incremental=True)
    started = time.monotonic()
    run_replay_warmup(SimpleNamespace(host=host), options, ds, plan)
    return ds, entries, time.monotonic() - started


def test_replay_warmup_rows_go_out_at_their_recorded_times_unscored_and_cover_the_lead_ins(
    mock_server, sent_log, tmp_path
) -> None:
    host, server_log = mock_server()
    plan = WarmupPlan("requests", requests=5, window_s=0.3)  # all five warm-up rows
    ds, entries, took = _replay_run(tmp_path, host, plan)
    # the warm-up rows in time order, and lead-in row 6 (s1) covered by the longer warm-up row 5, lead-in row 2 (s3)
    # sent in its place in the time order, one token
    assert [(row, one_token) for row, one_token, _, _ in sent_log] == [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (4, False),
        (5, False),
    ]
    at = [t - sent_log[0][3] for *_, t in sent_log]
    assert at == pytest.approx([0.0, 0.2, 0.35, 0.4, 0.6, 0.7], abs=0.12)  # the recorded gaps
    assert took >= 0.85  # and it waits out the recorded gap to time 0
    assert (Warmup.mode, Warmup.reason, Warmup.requests) == ("requests", "requests", 5)
    assert ds.lead_in_skipped == 1 and (LeadIns.sent, LeadIns.skipped, LeadIns.dropped) == (1, 1, 0)
    assert len(server_log.read_text().splitlines()) == 6
    # unscored: no designed cached count, no role; the measured row is laid out and scored
    plans = [ds._replay[e.index] for e in entries]
    assert [p["designed"] is None for p in plans] == [True] * 7 + [False]
    assert [p["role"] for p in plans][-1] == "first" and ds.replay_first_rows
    # the scored follow-up reuses what was actually sent for its session: the longer warm-up row
    scored = ds.replay_request(entries[7])
    tokenizer = ds._tokenizer
    warm_s1 = tokenizer.encode(sent_log[5][2])
    assert scored.reusable_tokens == 900 and _common_prefix(tokenizer.encode(scored.prompt), warm_s1) >= 900


def test_a_target_run_drops_the_rest_of_the_warmup_and_its_lead_ins_and_sends_what_is_still_uncovered(
    mock_server, sent_log, tmp_path
) -> None:
    host, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.2, tolerance=0.0, window_s=0.3, cap_s=30.0)
    ds, entries, took = _replay_run(tmp_path, host, plan)
    rows = [row for row, *_ in sent_log]
    assert Warmup.reason == "target" and 0 < Warmup.requests < 5
    assert took < 0.9  # no waiting out the rest of the recorded warm-up
    assert 3 not in rows and 4 not in rows and 5 not in rows  # dropped warm-up rows are never sent
    assert 2 not in rows and LeadIns.dropped == 1  # lead-in row 2 went with its session's dropped row 3
    # lead-in row 6: s1's earlier warm-up row 5 was dropped and row 0 is shorter than it, so it is sent, one token
    assert rows[-1] == 6 and sent_log[-1][1] is True and ds.lead_in_skipped == 0 and LeadIns.sent == 1
    tokenizer = ds._tokenizer
    scored = ds.replay_request(entries[7])  # laid out against what was sent: it reuses the lead-in
    assert scored.reusable_tokens == 900
    assert _common_prefix(tokenizer.encode(scored.prompt), tokenizer.encode(sent_log[-1][2])) >= 900


def test_a_count_below_the_row_count_sends_exactly_that_many_and_cuts_the_rest(mock_server, sent_log, tmp_path) -> None:
    host, _ = mock_server()
    _, _, took = _replay_run(tmp_path, host, WarmupPlan("requests", requests=2, window_s=0.3))
    assert [row for row, *_ in sent_log] == [0, 1, 6]  # the first two warm-up rows, then the uncovered lead-in
    assert Warmup.requests == 2 and Warmup.reason == "requests" and LeadIns.dropped == 1  # lead-in row 2 dropped
    assert took < 0.9


def test_no_warmup_drops_every_warmup_row_and_sends_the_uncovered_lead_ins(mock_server, sent_log, tmp_path) -> None:
    host, _ = mock_server()
    ds, _, _ = _replay_run(tmp_path, host, WarmupPlan(), no_warmup=True)
    assert [row for row, *_ in sent_log] == [6] and Warmup.mode == "none" and Warmup.requests == 0
    assert (
        ds.replay_request(load_replay_schedule(str(tmp_path / "schedule.jsonl"), ("chat",))[7]).reusable_tokens == 900
    )


def test_a_schedule_with_warmup_rows_but_no_flag_replays_all_of_them(mock_server, sent_log, tmp_path) -> None:
    host, _ = mock_server()
    _replay_run(tmp_path, host, WarmupPlan())
    assert [row for row, one_token, *_ in sent_log if not one_token] == [0, 1, 3, 4, 5]
    assert Warmup.mode == "requests" and Warmup.requests == 5


def test_replay_exhausted_when_the_target_is_never_reached(mock_server, sent_log, tmp_path, caplog) -> None:
    host, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.999, tolerance=0.0, window_s=0.3, cap_s=60.0)
    with caplog.at_level("WARNING", logger="load_test"):
        _replay_run(tmp_path, host, plan)
    assert Warmup.reason == "exhausted" and Warmup.requests == 5
    assert "Warm-up stopped on exhausted" in caplog.text


def test_replay_cap(mock_server, sent_log, tmp_path) -> None:
    host, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.999, tolerance=0.0, window_s=0.3, cap_s=0.5)
    _replay_run(tmp_path, host, plan)
    assert Warmup.reason == "cap" and 0 < Warmup.requests < 5


def test_warmup_rows_in_the_schedule_file(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in REPLAY_ROWS) + "\n")
    entries = load_replay_schedule(str(path), ("chat",))
    assert [e.warmup for e in entries] == [True, True, False, True, True, True, False, False]
    assert [e.lead_in for e in entries] == [False, False, True, False, False, False, True, False]
    assert [round(e.offset, 2) for e in entries] == [-1.0, -0.8, -0.65, -0.6, -0.4, -0.3, -0.1, 0.0]  # time 0: scored
    pacer = ReplayPacer(entries, "s.jsonl")
    assert [e.index for e in pacer.entries] == [7] and len(pacer.warmup_rows) == 5 and len(pacer.lead_ins) == 2
    retimed = load_replay_schedule(str(path), ("chat",), retime="poisson", seed=1)
    assert [e.offset for e in retimed if e.warmup] == [-1.0, -0.8, -0.6, -0.4, -0.3]  # warm-up rows keep their times
    both = tmp_path / "both.jsonl"
    both.write_text(json.dumps(_row(-1, "a", 700, 400, warmup=True, lead_in=True)) + "\n")
    with pytest.raises(ValueError, match="not both"):
        load_replay_schedule(str(both), ("chat",))
    late = tmp_path / "late.jsonl"
    late.write_text(json.dumps(_row(0, "a", 700, 400)) + "\n" + json.dumps(_row(2, "a", 700, 400, warmup=True)) + "\n")
    with pytest.raises(ValueError, match="before the first scored row"):
        load_replay_schedule(str(late), ("chat",))


def test_a_schedules_meta_block_gives_the_target_and_the_replayed_pods_figure(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    rows = [dict(r, pod="p1" if i % 2 else "p2") for i, r in enumerate(REPLAY_ROWS)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = {"warmup": {"hit_rate": 0.5, "hit_rate_by_pod": {"p1": 0.7, "p2": 0.3}, "source": "prod", "rows": 5}}
    (tmp_path / "s.jsonl.meta.json").write_text(json.dumps(meta))
    options = _run_options(
        REPLAY_MIX, replay_schedule=str(path), replay_pods=1, replay_pod_index=0, qps=None, warmup_cap_s=60.0
    )
    options.replay_retime = options.replay_schedule_meta = None
    options.replay_retime_min_parent_gap = "off"
    options.replay_pod = "p1"
    assert warmup_plan(options).hit_rate == 0.7
    options.replay_pod = "p2"
    assert warmup_plan(options).hit_rate == 0.3
    options.replay_pod = "p1,p2"  # a merge of pods: the schedule's own figure
    assert warmup_plan(options).hit_rate == 0.5
    meta["warmup"] = {"rows": 5, "source": "unknown hit rate"}  # unknown: the key is left out
    (tmp_path / "s.jsonl.meta.json").write_text(json.dumps(meta))
    options.replay_pod = "p1"
    assert warmup_plan(options) == WarmupPlan()
    (tmp_path / "s.jsonl.meta.json").write_text(json.dumps({"pool_prefix": {}}))  # no warmup block at all
    assert warmup_plan(options) == WarmupPlan()


# --- the delivery check ----------------------------------------------------------------------------------------------


def _timeline(intended, actual) -> None:
    SendTimeline.reset()
    for i, a in zip(intended, actual):
        SendTimeline.record(i, a)


def test_delivery_check_flags_a_short_rate_late_sends_and_failures() -> None:
    options = _run_options(MIX, qps=10.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    on_time = [i / 10 for i in range(101)]
    _timeline(on_time, on_time)
    entries, failures = delivery_entries(options, None, 0.0)
    assert failures == [] and entries["delivery_check"] == "pass"
    assert entries["delivered_over_target"] == pytest.approx(1.0)
    slow = [i / 8 for i in range(101)]  # 8 QPS delivered against 10
    _timeline(on_time, slow)
    entries, failures = delivery_entries(options, None, 0.0)
    assert entries["delivery_check"] == "fail" and len(failures) >= 1 and "delivered" in failures[0]
    late = [t + (1.0 if i % 5 == 0 else 0.0) for i, t in enumerate(on_time)]
    _timeline(on_time, late)
    _, failures = delivery_entries(options, None, 0.0)
    assert any("late-send share" in f for f in failures)
    _timeline(on_time, on_time)
    _, failures = delivery_entries(options, None, 0.2)
    assert any("failed-request share" in f for f in failures)
    options.fail_on_delivery = False
    entries, failures = delivery_entries(options, None, 0.2)
    assert entries["delivery_check"] == "off" and failures == []


def test_delivery_check_default_follows_whether_the_run_warms_up() -> None:
    options = _run_options(MIX, qps=10.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, None
    on_time = [i / 10 for i in range(101)]
    _timeline(on_time, [i / 5 for i in range(101)])  # 5 QPS against 10
    assert delivery_entries(options, None, 0.0)[0]["delivery_check"] == "off"  # no warm-up: exit codes as before
    Warmup.mode = "requests"
    entries, failures = delivery_entries(options, None, 0.0)
    assert entries["delivery_check"] == "fail" and failures
    options.fail_on_delivery = False  # --no-fail-on-delivery
    assert delivery_entries(options, None, 0.0)[0]["delivery_check"] == "off"


def test_a_replays_late_row_share_and_rate_come_from_its_pacer() -> None:
    entries = tuple(
        load_test.ScheduleEntry(offset=i * 0.1, class_index=0, index=i) for i in range(50)
    )  # 10 QPS of rows
    pacer = ReplayPacer(entries, "s.jsonl")
    pacer.late_rows = 10
    options = _run_options(MIX, qps=None, replay_schedule="s.jsonl", qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    _timeline([e.offset for e in entries], [e.offset for e in entries])
    summary, failures = delivery_entries(options, pacer, 0.0)
    assert summary["target_qps"] == pytest.approx(10.0) and summary["late_send_share"] == pytest.approx(0.2)
    assert len(failures) == 1 and "late-send share" in failures[0]


def _fire_quitting(options, tmp_path) -> int:
    """Run the quitting listener on an environment whose requests all succeeded; the process exit code it sets."""
    from locust.env import Environment
    from locust.stats import RequestStats

    environment = Environment(user_classes=[])
    environment.parsed_options = options
    environment.stats = RequestStats()
    environment.stats.log_request("METRIC", "total_latency", 100, 0)
    environment.stats.log_request("POST", "/v1/completions", 100, 10)
    load_test.events.quitting.fire(environment=environment)
    return environment.process_exit_code


def test_a_run_that_did_not_deliver_exits_3_only_with_the_check_on(tmp_path, monkeypatch) -> None:
    options = _run_options(
        MIX,
        qps=10.0,
        qps_max_lateness=0.5,
        qps_min_delivered=0.95,
        max_late_share=0.05,
        max_fail_ratio=0.01,
        summary_file=None,
        fail_on_delivery=None,
        embeddings=False,
        rerank=False,
        replay_schedule=None,
        client_in_flight_interval=0,
    )
    on_time = [i / 10 for i in range(101)]
    monkeypatch.setattr(load_test.InitTracker, "logging_params", {})
    monkeypatch.setattr(load_test.FixedQPSPacer, "_instance", None)
    _timeline(on_time, [i / 5 for i in range(101)])  # delivers 5 QPS of a target 10
    assert _fire_quitting(options, tmp_path) == 0  # no warm-up: the exit code is what it was
    Warmup.mode = "requests"
    assert _fire_quitting(options, tmp_path) == 3  # warmed up: the delivery check is on
    options.fail_on_delivery = False
    assert _fire_quitting(options, tmp_path) == 0
    options.fail_on_delivery = None
    _timeline(on_time, on_time)  # delivered
    assert _fire_quitting(options, tmp_path) == 0


# --- compare -----------------------------------------------------------------------------------------------------


def _summary(path, rows) -> str:
    base = {
        "Workload Sha256": "abc",
        "Request Seed": "11",
        "Warmup Requests": "120",
        "Target Qps": "40.0",
        "Llm Bench Version": "1",
        "Qps": "39.0",
        "P50 Time To First Token": "100",
        "P99 Total Latency": "900",
        "Measured Hit Rate": "0.60",
        "Fail Ratio": "0.0",
    }
    merged = [{**base, **row} for row in rows]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    return str(path)


@pytest.mark.parametrize(
    "change",
    [
        {"Workload Sha256": "def"},
        {"Request Seed": "12"},
        {"Warmup Requests": "130"},
        {"Target Qps": "50.0"},
        {"Llm Bench Version": "2"},
    ],
)
def test_compare_refuses_when_the_workload_identity_differs(tmp_path, capsys, change) -> None:
    a = _summary(tmp_path / "a.csv", [{}])
    b = _summary(tmp_path / "b.csv", [change])
    assert compare.main(["--a", a, "--b", b]) == 2
    assert "Refusing to compare" in capsys.readouterr().err


def test_compare_refuses_when_a_sides_repeats_disagree_or_the_identity_is_missing(tmp_path) -> None:
    a = _summary(tmp_path / "a.csv", [{}, {"Warmup Requests": "121"}])
    b = _summary(tmp_path / "b.csv", [{}])
    assert compare.main(["--a", a, "--b", b]) == 2
    old = _summary(tmp_path / "old.csv", [{"Workload Sha256": "", "Llm Bench Version": ""}])
    assert compare.main(["--a", old, "--b", old]) == 2  # a summary from a client that doesn't write the identity


def test_compare_reports_deltas_against_the_noise_floor(tmp_path, capsys) -> None:
    a = [
        _summary(tmp_path / "a1.csv", [{"P50 Time To First Token": "100"}]),
        _summary(tmp_path / "a2.csv", [{"P50 Time To First Token": "110", "Qps": "39.0"}]),
    ]
    b = _summary(
        tmp_path / "b.csv",
        [{"P50 Time To First Token": "150", "P99 Total Latency": "890", "Qps": "39.2", "Measured Hit Rate": "0.7"}],
    )
    out = tmp_path / "out.json"
    md = tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", b, "--json", str(out), "--markdown", str(md)]) == 0
    by_metric = {m["metric"]: m for m in json.loads(out.read_text())["metrics"]}
    ttft = by_metric["P50 Time To First Token"]
    assert ttft["verdict"] == "worse" and ttft["noise"] == pytest.approx(10 / 105)  # the repeats' spread beats 3%
    assert by_metric["Qps"]["verdict"] == "same"  # +0.5% against a 3% floor
    assert by_metric["P99 Total Latency"]["verdict"] == "same"  # -1.1%
    assert by_metric["Measured Hit Rate"]["verdict"] == "better" and by_metric["Measured Hit Rate"]["delta"] == (
        pytest.approx(0.1)
    )
    assert "| P50 Time To First Token |" in md.read_text() and "worse" in capsys.readouterr().out


def test_the_summary_carries_the_identity_compare_needs(tmp_path) -> None:
    mix_path = tmp_path / "mix.json"
    mix_path.write_text(json.dumps(MIX))
    options = _run_options(MIX, shape_mix=f"@{mix_path}", replay_schedule=None)
    first = load_test.workload_sha256(options)
    assert first == load_test.workload_sha256(options) and len(first) == 64
    mix_path.write_text(json.dumps(dict(MIX, seed=12)))
    assert load_test.workload_sha256(options) != first
    assert load_test.LLM_BENCH_VERSION
    assert LIMERICKS  # the corpus the test datasets use


def test_the_cli_flags(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    load_test.init_parser(parser)
    defaults = parser.parse_args([])
    assert defaults.warmup_hit_rate is None and defaults.warmup_requests is None and not defaults.no_warmup
    assert (defaults.warmup_tolerance, defaults.warmup_window_s, defaults.warmup_cap_s) == (0.02, 30.0, None)
    assert defaults.fail_on_delivery is None and defaults.qps_min_delivered == 0.95  # unset: on only with a warm-up
    parsed = parser.parse_args(
        ["--warmup-requests", "120", "--no-warmup", "--no-fail-on-delivery", "--warmup-cap-s", "60", "--qps", "5"]
    )
    assert (parsed.warmup_requests, parsed.no_warmup, parsed.fail_on_delivery, parsed.warmup_cap_s) == (
        120,
        True,
        False,
        60.0,
    )
    assert parser.parse_args(["--fail-on-delivery"]).fail_on_delivery is True


def test_the_summary_row_has_the_warmup_fields_and_the_identity(tmp_path, monkeypatch) -> None:
    options = _run_options(
        MIX,
        qps=10.0,
        qps_max_lateness=0.5,
        qps_min_delivered=0.95,
        max_late_share=0.05,
        max_fail_ratio=0.01,
        summary_file=str(tmp_path / "summary.csv"),
        fail_on_delivery=False,
        embeddings=False,
        rerank=False,
        client_in_flight_interval=0,
    )
    monkeypatch.setattr(load_test.InitTracker, "logging_params", {})
    monkeypatch.setattr(load_test.FixedQPSPacer, "_instance", None)
    _timeline([i / 10 for i in range(11)], [i / 10 for i in range(11)])
    Warmup.mode, Warmup.reason, Warmup.requests, Warmup.hit_rate, Warmup.target = "target", "cap", 88, 0.41, 0.5
    Warmup.prompt_tokens, Warmup.duration_s = 26400, 12.5
    MeasuredReuse.record(1000, 600)
    MeasuredReuse.record(1000, 700)
    _fire_quitting(options, tmp_path)
    (row,) = list(csv.DictReader(open(tmp_path / "summary.csv")))
    assert (row["Warmup Mode"], row["Warmup Reason"], row["Warmup Requests"]) == ("target", "cap", "88")
    assert (row["Warmup Prompt Tokens"], row["Warmup Duration S"]) == ("26400", "12.5")
    assert (row["Warmup Hit Rate"], row["Warmup Target Hit Rate"], row["Measured Hit Rate"]) == ("0.41", "0.5", "0.65")
    assert row["Workload Sha256"] == load_test.workload_sha256(options) and row["Request Seed"] == "11"
    assert row["Llm Bench Version"] == load_test.LLM_BENCH_VERSION and row["Delivery Check"] == "off"
    # and compare reads exactly those columns
    assert {column for column, _ in compare.IDENTITY} <= set(row)
