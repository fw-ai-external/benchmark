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
    assert entries["delivered_over_intended"] == pytest.approx(1.0)
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
    pacer.late_rows, pacer.sent_rows = 10, 50  # 50 rows were due when the run ended
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
        "Workload Options Sha256": "opts",
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
        {"Workload Options Sha256": "other"},
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
        _summary(tmp_path / f"a{i}.csv", [{"P50 Time To First Token": ttft, "Qps": qps, "Measured Hit Rate": "0.6"}])
        for i, (ttft, qps) in enumerate([("100", "39.0"), ("110", "39.0"), ("105", "39.0")])
    ]
    b = [
        _summary(
            tmp_path / f"b{i}.csv",
            [{"P50 Time To First Token": "150", "P99 Total Latency": "890", "Qps": "39.2", "Measured Hit Rate": "0.7"}],
        )
        for i in range(3)
    ]
    out = tmp_path / "out.json"
    md = tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out), "--markdown", str(md)]) == 0
    by_metric = {m["metric"]: m for m in json.loads(out.read_text())["metrics"]}
    ttft = by_metric["P50 Time To First Token"]
    assert ttft["verdict"] == "worse" and ttft["noise"] == pytest.approx(10 / 105)  # the repeats' spread beats 3%
    assert ttft["noise_from"] == "repeat spread" and by_metric["Qps"]["noise_from"] == "--min-noise"
    assert by_metric["Qps"]["verdict"] == "same"  # +0.5% against a 3% floor
    assert by_metric["P99 Total Latency"]["verdict"] == "same"  # -1.1%
    assert by_metric["Measured Hit Rate"]["verdict"] == "better" and by_metric["Measured Hit Rate"]["delta"] == (
        pytest.approx(0.1)
    )
    text = md.read_text()
    assert "| P50 Time To First Token |" in text and "(repeat spread)" in text and "| 3/3 |" in text


def test_g3_a_better_or_worse_verdict_needs_three_repeats_per_side(tmp_path) -> None:
    ident = {c: "1" for c, _ in compare.IDENTITY}
    one = lambda v: dict(ident, **{"P90 Time To First Token": v})
    for repeats in (1, 2):
        verdicts = compare.compare([one("100")] * repeats, [one("110")] * repeats, 0.03)
        assert verdicts[0]["verdict"] == "inconclusive (too few repeats)", repeats
    assert compare.compare([one("100")] * 3, [one("110")] * 3, 0.03)[0]["verdict"] == "worse"
    assert compare.compare([one("100")] * 3, [one("110")] * 2, 0.03)[0]["verdict"] == "inconclusive (too few repeats)"
    assert compare.compare([one("100")] * 2, [one("110")] * 2, 0.03, min_repeats=2)[0]["verdict"] == "worse"
    assert compare.compare([one("100")] * 2, [one("101")] * 2, 0.03)[0]["verdict"] == "same"  # inside the floor


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


# --- fix round 1 ---------------------------------------------------------------------------------------------------


def test_f1_a_perfectly_delivered_random_run_never_fails_the_rate_check() -> None:
    """A Poisson run's realized rate scatters around --qps; delivery is judged against its own intended schedule."""
    options = _run_options(MIX, qps=2.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    for seed in range(60):
        pacer = load_test.FixedQPSPacer(2.0, "exponential", 1.0, seed)
        times = []
        while (t := pacer.next_offset()) <= 60:
            times.append(t)
        _timeline(times, times)  # actual == intended: a perfect client
        entries, failures = delivery_entries(options, None, 0.0)
        assert failures == [], (seed, failures)
        assert entries["delivered_over_intended"] == pytest.approx(1.0)


def test_f1_late_share_and_rate_count_only_the_rows_due_when_the_run_ended() -> None:
    entries = tuple(load_test.ScheduleEntry(offset=i * 0.1, class_index=0, index=i) for i in range(100))
    pacer = ReplayPacer(entries, "s.jsonl")
    pacer.sent_rows, pacer.late_rows = 20, 3  # -t cut the replay after 20 of 100 rows; 3 of those 20 were late
    options = _run_options(MIX, qps=None, replay_schedule="s.jsonl", qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    due = [e.offset for e in entries[:20]]
    _timeline(due, due)
    summary, failures = delivery_entries(options, pacer, 0.0)
    assert summary["late_send_share"] == pytest.approx(0.15)  # of the 20 due, not of the 100 rows
    assert len(failures) == 1 and "late-send share" in failures[0]
    # skipped rows were due and did not go out
    pacer.late_rows, pacer.skipped_rows, SendTimeline.skipped = 0, 5, 5
    summary, failures = delivery_entries(options, pacer, 0.0)
    assert summary["delivered_over_intended"] == pytest.approx(20 / 25) and any("delivered" in f for f in failures)


def _rows_with_session_context(second_lead_in_input: int) -> list[dict]:
    """A session warmed up by one row (700 prompt tokens, a 20-token reply) and a lead-in, then a scored follow-up."""
    return [
        _row(-0.5, "s1", 700, 400, warmup=True, output_length=20),
        _row(-0.1, "s1", second_lead_in_input, 0, lead_in=True),
        _row(0.0, "s1", second_lead_in_input + 300, second_lead_in_input),
    ]


def test_f2_a_lead_in_of_the_prompt_plus_its_reply_is_covered_by_the_sent_prompt(
    mock_server, sent_log, tmp_path
) -> None:
    host, _ = mock_server()
    ds, entries, _ = _replay_run(tmp_path, host, WarmupPlan(), rows=_rows_with_session_context(720))  # 700 + 20
    assert [(row, one_token) for row, one_token, *_ in sent_log] == [(0, False)]  # the lead-in (row 1) isn't resent
    assert ds.lead_in_skipped == 1 and LeadIns.sent == 0


def test_f2_a_lead_in_that_isnt_covered_continues_the_text_the_session_was_sent(
    mock_server, sent_log, tmp_path
) -> None:
    host, _ = mock_server()
    ds, entries, _ = _replay_run(tmp_path, host, WarmupPlan(), rows=_rows_with_session_context(900))
    assert [(row, one_token) for row, one_token, *_ in sent_log] == [(0, False), (1, True)]
    tokenizer = ds._tokenizer
    warm, lead = (tokenizer.encode(prompt) for _, _, prompt, _ in sent_log)
    assert _common_prefix(warm, lead) >= 700  # not fresh text: the whole warm-up prompt is held ahead of its tail


def test_f3_either_hit_rate_may_be_absent_from_the_meta(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    rows = [dict(r, pod="p1" if i % 2 else "p2") for i, r in enumerate(REPLAY_ROWS)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "s.jsonl.meta.json").write_text(
        json.dumps({"warmup": {"hit_rate_by_pod": {"p1": 0.7}, "rows": 5, "source": "prod"}})
    )
    options = _run_options(REPLAY_MIX, replay_schedule=str(path), replay_pods=1, replay_pod_index=0, qps=None)
    options.replay_retime = options.replay_schedule_meta = None
    options.replay_retime_min_parent_gap = "off"
    options.replay_pod = "p1"
    assert warmup_plan(options).hit_rate == 0.7
    options.replay_pod = "p2"  # no figure for this pod and no schedule-wide one: nothing to warm up to
    assert warmup_plan(options) == WarmupPlan()


def test_f4_a_replay_target_needs_no_cap_but_a_shape_mix_one_does(tmp_path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in REPLAY_ROWS) + "\n")
    (tmp_path / "s.jsonl.meta.json").write_text(json.dumps({"warmup": {"hit_rate": 0.5, "rows": 5}}))
    options = _run_options(REPLAY_MIX, replay_schedule=str(path), replay_pods=1, replay_pod_index=0, qps=None)
    options.replay_retime = options.replay_schedule_meta = options.replay_pod = None
    options.replay_retime_min_parent_gap = "off"
    plan = warmup_plan(options)
    assert (plan.mode, plan.hit_rate, plan.cap_s) == ("target", 0.5, None)  # the warm-up rows bound it
    options.prefix_fill = True  # the fill step doesn't warm up, whatever the meta says
    assert warmup_plan(options) == WarmupPlan()
    mix = dict(REPLAY_MIX, warmup={"hit_rate": 0.5, "source": "prod"})
    options = _run_options(mix, prefix_fill=True)
    assert warmup_plan(options) == WarmupPlan()  # nor does a mix file's block fail the fill run
    with pytest.raises(ValueError, match="--warmup-cap-s"):
        warmup_plan(_run_options(mix))
    with pytest.raises(ValueError, match="could cut B's warm-up"):  # A's cap left on B's command line
        warmup_plan(_run_options(MIX, warmup_requests=7, warmup_cap_s=5.0))


def test_f4_a_replay_target_without_a_cap_runs_to_exhaustion(mock_server, sent_log, tmp_path) -> None:
    host, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.999, tolerance=0.0, window_s=0.3, cap_s=None)
    _replay_run(tmp_path, host, plan)
    assert Warmup.reason == "exhausted" and Warmup.requests == 5


def test_f5_a_lead_in_completes_before_its_sessions_next_row_but_other_sessions_go_on(monkeypatch, tmp_path) -> None:
    timeline = {}
    post = UnscoredSender.post

    def slow(self, request, one_token=False):
        start = time.monotonic()
        if one_token:
            time.sleep(0.4)  # a slow lead-in (the real one is a request to the server)
        timeline[request.schedule_row] = (start, time.monotonic())
        return (request.prompt_tokens, 0)

    # gevent-patched sleep: the other sessions' sends run while this one waits
    monkeypatch.setattr(UnscoredSender, "post", slow)
    host = "http://127.0.0.1:9"
    _replay_run(tmp_path, host, WarmupPlan("requests", requests=5, window_s=0.3))
    lead_end = timeline[2][1]  # row 2: the lead-in of session s3; row 3: s3's warm-up row; row 4: session s4's
    assert timeline[3][0] >= lead_end - 0.01  # s3's row waits for its lead-in
    assert timeline[4][0] < lead_end  # s4's goes out meanwhile
    assert post  # the unpatched sender is what the other tests use


def test_f6_a_run_given_the_count_makes_the_same_lead_in_decisions_as_the_run_that_reported_it(
    mock_server, sent_log, tmp_path
) -> None:
    # a lead-in for s4 sits just before s4's first warm-up row; a target run that stops at that row drops both
    rows = REPLAY_ROWS[:5] + [_row(-0.45, "s4", 800, 400, lead_in=True)] + REPLAY_ROWS[5:]
    host_a, _ = mock_server()
    plan = WarmupPlan("target", hit_rate=0.2, tolerance=0.0, window_s=0.6, cap_s=60.0)
    _replay_run(tmp_path, host_a, plan, rows=rows)
    n, dropped = Warmup.requests, LeadIns.dropped
    a_sent = sorted((row, one_token) for row, one_token, *_ in sent_log)  # the same sends (their order may vary)
    assert Warmup.reason == "target" and 0 < n < 5
    sent_log.clear()
    LeadIns.sent = LeadIns.dropped = 0
    host_b, _ = mock_server()
    _replay_run(tmp_path, host_b, WarmupPlan("requests", requests=n, window_s=0.6), rows=rows)
    assert sorted((row, one_token) for row, one_token, *_ in sent_log) == a_sent and LeadIns.dropped == dropped
    assert dropped == 1 and (5, True) not in a_sent  # lead-in row 5, for s4, went with s4's dropped row


def test_f7_a_warmup_whose_rows_were_all_sent_waits_out_the_gap_to_time_0_however_it_ended(
    mock_server, sent_log, tmp_path
) -> None:
    host, _ = mock_server()
    # the cap passes while the lead-in after the last warm-up row is due: every warm-up row had been sent
    plan = WarmupPlan("target", hit_rate=0.999, tolerance=0.0, window_s=0.3, cap_s=0.8)
    _, _, took_a = _replay_run(tmp_path, host, plan)
    assert (Warmup.reason, Warmup.requests) == ("exhausted", 5) and took_a >= 0.85
    host_b, _ = mock_server()
    _, _, took_b = _replay_run(tmp_path, host_b, WarmupPlan("requests", requests=5, window_s=0.3))
    assert Warmup.reason == "requests" and took_b >= 0.85  # B with the same count takes the same branch


def test_f8_the_workload_options_are_reported_and_every_name_is_a_real_option() -> None:
    parser = argparse.ArgumentParser()
    load_test.init_parser(parser)
    dests = {a.dest for a in parser._actions}
    assert set(load_test.WORKLOAD_OPTIONS) <= dests
    options = _run_options(MIX, replay_pod=None, replay_retime=None, tokenizer=None)
    base = load_test.workload_options(options)
    for name, value in [
        ("replay_pod", "p2"),
        ("replay_pods", 2),
        ("replay_pod_index", 1),
        ("replay_retime", "poisson"),
        ("qps_distribution", "gamma"),
        ("qps_cv2", 4.0),
        ("qps_dispersion", 3.0),
        ("shape_mix_corpus", "x.txt"),
        ("tokenizer", "other"),
        ("shape_mix_sample_lengths", True),
        ("shape_mix_output_mode", "max"),
        ("max_tokens", 10),
        ("reasoning_effort", "low"),
    ]:
        assert load_test.workload_options(_run_options(MIX, **{name: value})) != base, name
    chat = _run_options(MIX, replay_pod=None, replay_retime=None, tokenizer=None)
    chat.chat = True
    assert load_test.workload_options(chat) != base
    load_test.InitTracker.deferred_run_time_seconds = 600
    try:
        assert load_test.workload_options(options) != base  # -t
    finally:
        load_test.InitTracker.deferred_run_time_seconds = None


def test_f8_compare_names_the_options_that_differ_and_ignores_non_measured_rows(tmp_path, capsys) -> None:
    a = _summary(tmp_path / "a.csv", [{"Workload Options": json.dumps({"qps_distribution": "gamma"})}])
    b = _summary(
        tmp_path / "b.csv",
        [{"Workload Options": json.dumps({"qps_distribution": "constant"}), "Workload Options Sha256": "z"}],
    )
    assert compare.main(["--a", a, "--b", b]) == 2
    assert "qps_distribution: A 'gamma' vs B 'constant'" in capsys.readouterr().err
    # a --prefix-fill step's row (another seed, no hit rate) is no run: left out of the comparison, and said so
    fill = _summary(tmp_path / "fill.csv", [{"Mode": "prefix_fill", "Request Seed": "99", "Qps": "1"}])
    ok = _summary(tmp_path / "ok.csv", [{}])
    assert compare.main(["--a", ok, fill, "--b", ok]) == 0
    assert "ignoring" in capsys.readouterr().err
    with pytest.raises(ValueError, match="no measured summary rows"):
        compare.read_rows([fill])


def test_f9_compare_verdicts(tmp_path, capsys) -> None:
    ident = {c: "1" for c, _ in compare.IDENTITY}
    zero_a = [dict(ident, **{"Requests Failed": "0", "Fail Ratio": "0", "P50 Total Latency": "100"})]
    zero_b = [dict(ident, **{"Requests Failed": "250", "Fail Ratio": "0.025", "P50 Total Latency": "100"})]
    by = {r["metric"]: r for r in compare.compare(zero_a, zero_b, 0.03)}
    assert by["Requests Failed"]["verdict"] == "different" and by["Requests Failed"]["delta"] == 250
    assert by["Requests Failed"]["delta_kind"] == "abs"  # the difference, not a relative change from 0
    assert by["P50 Total Latency"]["verdict"] == "same"
    # a noisy side swallows the delta: not "same"
    noisy = [dict(ident, **{"P50 Total Latency": v}) for v in ("100", "160")]
    steady = [dict(ident, **{"P50 Total Latency": v}) for v in ("131", "132")]
    verdict = {r["metric"]: r for r in compare.compare(noisy, steady, 0.03)}["P50 Total Latency"]["verdict"]
    assert verdict == "inconclusive (noisy repeats)"
    # runs that failed delivery or the failure ratio are flagged in the output
    a = _summary(tmp_path / "a.csv", [{}])
    b = _summary(
        tmp_path / "b.csv", [{"Delivery Check": "fail", "Delivery Failures": "late-send share 0.2", "Failed": "1"}]
    )
    out = tmp_path / "out.json"
    assert compare.main(["--a", a, "--b", b, "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "!!! B run 1: delivery check failed (late-send share 0.2); exceeded --max-fail-ratio" in printed
    assert json.loads(out.read_text())["flagged_runs"][0]["side"] == "B"


def test_f10_processes_with_a_warmup_is_refused() -> None:
    options = argparse.Namespace(
        shape_mix=json.dumps(MIX),
        chat=False,
        tokenizer=None,
        shape_mix_corpus=None,
        prefix_reuse_overhang=128,
        qps=5.0,
        processes=2,
        warmup_requests=5,
    )
    with pytest.raises(ValueError, match="--processes N with a warm-up"):
        load_test.validate_shape_mix_options(options)
    options.warmup_requests, options.processes = None, 2
    load_test.validate_shape_mix_options(options)  # no warm-up: processes as before
    options.warmup_requests, options.processes = 5, 1
    load_test.validate_shape_mix_options(options)


def test_f11_warmup_draws_leave_the_measured_counters_alone() -> None:
    ds = _dataset(MIX, tokenizer=WordTokenizer(), context_tokens=302)  # every 300 + 4 token request is cut to fit
    ds.build_prefixes()
    for _ in range(5):
        assert ds.warmup_request().generated_tokens == 2
    assert ds.context_cut_requests == 0
    next(ds)
    assert ds.context_cut_requests == 1


def test_f11_the_last_window_is_over_the_full_window_whenever_it_is_asked() -> None:
    window = HitRateWindow(10.0)
    for at, cached in ((1.0, 0), (6.0, 100), (11.0, 100), (14.0, 100)):
        window.add(at, 100, cached)
    assert window.rate(14.0) == pytest.approx(1.0)  # [4, 14]: three responses, all hits
    assert window.rate(11.0) == pytest.approx(2 / 3)  # [1, 11]: asked later, an earlier window is still whole
    assert window.rate(14.0) == pytest.approx(1.0)


# --- fix round 2 ---------------------------------------------------------------------------------------------------


def test_g1_a_scored_row_continues_from_the_sessions_sent_text_when_that_covers_its_lead_in(tmp_path) -> None:
    """The class pool prefix (sized to the longest recorded shared length) is longer than the session's text, and
    used to win: the first scored row after a full warm-up reused none of the warm-up. The bar a lead-in is covered by
    (its content against the session's last prompt plus reply) is the bar the scored row now continues by."""
    mix = {
        "seed": 5,
        "classes": [
            {
                "name": "chat",
                "weight": 1.0,
                "prompt_tokens": 1500,
                "cached_prompt_tokens": 800,
                "generated_tokens": 4,
                "prefix_pool_size": 1,
            }
        ],
    }
    # 30 template tokens and a 10-token head: the lead-in's content (720 - 30) is held by the warm-up row (670 content
    # tokens + a 20-token reply), but the scored row wants 720 - 10 shared, which that text is just short of
    rows = [
        _row(-0.5, "s1", 700, 0, warmup=True, output_length=20),
        _row(-0.1, "s1", 720, 0, lead_in=True),
        _row(0.0, "s1", 1200, 720),
    ]
    path = tmp_path / "s.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tokenizer = WordTokenizer()
    ds = _dataset(mix, tokenizer=tokenizer, split_restore_classes=False, chat_overhead_tokens=30, chat_head_tokens=10)
    ds.build_prefixes()
    entries = load_replay_schedule(str(path), ("chat",))
    ds.prepare_replay(entries, message_overhead=lambda n: 30, incremental=True)
    assert ds.lay_out_replay_row(entries[0]) and not ds.lay_out_replay_row(entries[1])  # the lead-in is covered
    assert ds.lay_out_replay_row(entries[2])
    warm, scored = ds.replay_request(entries[0]), ds.replay_request(entries[2])
    shared = _common_prefix(tokenizer.encode(warm.prompt), tokenizer.encode(scored.prompt))
    assert shared >= 650 and scored.reusable_tokens >= 650  # the warm-up's text, not the pool prefix and a fresh tail


def _clock(monkeypatch, jump: float) -> None:
    """The wall clock `jump` seconds ahead of where the monotonic clock says it is (a host that slept)."""
    real = time.time
    monkeypatch.setattr(load_test.time, "time", lambda: real() + jump)


def test_g2_a_stall_that_stopped_the_monotonic_clock_fails_the_delivery_check(monkeypatch) -> None:
    options = _run_options(MIX, qps=2.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    on_time = [i / 2 for i in range(40)]
    _timeline(on_time, on_time)  # every send went out exactly on its intended time on the monotonic clock
    SendTimeline.spawned()
    assert delivery_entries(options, None, 0.0)[1] == []  # a healthy run
    _clock(monkeypatch, 944.0)  # ... but 944 s passed on the wall clock: the laptop slept
    entries, failures = delivery_entries(options, None, 0.0)
    assert entries["delivery_check"] == "fail" and entries["clock_stall_s"] == pytest.approx(944, abs=1)
    assert any("clock stopped for 944 s" in f for f in failures)


def test_g2_a_replay_that_sent_a_fraction_of_the_rows_due_by_the_wall_clock_fails(monkeypatch) -> None:
    entries = tuple(load_test.ScheduleEntry(offset=i * 3.5, class_index=0, index=i) for i in range(280))
    pacer = ReplayPacer(entries, "s.jsonl")
    pacer.sent_rows = 81  # the schedule paused when the host slept: 81 of 280 rows went out
    options = _run_options(MIX, qps=None, replay_schedule="s.jsonl", qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    sent = [e.offset for e in entries[:81]]
    _timeline(sent, sent)
    SendTimeline.spawned()
    _clock(monkeypatch, 944.0)
    summary, failures = delivery_entries(options, pacer, 0.0)
    assert summary["delivery_check"] == "fail"
    assert any(
        "only 81 of 269 rows due by the wall clock were sent" in f for f in failures
    )  # 3.5 s apart, to 944 s less the slack


def test_g2_a_burst_after_a_stall_is_not_delivery() -> None:
    options = _run_options(MIX, qps=2.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    intended = [i / 2 for i in range(21)]  # 10 s of schedule
    _timeline(intended, [10.0 + i / 5 for i in range(21)])  # sent in 4 s: a backlog released at once
    entries, failures = delivery_entries(options, None, 0.0)
    assert entries["delivered_over_intended"] == pytest.approx(2.5)
    assert any("faster than intended" in f for f in failures) and entries["delivery_check"] == "fail"


def test_g4_the_lead_ins_after_a_cut_go_out_in_schedule_order_one_after_another(monkeypatch, tmp_path) -> None:
    spans = []

    def slow(self, request, one_token=False):
        start = time.monotonic()
        time.sleep(0.15)
        spans.append((request.schedule_row, start, time.monotonic()))
        return (request.prompt_tokens, 0)

    monkeypatch.setattr(UnscoredSender, "post", slow)
    rows = [
        _row(-0.3, "s1", 700, 400, warmup=True),
        _row(-0.2, "s2", 700, 400, warmup=True),
        _row(-0.1, "s3", 800, 0, lead_in=True),  # three scored sessions' lead-ins, none ahead of a dropped row
        _row(-0.1, "s4", 800, 0, lead_in=True),
        _row(-0.1, "s5", 800, 0, lead_in=True),
        _row(0.0, "s3", 1000, 800),
    ]
    _replay_run(tmp_path, "http://127.0.0.1:9", WarmupPlan("requests", requests=0, window_s=0.3), rows=rows)
    ahead = sorted(s for s in spans if s[0] in (2, 3, 4))
    assert [row for row, *_ in ahead] == [2, 3, 4]  # schedule order
    assert ahead[1][1] >= ahead[0][2] - 0.01 and ahead[2][1] >= ahead[1][2] - 0.01  # each after the one before
