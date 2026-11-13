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


def _fire_quitting(options, tmp_path, host=None, scored=True) -> int:
    """Run the quitting listener on an environment whose requests all succeeded; the process exit code it sets."""
    from locust.env import Environment
    from locust.stats import RequestStats

    environment = Environment(user_classes=[])
    environment.parsed_options = options
    environment.host = host
    environment.stats = RequestStats()
    if scored:
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


def test_compare_refuses_an_unmeasurable_identity(tmp_path, capsys) -> None:
    old = _summary(tmp_path / "old.csv", [{"Workload Sha256": "", "Llm Bench Version": ""}])
    assert compare.main(["--a", old, "--b", old]) == 2  # a summary from a client that doesn't write the identity


def _pairs(tmp_path, n=3, a_extra=None, b_extra=None):
    """n paired runs: pair i has seed 100 + i and warm-up requests 50 + i on both sides (as A_i's printed N_i)."""
    a, b = [], []
    for i in range(n):
        ident = {"Request Seed": str(100 + i), "Warmup Requests": str(50 + i)}
        a.append(_summary(tmp_path / f"a{i}.csv", [{**ident, **(a_extra(i) if a_extra else {})}]))
        b.append(_summary(tmp_path / f"b{i}.csv", [{**ident, **(b_extra(i) if b_extra else {})}]))
    return a, b


def test_r1_runs_are_paired_each_pair_with_its_own_seed_and_warmup_count(tmp_path, capsys) -> None:
    a, b = _pairs(tmp_path)  # three pairs with three seeds and three counts: refused before pairing existed
    out = tmp_path / "out.json"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out)]) == 0
    assert json.loads(out.read_text())["metrics"][0]["pairs"] == 3


def test_r1_runs_of_one_file_pair_in_row_order_and_the_counts_must_match(tmp_path, capsys) -> None:
    rows_a = [{"Request Seed": str(i), "Warmup Requests": "7"} for i in range(3)]
    a = _summary(tmp_path / "a.csv", rows_a)  # one summary file with all three runs (--summary-file appends rows)
    b = _summary(tmp_path / "b.csv", rows_a)
    assert compare.main(["--a", a, "--b", b]) == 0
    short = _summary(tmp_path / "short.csv", rows_a[:2])
    assert compare.main(["--a", a, "--b", short]) == 2
    assert "A has 3 run(s) and B 2" in capsys.readouterr().err


def test_r1_a_pair_whose_seed_or_warmup_count_differs_is_refused_and_named(tmp_path, capsys) -> None:
    a, b = _pairs(tmp_path, b_extra=lambda i: {"Request Seed": "999"} if i == 1 else {})
    assert compare.main(["--a", *a, "--b", *b]) == 2
    assert "pair 2: request seed: A 101 vs B 999" in capsys.readouterr().err
    a, b = _pairs(tmp_path, b_extra=lambda i: {"Warmup Requests": "1"} if i == 2 else {})
    assert compare.main(["--a", *a, "--b", *b]) == 2
    assert "pair 3: warm-up requests: A 52 vs B 1 (give B_i the --warmup-requests N_i" in capsys.readouterr().err


def test_r1_a_seed_repeated_across_pairs_is_refused(tmp_path, capsys) -> None:
    a, b = _pairs(tmp_path, a_extra=lambda i: {"Request Seed": "5"}, b_extra=lambda i: {"Request Seed": "5"})
    assert compare.main(["--a", *a, "--b", *b]) == 2
    err = capsys.readouterr().err
    assert "request seed 5 is the same in pairs 1, 2, 3" in err and "resends the first run's prompts" in err


def test_r1_the_workload_must_match_across_every_run_but_the_seed_may_differ_in_the_options(tmp_path, capsys) -> None:
    a, b = _pairs(tmp_path, a_extra=lambda i: {"Workload Sha256": "other"} if i == 2 else {})
    assert compare.main(["--a", *a, "--b", *b]) == 2  # pair 3's A differs from every other run
    assert "workload file sha256 differs between runs" in capsys.readouterr().err
    options = lambda seed: json.dumps({"qps_distribution": "gamma", "request_seed": seed})
    a, b = _pairs(
        tmp_path,
        a_extra=lambda i: {"Workload Options": options(i), "Workload Options Sha256": "x"},
        b_extra=lambda i: {
            "Workload Options": options(i + 10).replace("gamma", "constant"),
            "Workload Options Sha256": "y",
        },
    )
    assert compare.main(["--a", *a, "--b", *b]) == 2
    err = capsys.readouterr().err
    assert "qps_distribution: A 'gamma' vs B 'constant'" in err and "request_seed" not in err


def test_r1_verdicts_come_from_the_paired_deltas(tmp_path) -> None:
    """Runs differ a lot from pair to pair (100, 200, 300) but B is 10% above A in each: a per-side spread would call
    that noise; paired, the change is plain."""
    ident = {c: "1" for c, _ in compare.IDENTITY}
    a = [dict(ident, **{"Latency Per Token": v}) for v in ("100", "200", "300")]
    b = [dict(ident, **{"Latency Per Token": v}) for v in ("110", "220", "330")]
    (result,) = compare.compare(a, b, 0.03)
    assert result["verdict"] == "worse" and result["pairs"] == 3 and result["noise_from"] == "--min-noise"
    assert result["pair_deltas"] == pytest.approx([0.1, 0.1, 0.1]) and result["delta"] == pytest.approx(0.1)
    # pairs that disagree set the floor: the spread of the per-pair deltas
    b = [dict(ident, **{"Latency Per Token": v}) for v in ("110", "200", "360")]  # +10%, 0%, +20%
    (result,) = compare.compare(a, b, 0.03)
    assert result["noise"] == pytest.approx(0.2) and result["noise_from"] == "pair spread"
    assert result["verdict"] == "inconclusive (noisy pairs)"
    # a pair is a pair: swapping B's order changes the deltas
    (swapped,) = compare.compare(a, b[::-1], 0.03)
    assert swapped["pair_deltas"] != result["pair_deltas"]


def test_r1_the_output_lists_every_pairs_delta(tmp_path) -> None:
    a, b = _pairs(
        tmp_path,
        a_extra=lambda i: {"P50 Total Latency": "100"},
        b_extra=lambda i: {"P50 Total Latency": str(110 + 10 * i)},
    )
    md = tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", *b, "--markdown", str(md)]) == 0
    row = next(l for l in md.read_text().splitlines() if l.startswith("| P50 Total Latency |"))
    assert "+10.0%, +20.0%, +30.0%" in row and "| 3 |" in row


def test_compare_reports_deltas_against_the_noise_floor(tmp_path, capsys) -> None:
    a = [
        _summary(
            tmp_path / f"a{i}.csv",
            [{"Request Seed": str(i), "P50 Time To First Token": ttft, "Qps": qps, "Measured Hit Rate": "0.6"}],
        )
        for i, (ttft, qps) in enumerate([("100", "39.0"), ("110", "39.0"), ("105", "39.0")])
    ]
    b = [
        _summary(
            tmp_path / f"b{i}.csv",
            [
                {
                    "Request Seed": str(i),
                    "P50 Time To First Token": "150",
                    "P99 Total Latency": "890",
                    "Qps": "39.2",
                    "Measured Hit Rate": "0.7",
                }
            ],
        )
        for i in range(3)
    ]
    out = tmp_path / "out.json"
    md = tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out), "--markdown", str(md)]) == 0
    by_metric = {m["metric"]: m for m in json.loads(out.read_text())["metrics"]}
    ttft = by_metric["P50 Time To First Token"]
    assert ttft["verdict"] == "worse" and ttft["noise"] == pytest.approx(0.5 - 150 / 110 + 1)  # +50%, +36%, +43%
    assert ttft["noise_from"] == "pair spread" and by_metric["Qps"]["noise_from"] == "--min-noise"
    assert by_metric["Qps"]["verdict"] == "same"  # +0.5% against a 3% floor
    assert by_metric["P99 Total Latency"]["verdict"] == "same"  # -1.1%
    assert by_metric["Measured Hit Rate"]["verdict"] == "better" and by_metric["Measured Hit Rate"]["delta"] == (
        pytest.approx(0.1)
    )
    text = md.read_text()
    assert "| P50 Time To First Token |" in text and "(pair spread)" in text and "| 3 |" in text


def test_g3_a_better_or_worse_verdict_needs_three_repeats_per_side(tmp_path) -> None:
    ident = {c: "1" for c, _ in compare.IDENTITY}
    one = lambda v: dict(ident, **{"Latency Per Token": v})
    for repeats in (1, 2):
        verdicts = compare.compare([one("100")] * repeats, [one("110")] * repeats, 0.03)
        assert verdicts[0]["verdict"] == "inconclusive (too few repeats)", repeats
    assert compare.compare([one("100")] * 3, [one("110")] * 3, 0.03)[0]["verdict"] == "worse"
    assert compare.compare([one("100")] * 3, [one("110")] * 2, 0.03)[0]["verdict"] == "inconclusive (too few repeats)"
    assert compare.compare([one("100")] * 2, [one("110")] * 2, 0.03, min_repeats=2)[0]["verdict"] == "worse"
    assert compare.compare([one("100")] * 3, [one("101")] * 3, 0.03)[0]["verdict"] == "same"  # inside the floor
    # with too few repeats "same" and "different" are claims too: every verdict reads inconclusive, the delta stays
    few = compare.compare([one("100")] * 2, [one("101")] * 2, 0.03)[0]
    assert few["verdict"] == "inconclusive (too few repeats)" and few["delta"] == pytest.approx(0.01)
    zero = lambda v: dict(ident, **{"Requests Failed": v})
    assert compare.compare([zero("0")], [zero("9")], 0.03)[0]["verdict"] == "inconclusive (too few repeats)"


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
    by = {r["metric"]: r for r in compare.compare(zero_a, zero_b, 0.03, min_repeats=1)}
    assert by["Requests Failed"]["verdict"] == "different" and by["Requests Failed"]["delta"] == 250
    assert by["Requests Failed"]["delta_kind"] == "abs"  # the difference, not a relative change from 0
    assert by["P50 Total Latency"]["verdict"] == "same"
    # a noisy side swallows the delta: not "same"
    noisy = [dict(ident, **{"Latency Per Token": v}) for v in ("100", "160")]
    steady = [dict(ident, **{"Latency Per Token": v}) for v in ("131", "132")]
    verdict = {r["metric"]: r for r in compare.compare(noisy, steady, 0.03, min_repeats=2)}["Latency Per Token"][
        "verdict"
    ]
    assert verdict == "inconclusive (noisy pairs)"
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


# --- fix round 3 ---------------------------------------------------------------------------------------------------


def test_g5_a_client_stall_that_shifted_the_schedule_fails_the_delivery_check() -> None:
    """A stalled client shifts the rest of the schedule later and its intended times move with it, so the rate against
    them still reads 1: the total shift is the stall, and fails the check above max(5 s, 1% of the window)."""
    options = _run_options(MIX, qps=3.0, qps_max_lateness=0.5, qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    pacer = load_test.FixedQPSPacer(3.0, "constant")
    pacer.start = time.monotonic()
    on_time = [i / 3 for i in range(60)]
    _timeline(on_time, on_time)
    assert delivery_entries(options, pacer, 0.0)[1] == []
    assert pacer.resync(0, 1.0)  # a 1 s hiccup: under the bar
    assert delivery_entries(options, pacer, 0.0)[1] == []
    assert pacer.resync(1, 20.0)  # the 20 s SIGSTOP of the joint check: the pacer's own shift path
    entries, failures = delivery_entries(options, pacer, 0.0)
    assert pacer.shifted_seconds == pytest.approx(21.0)
    assert entries["delivered_over_intended"] == pytest.approx(1.0) and entries["late_send_share"] == 0.0
    assert entries["delivery_check"] == "fail" and any("schedule moved 21.0 s later" in f for f in failures)


def test_g5_a_replays_shifted_schedule_fails_too() -> None:
    entries = tuple(load_test.ScheduleEntry(offset=i * 0.5, class_index=0, index=i) for i in range(100))
    pacer = ReplayPacer(entries, "s.jsonl")
    pacer.start = time.monotonic()
    pacer.sent_rows = 100
    options = _run_options(MIX, qps=None, replay_schedule="s.jsonl", qps_min_delivered=0.95, max_late_share=0.05)
    options.max_fail_ratio, options.fail_on_delivery = 0.01, True
    sent = [e.offset for e in entries]
    _timeline(sent, sent)
    assert pacer.resync(0, 30.0)
    assert any("schedule moved 30.0 s later" in f for f in delivery_entries(options, pacer, 0.0)[1])


# --- fix round 4 ---------------------------------------------------------------------------------------------------


def test_g7_the_bin_widths_are_locusts_rounding() -> None:
    """compare's bins are the ones Locust rounds response times into: confirmed against the installed Locust."""
    from locust.stats import StatsEntry

    entry = StatsEntry(None, "x", "GET")
    for value in range(1, 30000):
        entry._log_response_time(value)
    keys = sorted(entry.response_times)
    for low, high in ((1, 99), (200, 990), (1100, 9900), (11000, 29000)):
        inside = [k for k in keys if low <= k <= high]
        gaps = {b - a for a, b in zip(inside, inside[1:])}
        assert gaps == {compare.bin_width_ms(low)}, (low, gaps)  # the keys are one bin apart there


def _tail_pairs(tmp_path, column, a_value, b_value):
    a, b = _pairs(tmp_path, a_extra=lambda i: {column: str(a_value + i)}, b_extra=lambda i: {column: str(b_value + i)})
    out = tmp_path / "out.json"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out)]) == 0
    return {m["metric"]: m for m in json.loads(out.read_text())["metrics"]}[column]


def test_g7_a_percentile_one_bin_apart_is_not_a_change(tmp_path) -> None:
    # P90 TTFT 1500 vs 1400: one 100 ms bin; the floor is at least a bin over the value (about 7%), not 3%
    result = _tail_pairs(tmp_path, "P90 Time To First Token", 1500, 1400)
    assert result["verdict"] == "same" and result["noise_from"] == "percentile bin 100 ms"
    assert result["noise"] == pytest.approx(100 / 1400, rel=0.01)
    # two bins apart is a change; the floor is reported
    assert _tail_pairs(tmp_path, "P90 Time To First Token", 1500, 1200)["verdict"] == "better"
    # below 100 ms the bin is 1 ms: a 10% change at 50 ms is a change
    assert _tail_pairs(tmp_path, "P50 Total Latency", 50, 55)["verdict"] == "worse"
    # what is not a percentile has no bin
    assert _tail_pairs(tmp_path, "Latency Per Token", 1500, 1400)["verdict"] == "better"


def test_g8_the_summary_records_the_target_host(tmp_path) -> None:
    options = _run_options(
        MIX,
        qps=2.0,
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
    load_test.InitTracker.logging_params = {}
    load_test.FixedQPSPacer._instance = None
    _timeline([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    _fire_quitting(options, tmp_path, host="http://server-a:8000")
    (row,) = list(csv.DictReader(open(tmp_path / "summary.csv")))
    assert row["Target Host"] == "http://server-a:8000"
    assert "Target Host" not in {c for c, _ in compare.IDENTITY}  # what A and B differ by: not the workload


SAME_ENDPOINT = "same endpoint: B reuses A's cache unless the server was restarted between runs; not a valid A/A"


def test_g8_compare_warns_loudly_when_both_sides_hit_one_host(tmp_path, capsys) -> None:
    host = lambda i: {"Target Host": "http://Server-A:8000/"}
    a, b = _pairs(tmp_path, a_extra=host, b_extra=lambda i: {"Target Host": "http://server-a:8000"})
    out, md = tmp_path / "out.json", tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out), "--markdown", str(md)]) == 0
    captured = capsys.readouterr()
    assert SAME_ENDPOINT in captured.err and SAME_ENDPOINT in captured.out  # in every output
    assert SAME_ENDPOINT in md.read_text()
    assert SAME_ENDPOINT in json.loads(out.read_text())["warnings"][0]
    # two servers: nothing to warn about
    a, b = _pairs(tmp_path, a_extra=host, b_extra=lambda i: {"Target Host": "http://server-b:8000"})
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out), "--markdown", str(md)]) == 0
    captured = capsys.readouterr()
    assert "same endpoint" not in captured.err + captured.out + md.read_text()
    assert json.loads(out.read_text())["warnings"] == []


def _validate(**overrides):
    options = argparse.Namespace(
        shape_mix=json.dumps(MIX), qps=None, burst=None, ramping_time=None, provider=None, num_users=None
    )
    for name, value in overrides.items():
        setattr(options, name, value)
    host = options.__dict__.pop("host", None)
    load_test.validate_run_options(options, host)


def test_g9_a_warmup_without_a_provider_fails_at_init_not_after_the_prefix_build() -> None:
    with pytest.raises(ValueError, match="a warm-up needs --provider"):
        _validate(qps=5.0, warmup_requests=10)
    with pytest.raises(ValueError, match="a warm-up needs --provider"):
        _validate(qps=5.0, warmup_hit_rate=0.5, warmup_cap_s=30.0)
    _validate(qps=5.0, warmup_requests=10, provider="openai")
    _validate(qps=5.0, warmup_requests=10, host="https://api.fireworks.ai")  # the host names it
    _validate(qps=5.0)  # no warm-up: no provider needed here
    _validate(qps=5.0, warmup_requests=10, provider="openai", no_warmup=True)


def test_g9_a_replay_with_warmup_rows_needs_the_provider_but_lead_ins_alone_do_not(tmp_path) -> None:
    def schedule(rows):
        path = tmp_path / "s.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(path)

    options = dict(shape_mix=json.dumps(REPLAY_MIX), replay_pods=1, replay_pod_index=0)
    warm = [_row(-1.0, "s1", 700, 400, warmup=True), _row(0.0, "s1", 900, 700)]
    with pytest.raises(ValueError, match="a warm-up needs --provider"):
        _validate(replay_schedule=schedule(warm), **options)
    _validate(replay_schedule=schedule(warm), no_warmup=True, **options)
    lead_in = [_row(-1.0, "s1", 700, 400, lead_in=True), _row(0.0, "s1", 900, 700)]
    _validate(replay_schedule=schedule(lead_in), **options)  # as before: only a warning when they can't be sent


def test_g9_the_option_checks_the_users_made_late_now_fail_at_init() -> None:
    with pytest.raises(ValueError, match="--replay-schedule needs --shape-mix"):
        _validate(shape_mix=None, replay_schedule="s.jsonl")
    with pytest.raises(ValueError, match="drop --qps, --burst and --ramping-time"):
        _validate(replay_schedule="s.jsonl", qps=3.0)
    with pytest.raises(ValueError, match="--prefix-fill needs --shape-mix"):
        _validate(shape_mix=None, prefix_fill=True)
    with pytest.raises(ValueError, match="--prefix-fill sends at"):
        _validate(prefix_fill=True, qps=3.0)
    with pytest.raises(ValueError, match="--prefix-fill-concurrency"):
        _validate(prefix_fill=True, prefix_fill_concurrency=0)
    with pytest.raises(ValueError, match="standalone"):
        _validate(prefix_reuse_probe=True, qps=3.0)
    with pytest.raises(ValueError, match="Burst and QPS"):
        _validate(qps=3.0, burst=5.0)
    with pytest.raises(ValueError, match="Burst and ramping"):
        _validate(ramping_time=5.0, burst=5.0)
    with pytest.raises(ValueError, match="requires -u"):
        _validate(ramping_time=5.0)
    with pytest.raises(ValueError, match="--qps-cv2 must be > 0"):  # the pacer's own check, built early
        _validate(qps=3.0, qps_distribution="gamma", qps_cv2=-1.0)
    with pytest.raises(ValueError, match="--shape-mix builds its own prompts"):
        _validate(dataset="other")
    with pytest.raises(ValueError, match="-pcml"):
        _validate(prompt_cache_max_len=10)


def test_g9_the_init_listener_exits_before_anything_is_built(caplog) -> None:
    options = argparse.Namespace(shape_mix=json.dumps(MIX), qps=5.0, provider=None, warmup_requests=10)
    with caplog.at_level("ERROR", logger="load_test"), pytest.raises(SystemExit):
        load_test._validate_run_options_on_init(SimpleNamespace(parsed_options=options, host="http://x:8000"))
    assert "a warm-up needs --provider" in caplog.text


def test_g10_the_probe_prints_the_flags_and_exits_0_when_it_measured_something(tmp_path, monkeypatch, capsys) -> None:
    probe = load_test.PrefixReuseProbe
    monkeypatch.setattr(probe, "points", [(1000, 872), (2000, 1872)])
    monkeypatch.setattr(probe, "overhang", 128)
    monkeypatch.setattr(probe, "boundary", 0)
    monkeypatch.setattr(probe, "failed", 0)
    assert load_test.prefix_reuse_probe_flags() == "--prefix-reuse-overhang 128"
    monkeypatch.setattr(probe, "boundary", 2048)
    assert load_test.prefix_reuse_probe_flags() == "--prefix-reuse-overhang 128 --prefix-restore-boundary 2048"
    assert (
        load_test.prefix_reuse_probe_entries()["prefix_reuse_probe_suggested"] == load_test.prefix_reuse_probe_flags()
    )
    options = _run_options(MIX, qps=None, prefix_reuse_probe=True, summary_file=None, embeddings=False, rerank=False)
    options.max_fail_ratio = 0.01
    monkeypatch.setattr(load_test.InitTracker, "logging_params", {})
    assert _fire_quitting(options, tmp_path, scored=False) == 0  # no scored request, by design: not a failure
    monkeypatch.setattr(probe, "points", [])  # measured nothing
    assert _fire_quitting(options, tmp_path, scored=False) == 1
    monkeypatch.setattr(probe, "points", [(1000, 872)])
    monkeypatch.setattr(probe, "failed", 1)  # a failed measurement
    assert _fire_quitting(options, tmp_path, scored=False) == 1
    options.prefix_reuse_probe = False  # any other run with no scored request is still a failure
    monkeypatch.setattr(probe, "failed", 0)
    assert _fire_quitting(options, tmp_path, scored=False) == 1


# --- iterating on changes ------------------------------------------------------------------------------------------


def test_h1_recipe_prints_each_pairs_flags_and_the_identity_to_match(tmp_path, capsys) -> None:
    a, _ = _pairs(tmp_path)  # a baseline: seeds 100..102, warm-up counts 50..52
    out = tmp_path / "recipe.json"
    assert compare.main(["--recipe", *a, "--json", str(out)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[:3] == [
        "pair 1: --request-seed 100 --warmup-requests 50",
        "pair 2: --request-seed 101 --warmup-requests 51",
        "pair 3: --request-seed 102 --warmup-requests 52",
    ]
    assert printed[3] == (
        "must match: workload file sha256 abc, workload options opts, rate (target QPS) 40.0, llm_bench version 1"
    )
    result = json.loads(out.read_text())
    assert [p["flags"] for p in result["pairs"]] == [
        "--request-seed 100 --warmup-requests 50",
        "--request-seed 101 --warmup-requests 51",
        "--request-seed 102 --warmup-requests 52",
    ]
    assert result["identity"] == {
        "Workload Sha256": "abc",
        "Workload Options Sha256": "opts",
        "Target Qps": "40.0",
        "Llm Bench Version": "1",
    }
    # the rows of one file are the pairs too
    one_file = _summary(
        tmp_path / "all.csv",
        [{"Request Seed": str(i), "Warmup Requests": str(7 + i)} for i in range(2)],
    )
    assert compare.main(["--recipe", one_file]) == 0
    assert "pair 2: --request-seed 1 --warmup-requests 8" in capsys.readouterr().out


def test_h1_recipe_refuses_a_file_that_is_not_a_valid_baseline(tmp_path, capsys) -> None:
    same_seed, _ = _pairs(tmp_path, a_extra=lambda i: {"Request Seed": "5"})
    assert compare.main(["--recipe", *same_seed]) == 2
    assert "not a valid baseline" in capsys.readouterr().err
    missing, _ = _pairs(tmp_path, a_extra=lambda i: {"Request Seed": ""} if i == 1 else {})
    assert compare.main(["--recipe", *missing]) == 2
    assert "run 2: request seed is missing" in capsys.readouterr().err
    no_count, _ = _pairs(tmp_path, a_extra=lambda i: {"Warmup Requests": ""} if i == 0 else {})
    assert compare.main(["--recipe", *no_count]) == 2
    assert "run 1: warm-up requests is missing" in capsys.readouterr().err
    mixed, _ = _pairs(tmp_path, a_extra=lambda i: {"Workload Sha256": "other"} if i == 2 else {})
    assert compare.main(["--recipe", *mixed]) == 2
    assert "workload file sha256 differs between runs" in capsys.readouterr().err
    out = tmp_path / "recipe.json"
    assert compare.main(["--recipe", *mixed, "--json", str(out)]) == 2 and not out.exists()  # nothing half-written


def test_h1_recipe_and_compare_are_separate_modes(tmp_path) -> None:
    a, b = _pairs(tmp_path)
    for argv in (["--recipe", *a, "--a", *a, "--b", *b], ["--a", *a], []):
        with pytest.raises(SystemExit):
            compare.main(argv)


def test_h2_restarted_between_runs_turns_the_same_endpoint_warning_into_a_note(tmp_path, capsys) -> None:
    host = {"Target Host": "http://server-a:8000"}
    a, b = _pairs(tmp_path, a_extra=lambda i: host, b_extra=lambda i: host)
    out, md = tmp_path / "out.json", tmp_path / "out.md"
    assert compare.main(["--a", *a, "--b", *b, "--json", str(out), "--markdown", str(md)]) == 0
    assert SAME_ENDPOINT in capsys.readouterr().err  # without the flag the warning stays
    assert (
        compare.main(["--a", *a, "--b", *b, "--restarted-between-runs", "--json", str(out), "--markdown", str(md)]) == 0
    )
    captured = capsys.readouterr()
    note = "same endpoint; the operator states the server was restarted before every run"
    assert note in captured.err and note in captured.out and note in md.read_text()
    assert note in json.loads(out.read_text())["notes"][0] and json.loads(out.read_text())["warnings"] == []
    assert "WARNING" not in captured.err + captured.out + md.read_text()
    # on two servers the flag changes nothing
    a, b = _pairs(tmp_path, a_extra=lambda i: host, b_extra=lambda i: {"Target Host": "http://server-b:8000"})
    assert compare.main(["--a", *a, "--b", *b, "--restarted-between-runs", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["notes"] == [] and json.loads(out.read_text())["warnings"] == []


# --- the replay warm-up's reported hit rate -----------------------------------------------------------------------


def test_g11_the_reported_hit_rate_is_the_one_the_stop_used(monkeypatch, tmp_path) -> None:
    """The warm-up stopped on a target with requests still in flight; when they completed (cold, large) the window
    after them read 0.897 against a 0.9745 target while the reason said target. The stop's own window is reported
    (and the end figure beside it)."""
    cached = {0: 700, 1: 0}  # row 0 is a hit and completes in time; row 1 is cold and slow, still in flight at the stop
    delay = {0: 0.25, 1: 0.6}

    def fake(self, request, one_token=False):
        time.sleep(delay.get(request.schedule_row, 0.0))
        return (700, cached.get(request.schedule_row, 0))

    monkeypatch.setattr(UnscoredSender, "post", fake)
    plan = WarmupPlan("target", hit_rate=0.95, tolerance=0.02, window_s=0.3, cap_s=None)
    _replay_run(tmp_path, "http://127.0.0.1:9", plan)
    assert Warmup.reason == "target" and Warmup.requests == 2
    assert Warmup.hit_rate == pytest.approx(1.0) and Warmup.hit_rate >= Warmup.target - 0.02  # what the stop saw
    assert Warmup.hit_rate_end == pytest.approx(0.0)  # the window once the cold request was in
    assert Warmup.summary_entries()["warmup_hit_rate_end"] == pytest.approx(0.0)


def test_g11_a_warmup_that_ran_out_of_rows_is_exhausted_and_reports_the_end_rate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(UnscoredSender, "post", lambda self, request, one_token=False: (700, 100))  # ~14% hits
    plan = WarmupPlan("target", hit_rate=0.95, tolerance=0.02, window_s=0.3, cap_s=None)
    _replay_run(tmp_path, "http://127.0.0.1:9", plan)
    assert Warmup.reason == "exhausted" and Warmup.requests == 5
    assert Warmup.hit_rate == pytest.approx(100 / 700) == pytest.approx(Warmup.hit_rate_end)
