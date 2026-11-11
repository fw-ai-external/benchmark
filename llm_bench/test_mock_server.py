import random

from mock_server import PrefixCacheEngine, _PrefixIndex


def _brute(ids, seen):
    best = 0
    for prev in seen:
        n = min(len(ids), len(prev))
        k = 0
        while k < n and ids[k] == prev[k]:
            k += 1
        best = max(best, k)
    return best


def test_prefix_index_matches_brute_force() -> None:
    """The block-hash index finds the same longest common prefix as scanning every stored prompt, on prompts that
    share pool prefixes of many lengths and diverge anywhere (mid-block, at a block edge, at token 0)."""
    rng = random.Random(3)
    pools = [[rng.randrange(1000) for _ in range(rng.randrange(1, 900))] for _ in range(12)]
    index, seen = _PrefixIndex(), []
    for _ in range(400):
        pool = rng.choice(pools)
        cut = rng.randrange(len(pool) + 1)
        ids = pool[:cut] + [rng.randrange(1000, 2000) for _ in range(rng.randrange(1, 200))]
        assert index.longest(ids) == _brute(ids, seen)
        index.add(ids)
        seen.append(ids)


class _Tok:
    """A tokenizer stand-in: one id per word; the chat template wraps the last message in two marker ids, and
    returns a dict as newer transformers do."""

    def encode(self, text, add_special_tokens=False):
        return [hash(w) % 50_000 for w in text.split()]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return {"input_ids": [7, 8] + self.encode(messages[-1]["content"]) + [9], "attention_mask": []}


def test_engine_reuse_rule_window_boundary_and_routing() -> None:
    words = " ".join(f"w{i}" for i in range(3000))
    tail = lambda k: f" tail{k} x y z"
    # token-exact with a 130-token window
    engine = PrefixCacheEngine(_Tok(), window=130)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    assert engine.serve(engine.token_ids({"prompt": words + tail(1)})) == 3000 - 130
    # 2,048-token restore blocks
    engine = PrefixCacheEngine(_Tok(), window=128, boundary=2048)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    assert engine.serve(engine.token_ids({"prompt": words + tail(1)})) == 2048 - 128
    # chat: the template's dict return is unpacked (it used to yield its 2 keys as the prompt)
    engine = PrefixCacheEngine(_Tok(), window=0)
    chat = {"messages": [{"role": "user", "content": words}]}
    assert len(engine.token_ids(chat)) == 3003
    # several caches: affinity keeps a key on one cache; without it a request can miss the warm prompt
    engine = PrefixCacheEngine(_Tok(), window=0, caches=8, affinity=True)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}), "k1")
    assert all(engine.serve(engine.token_ids({"prompt": words + tail(i)}), "k1") == 3000 for i in range(1, 6))
    engine = PrefixCacheEngine(_Tok(), window=0, caches=8, affinity=False, seed=1)
    engine.serve(engine.token_ids({"prompt": words + tail(0)}))
    hits = [engine.serve(engine.token_ids({"prompt": words + tail(i)})) for i in range(1, 9)]
    assert 0 in hits  # landed on a cache that hadn't seen it


def test_preflight_runner_reads_a_tier_plan_and_picks_the_scored_step() -> None:
    """D-117: the runner takes task 07's plan as written (traffic_generator_create_args) and scores the step at
    the plan's target rate, after the fill; the measured probe comes from record.reuse_probe."""
    from preflight_run import plan_steps, scored_step

    mix = '{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":900,"cached_prompt_tokens":500}]}'

    def create(gid, conc, dur, extra, qps=None):
        args = ["traffic-generator", "create", "--target", "dev-pod/x", "--traffic-generator-id", gid]
        args += ["--concurrency", str(conc), "--spawn-rate", str(conc), "--duration", dur]
        if qps is not None:
            args += ["--qps", str(qps)]
        return args + ["--image-tag", "in-pod", "--extra-args", extra]

    plan = {
        "record": {"target_qps": 3.63164, "reuse_probe": {"overhang": 130, "boundary": 0}},
        "traffic_generator_create_args": {
            "fill-fill": create("f", 1, "3600s", f"--shape-mix '{mix}' --prefix-fill --prefix-reuse-overhang 130"),
            "warm-warm": create("w", 599, "600s", f"--shape-mix '{mix}' --qps-distribution gamma", 3.63164),
            "probe-p1": create("p1", 300, "180s", f"--shape-mix '{mix}'", 1.81582),
            "ramp-3": create("r3", 599, "900s", f"--shape-mix '{mix}' --qps-cv2 2.75", 3.63164),
        },
    }
    steps = plan_steps(plan)
    assert [s[0] for s in steps] == ["fill-fill", "warm-warm", "probe-p1", "ramp-3"]
    name, conc, rate, seconds, extra = scored_step(plan, steps)
    assert (name, conc, rate, seconds) == ("ramp-3", 599, 3.63164, 900.0)  # the LAST step at the target rate
    assert extra[:2] == ["--shape-mix", mix] and "--qps-cv2" in extra  # the plan's exact args, unquoted
    assert "--prefix-fill" in steps[0][4]
    assert scored_step(plan, steps, "probe-p1")[0] == "probe-p1"


def test_preflight_requires_a_mode_and_checks_the_logged_paths(tmp_path) -> None:
    """Every step must name --chat or --no-chat, and the requests the mock logged for it must hit that path."""
    from preflight_run import path_mismatches, step_path

    assert step_path(("s", 1, None, None, ["--shape-mix", "x", "--no-chat"])) == "/v1/completions"
    assert step_path(("s", 1, None, None, ["--no-chat", "--chat"])) == "/v1/chat/completions"  # the last wins
    assert step_path(("s", 1, None, None, ["--shape-mix", "x"])) is None
    assert path_mismatches(str(tmp_path / "missing.requests"), 0, "/v1/completions") == (
        0,
        {},
    )  # 0 seen: main() fails it
    log = tmp_path / "mock.requests"
    log.write_text("1.0 900 0 0 /v1/completions\n2.0 900 400 0 /v1/completions\n3.0 900 400 0 /v1/chat/completions\n")
    assert path_mismatches(str(log), 0, "/v1/completions") == (3, {"/v1/chat/completions": 1})
    assert path_mismatches(str(log), 0, "/v1/chat/completions") == (3, {"/v1/completions": 2})
    assert path_mismatches(str(log), 2, "/v1/chat/completions") == (1, {})  # only this step's lines
    # A log from before the path column counts as a mismatch, not a silent pass
    log.write_text("1.0 900 0 0\n")
    assert path_mismatches(str(log), 0, "/v1/completions") == (1, {"?": 1})


def test_preflight_refuses_a_layout_that_doesnt_match_the_mock(tmp_path, capsys) -> None:
    """The layout overhang must equal what the mock withholds (its window, less its chat head in chat), or the
    verdict measures the mismatch: P9's 2-log pre-flight laid prefixes out +128 on a window-0 mock and "failed"
    its short classes by -67%. The pre-flight refuses before starting; --allow-mismatch is for experiments."""
    import json

    from preflight_run import applied_overhang, expected_overhang, main, summary_mismatch

    chat = ("s", 1, None, None, ["--shape-mix", "x", "--chat", "--prefix-reuse-overhang", "102"])
    completions = ("s", 1, None, None, ["--shape-mix", "x", "--no-chat", "--prefix-reuse-overhang", "128"])
    assert expected_overhang(chat, 128, 26) == 102 and expected_overhang(completions, 128, 26) == 128
    assert applied_overhang(chat) == 102
    # resolved by llm_bench from the served model's config when the step passes no overhang
    config = tmp_path / "dsv4"
    config.mkdir()
    (config / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "sliding_window": 128}))
    assert applied_overhang(("s", 1, None, None, ["--no-chat", "--served-model-config", str(config)])) == 128

    mix = '{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":900,"cached_prompt_tokens":500}]}'
    plan = {
        "record": {"target_qps": 1.0},
        "traffic_generator_create_args": {
            "ramp-1": [
                "traffic-generator",
                "create",
                "--concurrency",
                "2",
                "--duration",
                "10s",
                "--qps",
                "1.0",
                "--extra-args",
                f"--shape-mix '{mix}' --no-chat --prefix-reuse-overhang 128",
            ],
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert main([str(path), "--out", str(tmp_path / "out"), "--window", "0"]) == 2  # before any mock starts
    assert "refusing" in capsys.readouterr().out
    assert not (tmp_path / "out" / "mock.requests").exists()

    summary = tmp_path / "s.csv"
    summary.write_text("Reuse Overhang Tokens,Chat Head Tokens,Reuse Restore Boundary\n128,20,0\n")
    assert "overhang 128" in summary_mismatch(str(summary), completions, 0, 0, "tok")
    assert summary_mismatch(str(summary), completions, 128, 0, "tok") is None
    headed = ("s", 1, None, None, ["--chat", "--tokenizer", "tok", "--prefix-reuse-overhang", "102"])
    summary.write_text("Reuse Overhang Tokens,Chat Head Tokens,Reuse Restore Boundary\n102,20,0\n")
    assert "chat head of 20" in summary_mismatch(str(summary), headed, 128, 26, "tok")
    assert summary_mismatch(str(summary), headed, 128, 26, "another-tok") is None  # a counting-tokenizer head


def test_preflight_mock_window_nets_a_chat_probe_and_leads_with_the_mock_fix(tmp_path) -> None:
    """N10-146-5: a chat probe's overhang (102) is net of the engine's head, so with --mock-chat-head 26 the mock's raw
    window is 128 and the probe's own 102 is accepted, not refused for 76. N10-146-6: the advice fits the mock to
    the layout first. N10-146-7: an --allow-mismatch run is marked and never exits as a plan verdict."""
    from preflight_run import expected_overhang, mismatch_message, mock_window, probe_mode

    chat = ("s", 1, None, None, ["--chat", "--prefix-reuse-overhang", "102"])
    probe = {"overhang": 102, "boundary": 0, "source": "probe-chat-fresh-102"}
    assert probe_mode(probe) == "chat" and probe_mode({"source": "probe-completions-fresh-129"}) == "completions"
    window = mock_window(probe, None, 26)
    assert window == 128 and expected_overhang(chat, window, 26) == 102  # the correct 102 passes
    assert mock_window({"overhang": 129, "source": "probe-completions-x"}, None, 26) == 129
    assert mock_window({"overhang": 102}, None, 26) is None  # mode unknown with a head: ambiguous, refused
    assert mock_window({"overhang": 102}, None, 0) == 102 and mock_window(probe, 120, 26) == 120
    message = mismatch_message(chat, 102, 76, 102, 26)
    assert message.index("--window 128 --mock-chat-head 26") < message.index("--prefix-reuse-overhang")


def test_preflight_sets_an_earlier_summary_aside(tmp_path) -> None:
    """A reused --out dir must not hold two runs' rows (llm_bench appends): P9's re-run was misread from v5's row.
    The earlier summary is moved aside, never deleted, and every summary row carries the run's id."""
    import re

    from load_test import RUN_ID
    from preflight_run import set_aside

    summary = tmp_path / "ramp-3.summary.csv"
    assert set_aside(str(summary)) is None  # nothing to move
    summary.write_text("old\n")
    assert set_aside(str(summary)) == f"{summary}.prev" and not summary.exists()
    summary.write_text("older\n")
    assert set_aside(str(summary)) == f"{summary}.prev2"  # never overwrites an earlier .prev
    assert (tmp_path / "ramp-3.summary.csv.prev").read_text() == "old\n"
    assert re.fullmatch(r"\d{8}T\d{6}Z-\d+", RUN_ID)


def test_preflight_checks_the_restore_boundary_and_a_summary_that_cant_vouch(tmp_path, capsys) -> None:
    """The restore boundary is compared like the overhang (a plan laid out for 2048-token blocks on a token-exact
    mock scores the mock's rounding), and a summary that's unreadable or lacks its reuse columns can't be scored."""
    import json

    from preflight_run import applied_reuse, main, summary_mismatch

    step = ("s", 1, None, None, ["--no-chat", "--prefix-reuse-overhang", "128", "--prefix-restore-boundary", "2048"])
    assert applied_reuse(step) == (128, 2048)
    summary = tmp_path / "s.csv"
    summary.write_text("Reuse Overhang Tokens,Reuse Restore Boundary\n128,2048\n")
    assert summary_mismatch(str(summary), step, 128, 0, "tok", boundary=2048) is None
    assert "restore boundary 2048" in summary_mismatch(str(summary), step, 128, 0, "tok", boundary=1)
    summary.write_text("Reuse Overhang Tokens,Reuse Restore Boundary\n128,0\n")  # 0 and the mock's 1: token-exact
    assert summary_mismatch(str(summary), step, 128, 0, "tok", boundary=1) is None
    summary.write_text("Num Requests\n10\n")
    assert "no Reuse Overhang Tokens" in summary_mismatch(str(summary), step, 128, 0, "tok")
    summary.write_text("")
    assert "can't be read" in summary_mismatch(str(summary), step, 128, 0, "tok")
    assert summary_mismatch(str(tmp_path / "missing.csv"), step, 128, 0, "tok") is None  # the caller reports that

    mix = '{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":900,"cached_prompt_tokens":500}]}'
    extra = f"--shape-mix '{mix}' --no-chat --prefix-reuse-overhang 128 --prefix-restore-boundary 2048"
    plan = {
        "record": {"target_qps": 1.0},
        "traffic_generator_create_args": {
            "ramp-1": [
                "traffic-generator",
                "create",
                "--concurrency",
                "2",
                "--duration",
                "10s",
                "--qps",
                "1.0",
                "--extra-args",
                extra,
            ],
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert main([str(path), "--out", str(tmp_path / "out"), "--window", "128", "--boundary", "1"]) == 2
    assert "restore boundary 2048" in capsys.readouterr().out
    assert not (tmp_path / "out" / "mock.requests").exists()  # refused before any mock started


def test_preflight_tokenizer_check_compares_content_and_blank_boundary_refused(tmp_path) -> None:
    """The tokenizer check compares what the tokenizers are (tokenizer.json's hash), so a local copy of the same
    tokenizer under another name matches and a different tokenizer sharing a name doesn't; unreadable ones fall back
    to the names, saying so. A text step's summary with a blank Reuse Restore Boundary can't vouch for its layout;
    an embeddings step carries no reuse columns at all."""
    from preflight_run import summary_mismatch, tokenizer_mismatch

    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d, content in ((a, '{"model": 1}'), (b, '{"model": 1}'), (c, '{"model": 2}')):
        d.mkdir()
        (d / "tokenizer.json").write_text(content)
    assert tokenizer_mismatch(str(a), str(a)) is None
    assert tokenizer_mismatch(str(a), str(b)) is None  # same tokenizer, two names
    assert "different tokenizer.json" in tokenizer_mismatch(str(a), str(c))
    assert "compared by name only" in tokenizer_mismatch(str(a), str(tmp_path / "missing"))

    step = ("s", 1, None, None, ["--no-chat", "--prefix-reuse-overhang", "128"])
    summary = tmp_path / "s.csv"
    summary.write_text("Reuse Overhang Tokens,Reuse Restore Boundary\n128,\n")
    assert "no Reuse Restore Boundary" in summary_mismatch(str(summary), step, 128, 0, "tok")
    embeddings = ("e", 1, None, None, ["--embeddings"])
    summary.write_text("Num Requests\n10\n")
    assert summary_mismatch(str(summary), embeddings, 128, 0, "tok") is None


def test_preflight_relative_out_resolves_against_the_caller_not_llm_bench(tmp_path) -> None:
    """D-315: a relative --out must not depend on the child llm_bench's cwd. The child runs with cwd=HERE, so an
    unresolved relative --out put the step summary under HERE (a dir the runner never made there), the check found
    nothing, and a step that exited 0 passed silently. The runner resolves --out at startup, and a step that exits
    0 without its summary row is a refusal naming the path. This runs the real runner (real _locust, real
    append_summary_row) from a cwd other than HERE with a relative --out: it fails on the unfixed code."""
    import json
    import os
    import subprocess
    import sys

    from pathlib import Path

    here = Path(__file__).resolve().parent
    caller = tmp_path / "caller"
    caller.mkdir()
    plan = {
        "record": {"target_qps": 1.0},
        "traffic_generator_create_args": {
            "ramp-1": [
                "traffic-generator",
                "create",
                "--concurrency",
                "1",
                "--duration",
                "8s",
                "--qps",
                "1.0",
                "--extra-args",
                '--shape-mix \'{"seed":1,"classes":[{"name":"c","weight":1,"prompt_tokens":150,'
                '"cached_prompt_tokens":50,"generated_tokens":2}]}\' --no-chat --prefix-reuse-overhang 0 '
                "--prefix-restore-boundary 1",
            ],
        },
    }
    (caller / "plan.json").write_text(json.dumps(plan))
    # Relative --out, cwd = the caller (NOT llm_bench's dir): the unfixed runner passes the relative path to the
    # child, whose cwd is HERE, so the summary lands under HERE and the caller-side check finds nothing.
    proc = subprocess.run(
        [sys.executable, os.path.join(here, "preflight_run.py"), "plan.json", "--out", "rel-out", "--window", "0"],
        cwd=caller,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "LLM_BENCH_TEST_CHILD": "1"},
    )
    out = proc.stdout + proc.stderr
    # The run passes, its summary lands at the CALLER's resolved --out (an absolute path in the output), and
    # llm_bench's dir never gains a rel-out. On the unfixed code the summary landed under HERE, the check
    # found nothing, and the run still exited 0 (verified against 4746e4c in this test's review). The summary
    # is checked FIRST so an unfixed failure names the missing file, not a later format assert.
    summary = caller / "rel-out" / "ramp-1.summary.csv"
    assert summary.exists(), out[-400:]
    assert proc.returncode == 0, out[-400:]
    assert "(1 row(s)" in out and "no summary row to show" not in out, out[-400:]
    # The last line is the bare path (tier_runner.mock_preflight reads it and opens it); the evidence line before
    assert out.strip().splitlines()[-1] == str(summary), out[-400:]
    assert f"check input {summary} (1 row(s), exit 0)" in out, out[-400:]
    assert not (here / "rel-out").exists()
    import csv

    with open(summary, newline="") as f:
        assert sum(1 for _ in csv.reader(f)) == 2  # header + this run's row


def test_preflight_refuses_a_step_that_exits_0_without_its_summary_row(tmp_path, capsys) -> None:
    """The refusal path (D-315): a step whose llm_bench exits 0 but writes no summary row is a failed run, exit 2,
    naming the expected path and the run record (out=, summary=, rows=, locust_exit=) — never a silent pass."""
    import json
    import os

    import preflight_run
    from preflight_run import main

    plan = {
        "record": {"target_qps": 1.0},
        "traffic_generator_create_args": {
            "ramp-1": [
                "traffic-generator",
                "create",
                "--concurrency",
                "1",
                "--duration",
                "8s",
                "--qps",
                "1.0",
                "--extra-args",
                "--no-chat",
            ],
        },
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    out_dir = tmp_path / "out"

    def fake_locust(port, step, duration_cap, summary, log):
        with open(log, "w") as f:
            f.write("child ran\n")
        return 0  # exits 0 without writing the summary: the silent-pass shape

    # The mock server still has to start (the runner boots it before the step); patch only the child
    real_locust = preflight_run._locust
    preflight_run._locust = fake_locust
    try:
        rc = main([str(path), "--out", str(out_dir), "--window", "0", "--duration", "8"])
    finally:
        preflight_run._locust = real_locust
    assert rc == 2
    out = capsys.readouterr().out
    assert "refusing: step ramp-1 exited 0 without its summary row" in out
    assert f"out={out_dir}" in out and "rows=None" in out and "locust_exit=0" in out
    assert not (out_dir / "ramp-1.summary.csv").exists()
