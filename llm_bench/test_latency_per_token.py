from __future__ import annotations

import pytest

import load_test
from load_test import generation_window, track_performance_metrics


def test_window_counts_intervals_between_tokens_not_tokens() -> None:
    # 10 tokens 20 ms apart: first at t=0.5 s, last at 0.68 s -> 9 intervals of 20 ms
    seconds, intervals = generation_window(0.5, 0.68, 10)
    assert intervals == 9
    assert seconds / intervals * 1000 == pytest.approx(20.0)


@pytest.mark.parametrize("num_tokens", [None, 0, 1])
def test_window_is_none_without_an_interval(num_tokens) -> None:
    assert generation_window(0.5, 0.7, num_tokens) is None


def test_window_is_none_when_last_token_is_not_after_first() -> None:
    assert generation_window(0.5, 0.5, 10) is None
    assert generation_window(0.5, None, 10) is None


def test_generation_rate_excludes_first_token_and_stream_tail(monkeypatch) -> None:
    recorded = {}
    monkeypatch.setattr(load_test, "add_custom_metric", lambda name, value, length=0: recorded.__setitem__(name, value))
    # 200 tokens: first at 0.5 s, last at 1.495 s (199 x 5 ms), stream closes 2 ms later
    track_performance_metrics(response_time=1.497, token_count=200, char_count=800, ttft=0.5, last_token_time=1.495)
    assert recorded["generation_time"] == pytest.approx(0.995)
    assert recorded["generation_tokens_per_second"] == pytest.approx(200.0)


def test_generation_rate_falls_back_to_stream_end(monkeypatch) -> None:
    recorded = {}
    monkeypatch.setattr(load_test, "add_custom_metric", lambda name, value, length=0: recorded.__setitem__(name, value))
    track_performance_metrics(response_time=1.5, token_count=11, char_count=40, ttft=0.5)
    assert recorded["generation_tokens_per_second"] == pytest.approx(10.0)
