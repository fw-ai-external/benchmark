from __future__ import annotations

import json
from pathlib import Path

import pytest
from locust.argument_parser import get_parser
from locust.env import Environment

import load_test
from load_test import PROMPT_CHAT_IMAGE_PLACEHOLDER, DatasetHolder, InitTracker, JsonlDataset, LLMUser

HOST = "https://api.fireworks.ai/inference"
MODEL = "accounts/pyroworks/deployments/test"


def _chat_line(i: int) -> dict:
    return {"messages": [{"role": "user", "content": f"turn {i}"}], "max_tokens": 16}


@pytest.fixture
def jsonl(tmp_path: Path) -> Path:
    path = tmp_path / "chat.jsonl"
    path.write_text("".join(json.dumps(_chat_line(i)) + "\n" for i in range(3)))
    return path


def test_users_share_one_iterator_and_take_consecutive_lines(jsonl: Path) -> None:
    dataset = JsonlDataset(str(jsonl))
    first_user, second_user = iter(dataset), iter(dataset)
    drawn = [next(first_user)[0], next(second_user)[0], next(first_user)[0], next(second_user)[0]]
    assert drawn == [_chat_line(0), _chat_line(1), _chat_line(2), _chat_line(0)]


def test_first_prompt_does_not_take_a_line(jsonl: Path) -> None:
    dataset = JsonlDataset(str(jsonl))
    assert dataset.first_prompt() == _chat_line(0)
    assert next(iter(dataset))[0] == _chat_line(0)


class _User(LLMUser):
    host = HOST  # locust's runner sets this from --host before spawning


def _user(jsonl: Path, *flags: str) -> LLMUser:
    options = get_parser().parse_args(
        ["-H", HOST, "--provider", "fireworks", "--model", MODEL, "--chat", "--dataset", f"@{jsonl}", *flags]
    )
    return _User(Environment(user_classes=[_User], host=HOST, parsed_options=options))


def test_on_start_with_chat_jsonl(jsonl: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DatasetHolder, "_instance", None)
    monkeypatch.setattr(load_test.time, "sleep", lambda _: None)

    def no_tokenizer(_path):
        raise AssertionError("a JSONL request body must not load a tokenizer")

    monkeypatch.setattr(InitTracker, "load_tokenizer", no_tokenizer)

    first_user, second_user = _user(jsonl), _user(jsonl)
    first_user._on_start()
    second_user._on_start()

    assert first_user.prompt_tokenizer_tokens == 0
    # Starting users took no lines, and the two users draw consecutive ones
    prompt, _, images = first_user._get_input()
    assert prompt == _chat_line(0)
    assert images is None
    assert second_user._get_input()[0] == _chat_line(1)
    payload = first_user.provider_formatter.format_payload(prompt, 16, images)
    assert payload["messages"] == _chat_line(0)["messages"]


def _start(monkeypatch: pytest.MonkeyPatch, jsonl: Path, *flags: str) -> LLMUser:
    monkeypatch.setattr(DatasetHolder, "_instance", None)
    monkeypatch.setattr(load_test.time, "sleep", lambda _: None)
    user = _user(jsonl, *flags)
    user._on_start()
    return user


def test_clear_assistant_leaves_the_shared_row_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
        {"role": "assistant", "content": "B"},
    ]
    path = tmp_path / "chat.jsonl"
    path.write_text(json.dumps({"messages": messages}) + "\n")
    monkeypatch.setattr(InitTracker, "load_tokenizer", lambda _path: pytest.fail("no tokenizer for a request body"))
    user = _start(monkeypatch, path, "--clear-assistant")

    # Two passes through the one-line cycle send the same request, and the dataset's row keeps all three turns
    first, second = (user.provider_formatter.format_payload(user._get_input()[0], 16, None) for _ in range(2))
    assert first["messages"] == second["messages"] == messages[:2]
    assert DatasetHolder._instance.first_prompt()["messages"] == messages


def test_each_test_run_rereads_the_file_from_line_0(jsonl: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(InitTracker, "load_tokenizer", lambda _path: pytest.fail("no tokenizer for a request body"))
    user = _start(monkeypatch, jsonl)
    assert user._get_input()[0] == _chat_line(0)
    assert user._get_input()[0] == _chat_line(1)

    # The Locust UI stops the run, the file changes, and a new run starts in the same process
    jsonl.write_text(json.dumps(_chat_line(7)) + "\n")
    load_test._reset_jsonl_dataset(environment=user.environment)
    assert _start(monkeypatch, jsonl)._get_input()[0] == _chat_line(7)


def test_string_line_count_includes_image_placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps("hello") + "\n")
    tokenizer = type("CharTokenizer", (), {"encode": staticmethod(list)})()
    monkeypatch.setattr(InitTracker, "load_tokenizer", lambda _path: tokenizer)
    user = _start(monkeypatch, path, "--prompt-images-with-resolutions", "8x8", "--prompt-images-positioning", "end")
    assert user.prompt_tokenizer_tokens == len("hello" + PROMPT_CHAT_IMAGE_PLACEHOLDER)
    assert user._get_input()[0] == "hello" + PROMPT_CHAT_IMAGE_PLACEHOLDER  # the peek took no line
