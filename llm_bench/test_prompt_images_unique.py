from __future__ import annotations

import base64
import io

from locust.argument_parser import get_parser
from PIL import Image

from load_test import LLMUser

MODEL = "accounts/pyroworks/deployments/test"


def _decode(data_uri: str) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    assert data_uri.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(data_uri[len(prefix) :])))


def test_default_images_are_identical() -> None:
    first = LLMUser._create_base64_image(None, 64, 48)
    assert first == LLMUser._create_base64_image(None, 64, 48)
    assert _decode(first).size == (64, 48)


def test_unique_images_differ_in_bytes_and_pixels() -> None:
    images = [LLMUser._create_base64_image(None, 64, 48, unique=True) for _ in range(20)]
    assert len(set(images)) == len(images)
    pixels = {_decode(image).convert("RGB").tobytes() for image in images}
    assert len(pixels) == len(images)
    assert all(_decode(image).size == (64, 48) for image in images)


def test_prompt_images_unique_defaults_on() -> None:
    base = ["-H", "https://api.fireworks.ai/inference", "--provider", "fireworks", "--model", MODEL, "--chat"]
    assert get_parser().parse_args(base).prompt_images_unique is True
    assert get_parser().parse_args([*base, "--prompt-images-unique"]).prompt_images_unique is True
    assert get_parser().parse_args([*base, "--no-prompt-images-unique"]).prompt_images_unique is False


def test_unique_images_do_not_block_the_event_loop() -> None:
    import gevent

    class _User:
        prompt_image_resolutions = [(756, 756)] * 40
        _create_base64_image = LLMUser._create_base64_image
        _create_unique_images = LLMUser._create_unique_images

    ticks = []

    def ticker() -> None:
        while True:
            gevent.sleep(0.001)
            ticks.append(1)

    tk = gevent.spawn(ticker)
    gevent.sleep(0.005)
    ticks.clear()
    images = _User()._create_unique_images()
    tk.kill()
    assert len(images) == 40 and len(set(images)) == 40
    # Built inline, the batch holds the loop for its whole duration and the ticker fires at most once.
    assert len(ticks) >= 5
