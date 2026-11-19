from __future__ import annotations

import base64
import io

import gevent
from PIL import Image

from load_test import LLMUser


def _decode(data_uri: str) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    assert data_uri.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(data_uri[len(prefix) :])))


class _User:
    _create_base64_image = LLMUser._create_base64_image
    _create_images = LLMUser._create_images

    def __init__(self, resolutions: list[tuple[int, int]]) -> None:
        self.prompt_image_resolutions = resolutions


def test_images_differ_in_bytes_and_pixels() -> None:
    images = [LLMUser._create_base64_image(None, 64, 48) for _ in range(20)]
    assert len(set(images)) == len(images)
    pixels = {_decode(image).convert("RGB").tobytes() for image in images}
    assert len(pixels) == len(images)
    assert all(_decode(image).size == (64, 48) for image in images)


def test_each_request_gets_fresh_images_in_the_requested_sizes() -> None:
    user = _User([(64, 48), (32, 32)])
    first, second = user._create_images(), user._create_images()
    assert [_decode(image).size for image in first] == [(64, 48), (32, 32)]
    assert not set(first) & set(second)


def test_building_images_does_not_block_the_event_loop() -> None:
    ticks = []

    def ticker() -> None:
        while True:
            gevent.sleep(0.001)
            ticks.append(1)

    tk = gevent.spawn(ticker)
    gevent.sleep(0.005)
    ticks.clear()
    images = _User([(756, 756)] * 40)._create_images()
    tk.kill()
    assert len(images) == 40 and len(set(images)) == 40
    # Built inline, the batch holds the loop for its whole duration and the ticker fires at most once.
    assert len(ticks) >= 5
