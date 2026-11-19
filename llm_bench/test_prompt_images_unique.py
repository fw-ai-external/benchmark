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


def test_prompt_images_unique_flag_defaults_off() -> None:
    base = ["-H", "https://api.fireworks.ai/inference", "--provider", "fireworks", "--model", MODEL, "--chat"]
    assert get_parser().parse_args(base).prompt_images_unique is False
    assert get_parser().parse_args([*base, "--prompt-images-unique"]).prompt_images_unique is True
