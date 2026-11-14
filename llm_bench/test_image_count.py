import argparse
import random

import pytest

import load_test
from load_test import ImageCountSampler


def test_sampler_follows_weights() -> None:
    random.seed(0)
    sampler = ImageCountSampler("0:1,6:2,19:1")
    draws = [sampler.sample() for _ in range(20000)]
    assert set(draws) == {0, 6, 19}
    assert abs(draws.count(6) / len(draws) - 0.5) < 0.02


@pytest.mark.parametrize("spec", ["6", "a:1", "6:-1", "-1:1", "0:0"])
def test_sampler_rejects_bad_spec(spec: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        ImageCountSampler(spec)


def test_unique_images_differ_and_are_stable() -> None:
    create = load_test.LLMUser._create_base64_image
    a, b = create(None, 64, 64, unique_id=1), create(None, 64, 64, unique_id=2)
    assert a != b
    assert a == create(None, 64, 64, unique_id=1)
    assert create(None, 64, 64) == create(None, 64, 64)
