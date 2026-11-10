"""Regression coverage for model exports without custom configuration modules."""

import json
from pathlib import Path
import tempfile
import unittest

from llm_bench.load_test import resolve_model_type


class ResolveModelTypeTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.config = self.directory / "config.json"

    def test_missing_custom_module_reads_text_config(self) -> None:
        self.config.write_text(
            json.dumps(
                {
                    "model_type": "kimi_k31",
                    "auto_map": {"AutoConfig": "configuration_kimi_k3.KimiK3Config"},
                    "text_config": {"model_type": "kimi_linear"},
                }
            )
        )
        self.assertEqual(resolve_model_type(str(self.directory)), "kimi_linear")

    def test_native_config_still_loads(self) -> None:
        self.config.write_text(json.dumps({"model_type": "llama"}))
        self.assertEqual(resolve_model_type(str(self.directory)), "llama")

    def test_malformed_config_still_fails(self) -> None:
        self.config.write_text("{invalid")
        with self.assertRaises(json.JSONDecodeError):
            resolve_model_type(str(self.directory))


if __name__ == "__main__":
    unittest.main()
