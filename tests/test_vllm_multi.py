"""Tests for VLLMServerManager GPU splitting, validation, and MultiVLLMClient."""
import threading
from unittest.mock import MagicMock, patch
import pytest

from onc_data_wrangler.llm.vllm_server import VLLMServerManager, _build_extra_args, _check_gpu_ids
from onc_data_wrangler.llm.multi_client import MultiVLLMClient
from onc_data_wrangler.llm.base import LLMResponse
from onc_data_wrangler.config import load_config, save_config, ProjectConfig


class TestGPUSplitting:
    """Test VLLMServerManager._split_gpus logic."""

    def test_even_split(self):
        """4 GPUs, 2 per server = 2 groups."""
        mgr = VLLMServerManager(model="test", gpus=[0, 1, 2, 3], gpus_per_server=2)
        groups = mgr._split_gpus()
        assert groups == [[0, 1], [2, 3]]

    def test_single_gpu_per_server(self):
        """4 GPUs, 1 per server = 4 groups."""
        mgr = VLLMServerManager(model="test", gpus=[0, 1, 2, 3], gpus_per_server=1)
        groups = mgr._split_gpus()
        assert groups == [[0], [1], [2], [3]]

    def test_all_gpus_one_server(self):
        """4 GPUs, 4 per server = 1 group."""
        mgr = VLLMServerManager(model="test", gpus=[0, 1, 2, 3], gpus_per_server=4)
        groups = mgr._split_gpus()
        assert groups == [[0, 1, 2, 3]]


class TestValidation:
    """Test VLLMServerManager validation."""

    def test_empty_gpus_raises(self):
        with pytest.raises(ValueError, match="gpus list must not be empty"):
            VLLMServerManager(model="test", gpus=[], gpus_per_server=1)

    def test_uneven_split_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            VLLMServerManager(model="test", gpus=[0, 1, 2], gpus_per_server=2)

    def test_bad_gpus_per_server(self):
        with pytest.raises(ValueError, match="gpus_per_server must be >= 1"):
            VLLMServerManager(model="test", gpus=[0], gpus_per_server=0)


class TestBuildExtraArgs:
    """Test _build_extra_args helper."""

    def test_string_value(self):
        result = _build_extra_args({"max_model_len": "4096"})
        assert result == ["--max-model-len", "4096"]

    def test_boolean_true(self):
        result = _build_extra_args({"enforce_eager": True})
        assert result == ["--enforce-eager"]

    def test_boolean_false_skipped(self):
        result = _build_extra_args({"enforce_eager": False})
        assert result == []

    def test_none_skipped(self):
        result = _build_extra_args({"some_flag": None})
        assert result == []

    def test_multiple_args(self):
        result = _build_extra_args({"max_model_len": "4096", "enforce_eager": True})
        assert "--max-model-len" in result
        assert "4096" in result
        assert "--enforce-eager" in result


class TestMultiVLLMClientRoundRobin:
    """Test MultiVLLMClient round-robin distribution."""

    @staticmethod
    def _make_mock_client(name: str):
        client = MagicMock()
        client.generate.return_value = LLMResponse(text=f"response_{name}", usage=None, raw=None)
        client.generate_structured.return_value = LLMResponse(text=f"structured_{name}", usage=None, raw=None)
        return client

    def test_round_robin_cycles(self):
        c1, c2 = self._make_mock_client("1"), self._make_mock_client("2")
        multi = MultiVLLMClient([c1, c2])

        r1 = multi.generate("test")
        r2 = multi.generate("test")
        r3 = multi.generate("test")

        assert r1.text == "response_1"
        assert r2.text == "response_2"
        assert r3.text == "response_1"

    def test_single_client(self):
        c1 = self._make_mock_client("1")
        multi = MultiVLLMClient([c1])

        r1 = multi.generate("test")
        r2 = multi.generate("test")
        assert r1.text == "response_1"
        assert r2.text == "response_1"

    def test_generate_structured_round_robin(self):
        c1, c2 = self._make_mock_client("1"), self._make_mock_client("2")
        multi = MultiVLLMClient([c1, c2])

        r1 = multi.generate_structured("test")
        r2 = multi.generate_structured("test")
        assert r1.text == "structured_1"
        assert r2.text == "structured_2"

    def test_empty_clients_raises(self):
        with pytest.raises(ValueError, match="clients list must not be empty"):
            MultiVLLMClient([])

    def test_thread_safety(self):
        c1, c2 = self._make_mock_client("1"), self._make_mock_client("2")
        multi = MultiVLLMClient([c1, c2])

        results = []
        def worker():
            for _ in range(100):
                r = multi.generate("test")
                results.append(r.text)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 400
        # Both clients should have been used
        assert "response_1" in results
        assert "response_2" in results


class TestConfigParsing:
    """Test config parsing related to vLLM servers."""

    def test_vllm_servers_parsed(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
project:
  name: test
  input_paths: []
  output_dir: /tmp/test

extraction:
  llm:
    provider: openai
    model: test-model
  vllm_servers:
    gpus: [0, 1, 2, 3]
    gpus_per_server: 2
    base_port: 29500
    extra_args:
      max_model_len: "4096"
      enforce_eager: true
""")
        config = load_config(str(config_file))
        assert config.extraction.vllm_servers.gpus == [0, 1, 2, 3]
        assert config.extraction.vllm_servers.gpus_per_server == 2
        assert config.extraction.vllm_servers.base_port == 29500
        assert config.extraction.vllm_servers.extra_args == {"max_model_len": "4096", "enforce_eager": True}

    def test_missing_vllm_servers_uses_defaults(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text("""
project:
  name: test
  input_paths: []
  output_dir: /tmp/test
""")
        config = load_config(str(config_file))
        assert config.extraction.vllm_servers.gpus == []
        assert config.extraction.vllm_servers.gpus_per_server == 1
        assert config.extraction.vllm_servers.base_port == 29500

    def test_save_config_roundtrip(self, tmp_path):
        config = ProjectConfig(
            name="roundtrip_test",
            input_paths=["/data/input"],
            output_dir="/data/output",
        )
        save_path = tmp_path / "saved.yaml"
        save_config(config, str(save_path))

        loaded = load_config(str(save_path))
        assert loaded.name == "roundtrip_test"
        assert loaded.input_paths == ["/data/input"]
        assert loaded.output_dir == "/data/output"
