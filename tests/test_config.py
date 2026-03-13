import pytest
from src.config import PAECConfig


def test_load_default_config():
    cfg = PAECConfig.from_yaml("configs/default.yaml")
    assert cfg.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.max_seq_length == 512
    assert cfg.num_virtual_tokens == 20
    assert cfg.conflict_epsilon == 1e-8
    assert cfg.seeds == [42, 123, 456]


def test_config_override():
    cfg = PAECConfig.from_yaml("configs/default.yaml", overrides={"batch_size": 8})
    assert cfg.batch_size == 8


def test_config_invalid_evidence_fn():
    with pytest.raises(ValueError):
        PAECConfig.from_yaml("configs/default.yaml", overrides={"evidence_fn": "invalid"})
