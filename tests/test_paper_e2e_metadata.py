import pytest

from argus.paper_baselines import workloads
from argus.paper_baselines.workloads import export_path, metadata


def test_sequence_sweep_metadata_matches_exported_abis():
    assert metadata("decode")["params"][:3] == [
        {"name": "q", "shape": [1, 32, 1, 128], "dtype": "float16"},
        {"name": "k", "shape": [1, 32, 512, 128], "dtype": "float16"},
        {"name": "v", "shape": [1, 32, 512, 128], "dtype": "float16"},
    ]
    assert metadata("mtpc")["params"][:3] == [
        {"name": "q", "shape": [1, 4, 16, 128], "dtype": "float16"},
        {"name": "k", "shape": [1, 4, 65536, 128], "dtype": "float16"},
        {"name": "v", "shape": [1, 4, 65536, 128], "dtype": "float16"},
    ]


@pytest.mark.parametrize("length", [256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
def test_decode_uses_current_length_specific_binary(length):
    path = export_path("decode", length)
    assert (path / "kernel.so").is_file()
    assert "paper_kernels.previous" not in str(path)
    if length >= 4096:
        assert "tilelang_auto" in path.parts
    params = metadata("decode", length)["params"]
    assert params[0]["shape"] == [1, 32, 1, 128]
    assert params[1]["shape"] == params[2]["shape"] == [1, 32, length, 128]


def test_decode_missing_current_binary_does_not_fall_back_to_legacy(tmp_path, monkeypatch):
    current, legacy = tmp_path / "paper_kernels", tmp_path / "paper_kernels.previous"
    old = legacy / "decode/tilelang/best/export"
    old.mkdir(parents=True)
    (old / "kernel.so").touch()
    monkeypatch.setattr(workloads, "PAPER_ROOT", current)
    monkeypatch.setattr(workloads, "LEGACY_PAPER_ROOT", legacy)
    with pytest.raises(FileNotFoundError, match="decode_L512"):
        export_path("decode")


@pytest.mark.parametrize("query_length", [2, 4, 8])
@pytest.mark.parametrize("length", [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
def test_mtpc_query_length_selects_matching_current_export(query_length, length):
    path = export_path("mtpc", length, query_length=query_length)
    assert path.name == "export" and f"mtpc_Q{query_length}_L{length}" in str(path)
    assert (path / "kernel.so").is_file()
    params = metadata("mtpc", length, query_length=query_length)["params"]
    assert params[0]["shape"] == params[3]["shape"] == [1, 4, query_length, 128]
    assert params[1]["shape"] == params[2]["shape"] == [1, 4, length, 128]
