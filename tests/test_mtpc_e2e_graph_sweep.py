import sys
from types import SimpleNamespace

import pytest

from scripts import bench_mtpc_e2e_graph_sweep as sweep


@pytest.mark.parametrize("query_length", [2, 4, 8])
@pytest.mark.parametrize("implementation", ["tileops", "flex", "nautilus"])
def test_query_sweep_uses_all_lengths_and_whole_step_graph(
    tmp_path, monkeypatch, query_length, implementation
):
    class Registry(dict):
        def register(self, name, function):
            self[name] = function

    registry = Registry(sdpa=object())
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils",
                        SimpleNamespace(ALL_ATTENTION_FUNCTIONS=registry))
    measured, cleaned = [], []
    monkeypatch.setattr(sweep, "measure", lambda args: measured.append(args))
    monkeypatch.setattr(sweep.torch._dynamo, "reset", lambda: None)
    monkeypatch.setattr(sweep.torch.cuda, "empty_cache", lambda: cleaned.append(True))
    monkeypatch.setattr(sweep.gc, "collect", lambda: None)
    monkeypatch.setattr(sys, "argv", ["sweep", implementation, str(tmp_path),
                                      "--query-length", str(query_length)])

    sweep.main()

    assert [args.kv_length for args in measured] == list(sweep.LENGTHS)
    assert len(measured) == 9
    assert all(args.variant == "mtpc" and args.implementation == implementation
               and args.query_length == query_length and args.cudagraph is True
               and args.repeat == 30 and args.warmup == 5 and not args.profile_child
               for args in measured)
    assert [args.output.name for args in measured] == [
        f"mtpc_L{length}_{implementation}" for length in sweep.LENGTHS
    ]
    assert len(cleaned) == 9
    assert registry["paper_e2e"] is registry["sdpa"]


def test_query_sweep_refuses_existing_measurement_before_any_work(tmp_path, monkeypatch):
    existing = tmp_path / "mtpc_L65536_flex.json"
    existing.write_text("preserved result\n")
    measured = []
    monkeypatch.setattr(sweep, "measure", lambda args: measured.append(args))
    monkeypatch.setattr(sys, "argv", ["sweep", "flex", str(tmp_path), "--query-length", "4"])

    with pytest.raises(SystemExit, match="2"):
        sweep.main()

    assert not measured
    assert existing.read_text() == "preserved result\n"
