import pytest
import torch

from scripts import bench_paper_e2e as e2e
from argus.paper_baselines.workloads import reference


def test_core_updates_fixed_export_buffers_and_rejects_broadcast(monkeypatch):
    captured = []

    def build(_, inputs):
        captured.extend(inputs)
        return lambda: inputs[0] + inputs[1]

    monkeypatch.setattr(e2e, "build_exported", build)
    monkeypatch.setattr(e2e, "reference", lambda _, inputs: inputs[0] + inputs[1])
    core = e2e.AttentionCore("decode", [torch.ones(2), torch.ones(2)], "nautilus")
    core.reference_mode = False
    first = core(torch.full((2,), 2.0), torch.full((2,), 3.0))
    second = core(torch.full((2,), 7.0), torch.full((2,), 11.0))
    assert torch.equal(first, torch.full((2,), 5.0))
    assert torch.equal(second, torch.full((2,), 18.0))
    assert captured[0].data_ptr() == core.buffers[0].data_ptr()
    with pytest.raises(ValueError, match="shape/dtype/device"):
        core(torch.ones(1), torch.ones(2))
    with pytest.raises(ValueError, match="count"):
        core(torch.ones(2))


def test_tileops_core_stages_once_into_native_bshd_storage(monkeypatch):
    from argus.paper_baselines import e2e_mtp_baselines

    seen = []

    def build(_, buffers):
        assert all(x.transpose(1, 2).is_contiguous() for x in buffers)
        views = [x.transpose(1, 2) for x in buffers]
        seen.extend(views)
        return lambda: buffers[0].clone(), {}

    monkeypatch.setattr(e2e_mtp_baselines, "build_mtp_baseline", build)
    monkeypatch.setattr(e2e, "reference", lambda _, inputs: inputs[0])
    tensors = [torch.ones(1, 4, 16, 128) for _ in range(3)]
    core = e2e.AttentionCore("mtpc", tensors, "tileops")
    core.reference_mode = False
    torch.testing.assert_close(core(*tensors), tensors[0])
    for x in tensors:
        x.fill_(3)
    torch.testing.assert_close(core(*tensors), tensors[0])
    assert all(x.data_ptr() == y.data_ptr() for x, y in zip(seen, core.buffers))


def test_mtpc_mask_is_lower_right_and_group_outputs_do_not_alias():
    q = torch.zeros(1, 4, 3, 128)
    kv = torch.zeros(1, 4, 7, 128)
    values = torch.arange(7).float()[None, None, :, None].expand_as(kv)
    actual = reference("mtpc", (q, kv, values))
    torch.testing.assert_close(actual[0, 0, :, 0], torch.tensor([2.0, 2.5, 3.0]))
    shared_output = torch.empty(1, 4, 3, 128)

    def core(query, *_):
        shared_output.copy_(query)
        return shared_output

    query = torch.arange(8).float()[None, :, None, None].expand(1, 8, 3, 128)
    output = e2e.grouped_attention(core, query, query, query, 4)
    torch.testing.assert_close(output, query)


@pytest.mark.parametrize("implementation", ["nautilus", "tileops", "flex"])
@pytest.mark.parametrize("query_length", [2, 4, 8, 16])
@torch.inference_mode()
def test_mtpc_grouped_llama_matches_hf_causal_attention(monkeypatch, implementation, query_length):
    tfs = pytest.importorskip("transformers")
    config = tfs.LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=8, num_key_value_heads=8, head_dim=128,
        attn_implementation="sdpa",
    )
    monkeypatch.setattr(tfs, "LlamaConfig", lambda **_: config)

    def factory(variant, inputs, persistent_kv=False):
        assert variant == "mtpc"
        assert persistent_kv
        assert inputs[0].shape == (1, 4, query_length, 128)
        return lambda *fresh: reference(variant, fresh).to(fresh[0].dtype)

    model, make_inputs, _ = e2e.build_decode_model(
        factory, device="cpu", variant="mtpc", kv_length=256, implementation=implementation,
        query_length=query_length,
    )
    initial = make_inputs()
    layer = initial["past_key_values"].layers[0]
    static_keys = layer.keys
    groups = layer.groups_bhsd()
    prefix = [k[:, :, :256 - query_length].clone() for k, _ in groups]
    assert initial["cache_position"].tolist() == list(range(256 - query_length, 256))
    actual = model(**make_inputs()).last_hidden_state
    assert make_inputs()["past_key_values"].layers[0].keys.data_ptr() == static_keys.data_ptr()
    for (key, _), expected_prefix in zip(groups, prefix):
        torch.testing.assert_close(key[:, :, :256 - query_length], expected_prefix)
    repeated = model(**make_inputs()).last_hidden_state
    torch.testing.assert_close(repeated, actual)
    # Rebuild standard BHSD storage only for the independent HF test oracle.
    standard = tfs.StaticCache(config=config, max_cache_len=256)
    for i, native in enumerate(initial["past_key_values"].layers):
        pairs = native.groups_bhsd()
        standard.update(torch.cat([k for k, _ in pairs], dim=1),
                        torch.cat([v for _, v in pairs], dim=1), i,
                        {"cache_position": torch.arange(256)})
    oracle = {**make_inputs(), "past_key_values": standard}
    model.config._attn_implementation = "eager"
    expected = model(**oracle).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=3e-3)


@pytest.mark.parametrize("implementation", ["nautilus", "tileops", "tileops_bound", "flex"])
@pytest.mark.parametrize("variant,q_len,heads", [("mtpc", 16, 4), ("decode", 1, 32)])
def test_persistent_core_binds_kv_without_full_cache_copy(monkeypatch, implementation,
                                                        variant, q_len, heads):
    if implementation == "tileops_bound" and variant != "decode":
        pytest.skip("Bound integration only supports decode")
    from argus.paper_baselines import e2e_mtp_baselines

    q = torch.ones(1, heads, q_len, 128)
    k = torch.ones(1, heads, 256, 128)
    v = torch.ones_like(k)
    if implementation in ("tileops", "tileops_bound"):
        k, v = (x.transpose(1, 2).contiguous().transpose(1, 2) for x in (k, v))
    copied = []
    original = torch.Tensor.copy_

    def copy(destination, source, *args, **kwargs):
        copied.append(tuple(source.shape))
        return original(destination, source, *args, **kwargs)

    def reference(_, inputs):
        return inputs[0] + inputs[1].mean() + inputs[2].mean()

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    monkeypatch.setattr(e2e, "reference", reference)
    monkeypatch.setattr(e2e, "build_exported", lambda _, inputs, **kw: lambda: reference(_, inputs))
    monkeypatch.setattr(e2e_mtp_baselines, "build_mtp_baseline",
                        lambda _, inputs: (lambda: reference(_, inputs), {}))
    core = e2e.AttentionCore(variant, (q, k, v), implementation, persistent_kv=True)
    assert core.buffers[1] is k and core.buffers[2] is v
    core.reference_mode = False
    torch.testing.assert_close(core(q, k, v), torch.full_like(q, 3))
    k.fill_(3)
    v.fill_(5)
    torch.testing.assert_close(core(q, k, v), torch.full_like(q, 9))
    assert copied == [tuple(q.shape)] * 3
    with pytest.raises(ValueError, match="binding changed"):
        core(q, k.clone(), v)


@pytest.mark.parametrize("implementation", ["nautilus", "tileops", "tileops_bound", "flex"])
@torch.inference_mode()
def test_decode_uses_model_projections_and_preserves_native_prefix(monkeypatch, implementation):
    tfs = pytest.importorskip("transformers")
    config = tfs.LlamaConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=32, num_key_value_heads=32, head_dim=128,
        attn_implementation="sdpa",
    )
    monkeypatch.setattr(tfs, "LlamaConfig", lambda **_: config)
    seen = []

    def factory(variant, inputs, persistent_kv=False):
        assert variant == "decode"
        assert persistent_kv

        def call(*fresh):
            seen.append(tuple(x.shape for x in fresh))
            return torch.nn.functional.scaled_dot_product_attention(*fresh)

        return call

    model, make_inputs, _ = e2e.build_decode_model(factory, device="cpu",
                                                  implementation=implementation)
    a, b = make_inputs(), make_inputs()
    assert a["past_key_values"] is b["past_key_values"]
    a_layer, b_layer = a["past_key_values"].layers[0], b["past_key_values"].layers[0]
    assert a_layer is b_layer
    assert a_layer.keys.data_ptr() == b_layer.keys.data_ptr()
    key, value = a_layer.groups_bhsd()[0]
    prefix = key[:, :, :511].clone()
    address = key.data_ptr()
    assert a["cache_position"].tolist() == [511]
    first = model(**a).last_hidden_state
    second = model(**b).last_hidden_state
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(key[:, :, :511], prefix)
    assert a_layer.groups_bhsd()[0][0].data_ptr() == address
    assert a["past_key_values"].get_seq_length() == 512
    assert make_inputs()["cache_position"].tolist() == [511]
    assert seen == [((1, 32, 1, 128), (1, 32, 512, 128), (1, 32, 512, 128))] * 2
