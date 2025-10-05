import numpy as np
from tvm import dlight, ir, relax, te, tir, transform
from tvm.relax.frontend import nn
from tvm.relax.pipeline import register_pipeline
from tvm.script import relax as R


def get_attention_module(
    q_shape,
    k_shape,
    v_shape,
    is_causal: bool,
    window_size: int | None = None,
    input_dt="float16",
    compute_dt="float32",
):
    sqrt_d = np.sqrt(q_shape[-1])
    neg_inf_val = tir.min_value(compute_dt)
    q_heads, k_heads = q_shape[1], k_shape[1]
    groups = q_heads // k_heads

    def causal_mask_te(score: te.Tensor):
        return te.compute(
            score.shape, lambda b, h, i, j: tir.Select(i >= j, score[b, h, i, j], neg_inf_val)
        )

    def window_mask_te(score: te.Tensor):
        return te.compute(
            score.shape,
            lambda b, h, i, j: tir.Select(
                tir.all(i >= j, i - j < window_size), score[b, h, i, j], neg_inf_val
            ),
        )

    bb = relax.BlockBuilder()
    q = relax.Var("q", R.Tensor(q_shape, input_dt))
    k = relax.Var("k", R.Tensor(k_shape, input_dt))
    v = relax.Var("v", R.Tensor(v_shape, input_dt))
    with bb.function("main", [q, k, v]):
        with bb.dataflow():
            if groups > 1:
                k = R.repeat(k, groups, 1)
                v = R.repeat(v, groups, 1)
            qkt = bb.emit(
                R.matmul(q, R.permute_dims(k, [0, 1, 3, 2]), compute_dt)
                / R.const(sqrt_d, compute_dt)
            )
            if window_size is not None:
                masked = bb.emit_te(window_mask_te, qkt)
            elif is_causal:
                masked = bb.emit_te(causal_mask_te, qkt)
            else:
                masked = qkt
            score = bb.emit(R.astype(R.nn.softmax(masked, axis=-1), input_dt))
            output = bb.emit(R.astype(R.matmul(score, v, compute_dt), input_dt))
            output = bb.emit_output(output)
        bb.emit_func_output(output)
    return bb.get()


@register_pipeline("opt_llm")
def _pipeline(ext_mods: list[nn.ExternModule] | None = None):
    ext_mods = ext_mods or []

    @transform.module_pass(opt_level=3)
    def _pipeline(mod: ir.IRModule, _ctx: transform.PassContext) -> ir.IRModule:
        seq = transform.Sequential(
            [
                # Phase 1. Passes on high-level operator graph
                # We can enable cublas for further optimization
                relax.transform.FuseTransposeMatmul(),
                # Phase 2. Lowering to TIR, inherited TVM Relax's official "zero" pipeline
                relax.transform.LegalizeOps(),
                relax.transform.AnnotateTIROpPattern(),
                relax.transform.FoldConstant(),
                relax.transform.FuseOps(),
                relax.transform.FuseTIR(),
                # Phase 3. Passes on TIR
                relax.transform.DeadCodeElimination(),
                # Phase 4. Low-level Optimizations
                dlight.ApplyDefaultSchedule(
                    dlight.gpu.Matmul(),
                    dlight.gpu.GEMV(),
                    dlight.gpu.Reduction(),
                    dlight.gpu.GeneralReduction(),
                    dlight.gpu.Fallback(),
                ),
                # Phase 5. Lowering to VM bytecode
                relax.transform.RewriteDataflowReshape(),
                relax.transform.ToNonDataflow(),
                relax.transform.RemovePurityChecking(),
                relax.transform.CallTIRRewrite(),
                relax.transform.StaticPlanBlockMemory(),
                relax.transform.RewriteCUDAGraph(),
                relax.transform.LowerAllocTensor(),
                relax.transform.KillAfterLastUse(),
                relax.transform.LowerRuntimeBuiltin(),
                relax.transform.VMShapeLower(),
                relax.transform.AttachGlobalSymbol(),
                relax.transform.AttachExternModules(ext_mods),
            ]
        )
        mod = seq(mod)
        return mod

    return _pipeline
