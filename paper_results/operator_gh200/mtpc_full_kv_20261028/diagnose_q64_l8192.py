import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right
from scripts.bench_paper_sweep import load_inputs, reference, export_dir

case='mtpc_Q64_L8192'; root=Path(__file__).parent
q,k,v=load_inputs(case, root/'nautilus'/f'{case}_inputs.pt')
ref32=reference(case,[q,k,v]).float()
module=torch.load if False else None
import tvm
out=torch.empty_like(q)
entry=tvm.runtime.load_module(str(export_dir(case)/'kernel.so'))['main']
entry(*[tvm.nd.from_dlpack(x) for x in (q,k,v,out)])
actual=out.float()
sdpa=F.scaled_dot_product_attention(q,k,v,attn_mask=causal_lower_right(q.shape[-2], k.shape[-2])).float()
for name, x in [('sdpa',sdpa),('kernel',actual)]:
 d=x-ref32
 print(name,'max_abs=',d.abs().max().item(),'allclose=',torch.allclose(x,ref32,rtol=1e-2,atol=3e-3),'rel_l2=',(d.norm()/ref32.norm()).item())
d=actual-sdpa
print('kernel_vs_sdpa','max_abs=',d.abs().max().item(),'allclose=',torch.allclose(actual,sdpa,rtol=1e-2,atol=3e-3),'rel_l2=',(d.norm()/sdpa.norm()).item())
