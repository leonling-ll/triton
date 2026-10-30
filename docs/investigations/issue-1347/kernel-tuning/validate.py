"""Full-output checks of the issue 1347 MFMA32 config override."""
import copy
import json
from pathlib import Path
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch

root = Path(__file__).resolve().parent
config = json.loads((root/'unshuffled-m51200-n106496-k16384.json').read_text())
results = []
for shape in [(1024,1024,1024),(768,1536,16384),(1056,1088,1024)]:
    x,w,_,xs,ws,_,_,_,out = generate_gemm_afp4wfp4_inputs(*shape,torch.bfloat16)
    ref = run_torch(x,w,xs,ws,torch.bfloat16)
    gemm_afp4wfp4(x,w,xs,ws,y=out,config=copy.deepcopy(config))
    torch.cuda.synchronize()
    torch.testing.assert_close(out,ref,rtol=0.02,atol=1.0)
    row=dict(shape=shape,full_output_pass=True,entries=out.numel(),max_abs_error=(out-ref).abs().max().item(),exact_match=bool(torch.equal(out,ref)))
    print(json.dumps(row),flush=True)
    results.append(row)
(root/'full-correctness.json').write_text(json.dumps(results,indent=2))
