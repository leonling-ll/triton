"""Issue 1347 full-shape public-wrapper tuning experiment."""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time
import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffled_scales
from aiter.ops.shuffle import shuffle_scale
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--configs', type=Path)
p.add_argument('--rep', type=int, default=400)
p.add_argument('--rounds', type=int, default=1)
p.add_argument('--shape', nargs=3, type=int, default=[51200, 106496, 16384])
args = p.parse_args()
M,N,K = args.shape
base = dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=4, num_warps=8, num_stages=2, waves_per_eu=1,
            matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1)
if args.configs:
    configs = json.loads(args.configs.read_text())
else:
    changes = [dict(),
        dict(matrix_instr_nonkdim=32),
        dict(num_stages=1), dict(num_stages=3),
        dict(BLOCK_SIZE_K=128), dict(BLOCK_SIZE_K=512),
        dict(BLOCK_SIZE_K=512, num_stages=1),
        dict(BLOCK_SIZE_K=512, matrix_instr_nonkdim=32),
        dict(BLOCK_SIZE_M=128), dict(BLOCK_SIZE_N=128),
        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=4),
        dict(BLOCK_SIZE_M=128, BLOCK_SIZE_K=512, num_warps=4),
        dict(BLOCK_SIZE_N=128, BLOCK_SIZE_K=512, num_warps=4),
        dict(BLOCK_SIZE_M=128, num_warps=4),
        dict(BLOCK_SIZE_N=128, num_warps=4),
        dict(BLOCK_SIZE_M=128, matrix_instr_nonkdim=32),
        dict(BLOCK_SIZE_N=128, matrix_instr_nonkdim=32),
        dict(num_warps=16), dict(waves_per_eu=2),
        dict(GROUP_SIZE_M=1),dict(GROUP_SIZE_M=8),dict(GROUP_SIZE_M=16),
        dict(cache_modifier='.cg'),dict(cache_modifier='.ca')]
    configs = [base | c for c in changes]
result = dict(shape=args.shape, device=str(torch.cuda.get_device_properties(0)),
              hip_visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'), variants=[])
print(json.dumps(result), flush=True)
x,w,_,xs,ws,_,_,_,out = generate_gemm_afp4wfp4_inputs(M,N,K,torch.bfloat16)
xsh = shuffle_scale(xs).view(M // 32,K)
wsh = shuffle_scale(ws).view(N // 32,K)
rows = torch.linspace(0,M-1,min(M,32),device=x.device).long()
cols = torch.linspace(0,N-1,min(N,32),device=x.device).long()
ref = run_torch(x[rows],w[cols],xs[rows],ws[cols],torch.bfloat16)
def call(c):
    return gemm_afp4wfp4_preshuffled_scales(x,w,xsh,wsh,y=out,config=copy.deepcopy(c))
def measure(c):
    return triton.testing.do_bench(lambda:call(c), warmup=100, rep=args.rep, return_mode='median')
for rnd in range(args.rounds):
    order = list(enumerate(configs))
    if rnd % 2:
        order.reverse()
    for index,c in order:
        entry = dict(index=index, round=rnd, config=c, start=time.time())
        try:
            call(c)
            torch.cuda.synchronize()
            torch.testing.assert_close(out[rows[:,None],cols[None,:]],ref,rtol=0.02,atol=1.0)
            entry['correctness'] = '32x32 sampled FP32 reference pass'
            entry['ms'] = measure(c)
            entry['tflops'] = 2*M*N*K/entry['ms']*1e-9
            if index % 5 == 0:
                entry['paired_baseline_ms'] = measure(base)
                entry['speedup'] = entry['paired_baseline_ms']/entry['ms']
        except Exception as e:
            entry['error'] = repr(e)
        result['variants'].append(entry)
        args.output.write_text(json.dumps(result,indent=2))
        print(json.dumps(entry),flush=True)
