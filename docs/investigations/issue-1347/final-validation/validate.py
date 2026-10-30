"""Paired exact-shape MXFP4 measurements with event and rotating-buffer graphs."""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch
from op_tests.op_benchmarks.triton.utils.benchmark_utils import do_bench_aiter_triton

ROOT = Path(__file__).parent
M, N, K = 51200, 106496, 16384
BASE = dict(BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256,
            GROUP_SIZE_M=4, num_warps=8, num_stages=2, waves_per_eu=1,
            matrix_instr_nonkdim=16, cache_modifier=None, NUM_KSPLIT=1)
CONFIGS = {'default': BASE, 'mfma32_group8': BASE | dict(matrix_instr_nonkdim=32, GROUP_SIZE_M=8)}
phase = 'setup'
stop = threading.Event()


def monitor():
    with (ROOT / 'telemetry.jsonl').open('w') as f:
        while not stop.is_set():
            try:
                started = time.time()
                tag = phase
                d = json.loads(subprocess.check_output([
                    'amd-smi', 'metric', '-g', '0000:78:00.0', '-p', '-c', '-t', '-u', '-v', '--json'
                ]))['gpu_data'][0]
                entry = dict(time=started, phase=tag,
                    power_w=d['power']['socket_power']['value'],
                    clocks_mhz=[d['clock'][f'gfx_{i}']['clk']['value'] for i in range(8)],
                    gfx_activity=d['usage']['gfx_activity']['value'],
                    hotspot_c=d['temperature']['hotspot']['value'],
                    throttle={k:d['throttle'][k] for k in [
                        'ppt_violation_status', 'socket_thermal_violation_status',
                        'vr_thermal_violation_status', 'hbm_thermal_violation_status']})
                f.write(json.dumps(entry)+'\n')
                f.flush()
            except Exception as exc:
                f.write(json.dumps({'error': repr(exc)})+'\n')
            stop.wait(.25)


def call(config, **kw):
    return gemm_afp4wfp4(**kw, config=copy.deepcopy(config))


torch.manual_seed(1347)
result = dict(shape=[M,N,K], device=str(torch.cuda.get_device_properties(0)),
    hip_visible_devices=os.getenv('HIP_VISIBLE_DEVICES'), configs=CONFIGS,
    triton_commit=subprocess.check_output(['git','-C','/home/leling/triton','rev-parse','HEAD'],text=True).strip(),
    aiter_commit=subprocess.check_output(['git','-C','/home/leling/aiter','rev-parse','HEAD'],text=True).strip(),
    measurements=[], correctness=[], graph_settings=dict(pool_len=4,kernels_per_graph=10,num_replays=10,warmup_replays=3))
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
pool=[]
try:
    for i in range(4):
        x,w,_,xs,ws,_,_,_,out=generate_gemm_afp4wfp4_inputs(M,N,K,torch.bfloat16)
        pool.append(dict(x=x,w=w,x_scales=xs,w_scales=ws,y=out))
        # Distinct allocations rotate memory addresses. The AITER generator resets
        # its seed, so values and sampled positions repeat across these buffers.
        rows=torch.cat([torch.tensor([0,M-1],device=x.device),torch.randperm(M,device=x.device)[:62]])
        cols=torch.cat([torch.tensor([0,N-1],device=x.device),torch.randperm(N,device=x.device)[:62]])
        ref=run_torch(x[rows],w[cols],xs[rows],ws[cols],torch.bfloat16)
        for name,c in CONFIGS.items():
            call(c,**pool[-1])
            torch.cuda.synchronize()
            torch.testing.assert_close(out[rows[:,None],cols[None,:]],ref,rtol=.02,atol=1.)
            result['correctness'].append(dict(buffer=i,variant=name,check='4096 samples vs decoded FP32 matmul',passed=True))
        print(f'buffer {i} ready and checked',flush=True)
    for mode,rounds in [('events',7),('graph',3)]:
        for rnd in range(rounds):
            names=list(CONFIGS)
            if rnd%2:
                names.reverse()
            for name in names:
                config=CONFIGS[name]
                fn=lambda config=config,**kw:call(config,**kw)
                phase=f'{mode}/{name}'
                start=time.time()
                if mode=='events':
                    ms=triton.testing.do_bench(lambda:fn(**pool[0]),warmup=100,rep=500,return_mode='median')
                else:
                    ms=do_bench_aiter_triton(fn,pool,warmup_iters=3,kernels_per_graph=10,num_replays=10,warmup_replays=3)
                row=dict(mode=mode,round=rnd,variant=name,start=start,end=time.time(),ms=ms,tflops=2*M*N*K/ms*1e-9)
                result['measurements'].append(row)
                (ROOT/'results.json').write_text(json.dumps(result,indent=2))
                print(json.dumps(row),flush=True)
    result['summary']={}
    for mode in ['events','graph']:
        result['summary'][mode]={}
        for name in CONFIGS:
            vals=[v['ms'] for v in result['measurements'] if v['mode']==mode and v['variant']==name]
            med=statistics.median(vals)
            result['summary'][mode][name]=dict(median_ms=med,median_tflops=2*M*N*K/med*1e-9,best_tflops=2*M*N*K/min(vals)*1e-9)
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary'],indent=2),flush=True)
finally:
    stop.set()
    thread.join()
