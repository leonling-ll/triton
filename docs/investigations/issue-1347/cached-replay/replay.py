"""Replay saved gfx950 HSACO without rebuilding it to isolate codegen differences."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import triton
from triton.compiler.compiler import CompiledKernel
from aiter.ops.shuffle import shuffle_scale
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs, run_torch

ROOT = Path('/home/leling/issue-investigations/1347')


def load_cached(cache, kernel):
    paths = [p for p in (ROOT / cache).glob(f'*/{kernel}.json')
             if p.with_suffix('.ttir').exists()]
    assert len(paths) == 1, paths
    metadata = paths[0]
    group = {p.name: str(p) for p in metadata.parent.iterdir()
             if p.name.startswith(kernel + '.')}
    # These saved IR modules have 5 pointers followed by 9 i32 arguments.
    # Unit strides and all constexpr arguments were removed by specialization.
    src = SimpleNamespace(signature={i: t for i, t in enumerate(
        ['*i8', '*i8', '*bf16', '*i8', '*i8'] + ['i32'] * 9)})
    return CompiledKernel(src, group, json.loads(metadata.read_text())['hash']), metadata


def load_generated(prefix, reference_cache):
    from triton.backends.amd.compiler import HIPBackend
    reference, ref_path = load_cached(reference_cache, '_gemm_afp4wfp4_kernel')
    prefix = ROOT / 'compiler-analysis' / prefix
    assembly = prefix.with_suffix('.amdgcn').read_text()
    hsaco = HIPBackend.make_hsaco(assembly, {}, SimpleNamespace(arch='gfx950'))
    out = ROOT / 'cached-replay' / 'generated' / prefix.name
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(ref_path.read_text())
    metadata['hash'] = hashlib.sha256(hsaco).hexdigest()
    metadata_path = out / '_gemm_afp4wfp4_kernel.json'
    metadata_path.write_text(json.dumps(metadata, indent=2))
    binary_path = out / '_gemm_afp4wfp4_kernel.hsaco'
    binary_path.write_bytes(hsaco)
    return CompiledKernel(reference.src, {metadata_path.name: str(metadata_path),
        binary_path.name: str(binary_path)}, metadata['hash']), metadata_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=7)
    p.add_argument('--output', type=Path, default=Path(__file__).with_name('results.json'))
    p.add_argument('--scales', action='store_true')
    p.add_argument('--agpr', action='store_true', help='Replay controlled AGPR attribute pairs')
    a = p.parse_args()
    if a.scales and a.agpr:
        p.error('--scales and --agpr are separate experiments')
    M, N, K = 51200, 106496, 16384
    torch.manual_seed(1347)
    result = {'shape': [M, N, K], 'device': str(torch.cuda.get_device_properties(0)),
              'started': time.time(), 'scales_shuffled': a.scales,
              'controlled_agpr_experiment': a.agpr,
              'timing': {'rounds': a.rounds, 'warmup_ms': 100, 'rep_ms': 500,
                         'round_statistic': 'median', 'variant_order': 'rotating'},
              'variants': {}}
    print(json.dumps(result), flush=True)
    x, w, _, xs, ws, _, _, _, out = generate_gemm_afp4wfp4_inputs(M, N, K, torch.bfloat16)
    rows = torch.linspace(0, M - 1, 32, device=x.device).long()
    cols = torch.linspace(0, N - 1, 32, device=x.device).long()
    ref = run_torch(x[rows], w[cols], xs[rows], ws[cols], torch.bfloat16)
    if a.scales:
        xsh = shuffle_scale(xs).view(M // 32, K)
        wsh = shuffle_scale(ws).view(N // 32, K)
    cases = [('oct27', 'cache-explicit'), ('oct29', 'cache-20261029'),
             ('current', 'cache-gap-baseline')]
    if a.agpr:
        cases = [('current', 'cache-gap-baseline')]
        cases += [(name, ref) for name, ref in (
            ('oldir-newcodegen-agpr-unchanged', 'cache-20261029'),
            ('oldir-newcodegen-agpr-0', 'cache-20261029'),
            ('currentir-agpr-unchanged', 'cache-gap-baseline'),
            ('currentir-agpr-remove', 'cache-gap-baseline'))]
    calls = {}
    kernels = []
    for label, cache in cases:
        suffix = '_preshuffle_scales' if a.scales else ''
        if 'agpr-' in label:
            kernel, metadata = load_generated(label, cache)
        else:
            kernel, metadata = load_cached(cache, '_gemm_afp4wfp4_kernel' + suffix)
        kernels.append(kernel)
        if a.scales:
            args = (x, w, out, xsh, wsh, M, N, K // 2, K // 2, K // 2, 0, N, K, K)
        else:
            args = (x, w, out, xs, ws, M, N, K // 2, K // 2, K // 2, 0, N, xs.stride(1), ws.stride(1))
        runner = kernel[(triton.cdiv(M, 256) * triton.cdiv(N, 256), 1, 1)]
        calls[label] = lambda runner=runner, args=args: runner(*args)
        calls[label]()
        torch.cuda.synchronize()
        torch.testing.assert_close(out[rows[:, None], cols[None, :]], ref, rtol=0.02, atol=1.0)
        result['variants'][label] = {'metadata': str(metadata), 'hsaco_sha256': hashlib.sha256(kernel.kernel).hexdigest(),
            'n_regs': kernel.n_regs, 'n_spills': kernel.n_spills, 'shared': kernel.metadata.shared,
            'correctness': 'passed 32x32 samples vs decoded FP32 matmul', 'ms': []}
        print(f'{label}: loaded and checked {kernel.n_regs=} {kernel.n_spills=}', flush=True)
    names = list(calls)
    for r in range(a.rounds):
        for name in names[r % len(names):] + names[:r % len(names)]:
            ms = triton.testing.do_bench(calls[name], warmup=100, rep=500, return_mode='median')
            result['variants'][name]['ms'].append(ms)
            print(f'round={r} {name}: {ms:.5f} ms {2*M*N*K/ms*1e-9:.1f} TFLOP/s', flush=True)
        a.output.write_text(json.dumps(result, indent=2))
    for v in result['variants'].values():
        v['median_ms'] = statistics.median(v['ms'])
        v['median_tflops'] = 2 * M * N * K / v['median_ms'] * 1e-9
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
