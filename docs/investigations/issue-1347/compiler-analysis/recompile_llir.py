"""Isolate codegen and AGPR allocation on already-optimized cached LLVM IR."""
import argparse
import hashlib
import json
import pathlib
import re

import triton
from triton.backends.amd import compiler

p = argparse.ArgumentParser()
p.add_argument("--source", type=pathlib.Path, required=True)
p.add_argument("--output-prefix", type=pathlib.Path, required=True)
p.add_argument("--agpr", choices=["unchanged", "remove", "0", "1"], default="unchanged")
p.add_argument("--flags", default="")
args = p.parse_args()
src = args.source.read_text()
# LLVM 23 added a disjoint-scopes flag to alias-scope domain metadata. False
# preserves the older domain semantics; this is an IR-format adaptation only.
src = src.replace('!{!"amdg.AsyncOps", !"Domain', '!{!"amdg.AsyncOps", i1 false, !"Domain')
if args.agpr != "unchanged":
    src = re.sub(r' "amdgpu-agpr-alloc"="[^"]*"', '', src)
    if args.agpr != "remove":
        src = src.replace('attributes #0 = {', 'attributes #0 = { "amdgpu-agpr-alloc"="' + args.agpr + '"')
args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
args.output_prefix.with_suffix('.llir').write_text(src)
assembly = compiler.compile_amdgpu(src, compiler.amd.get_target_triple('gfx950'), 'gfx950', '',
                                  flags=args.flags.split(',') if args.flags else [], enable_fp_fusion=False,
                                  disable_optimization=False, canonicalize_gep=False,
                                  disabled_passes='', dump_ir=False, enable_timing=False)
args.output_prefix.with_suffix('.amdgcn').write_text(assembly)
code = [line.strip() for line in assembly.splitlines() if re.match(r'\s+[sv]|\s+(buffer|global|scratch|ds)_', line)]
result = dict(source=str(args.source), agpr=args.agpr, flags=args.flags,
              codegen_revision=compiler.get_amd_codegen_revision(),
              inst_sha256=hashlib.sha256('\n'.join(code).encode()).hexdigest(),
              inst_count=len(code))
for key in ['vgpr_count', 'vgpr_spill_count', 'sgpr_count', 'sgpr_spill_count', 'private_segment_fixed_size']:
    result[key] = int(re.search(r'\.'+key+r':\s*(\d+)', assembly).group(1))
result['scratch_instructions'] = sum(x.startswith('scratch_') for x in code)
result['accvgpr_instructions'] = sum(x.startswith('v_accvgpr') for x in code)
args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
