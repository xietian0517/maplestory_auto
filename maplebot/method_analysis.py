"""Offline method mapping and bounded disassembly, no live game access."""
import argparse
from bisect import bisect_right
import json
import hashlib
from pathlib import Path
import struct
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from .pe_analysis import PE


def method_span(va, ranges, starts, pointers):
    position = bisect_right(starts,va)-1
    if position >= 0 and ranges[position][0] <= va < ranges[position][1]:
        end = ranges[position][1]
        boundary = 'PE exception function range'
    else:
        end = va+256
        boundary = '256-byte cap; may include padding'
    following = bisect_right(pointers,va)
    if following < len(pointers) and pointers[following] < end:
        end = pointers[following]
        boundary = 'next known method (exception ranges may span several methods)'
    return end,boundary


def analyze(binary_path, directory):
    binary = binary_path.read_bytes()
    header = json.loads((directory/'header.json').read_text(encoding='utf-8'))
    if header.get('native',{}).get('sha256') != hashlib.sha256(binary).hexdigest():
        raise ValueError('binary fingerprint does not match metadata analysis')
    if header['sha256'] != '91a1e8ee417864de6163517dd2b2ce001ea39c127fc840060e59d81610d1985d':
        raise ValueError('reviewed type indices apply only to the analyzed client build')
    pe = PE(binary)
    types = json.loads((directory/'types.json').read_text(encoding='utf-8'))
    game = [t for t in types if t['assembly']=='Assembly-CSharp.dll']
    methods = [(t,m) for t in game for m in t['methods']]
    module = pe.codegen_module('Assembly-CSharp.dll',max(m['token'] & 0xffffff for _,m in methods))
    mapping, names = [], {}
    for typ,method in methods:
        va = module['pointers'][(method['token'] & 0xffffff)-1]
        record = dict(type_index=typ['index'],method_index=method['index'],name=method['name'],
                      return_type=method['return_type_name'],parameter_count=method['parameter_count'],
                      rva=va-pe.base if va else None, file_offset=pe.raw(va) if va else None)
        mapping.append(record)
        if va:
            names.setdefault(va,[]).append(f"T{typ['index']}/M{method['index']} {method['name']}")
    # Resolve ordinary engine calls as additional evidence for position handling.
    engine_types = [t for t in types if t['assembly']=='UnityEngine.CoreModule.dll']
    engine_methods = [(t,m) for t in engine_types for m in t['methods']]
    engine = pe.codegen_module('UnityEngine.CoreModule.dll',max(m['token'] & 0xffffff for _,m in engine_methods))
    for typ,method in engine_methods:
        va = engine['pointers'][(method['token'] & 0xffffff)-1]
        if va:
            names.setdefault(va,[]).append(f"{typ['name']}.{method['name']}")
    pdata = next(s for s in pe.sections if s['name']=='.pdata')
    ranges = sorted((pe.base+b,pe.base+e) for b,e,_ in struct.iter_unpack('<III',pe.data[pdata['raw']:pdata['raw']+pdata['virtual_size']]))
    starts = [b for b,_ in ranges]
    pointers = sorted(names)
    cs = Cs(CS_ARCH_X86,CS_MODE_64)
    targets = {1620,1630,1540,1541,1686,1593,1610,1616,2313}
    output, getter_candidates = [], []
    for row in mapping:
        if row['type_index'] not in targets or row['rva'] is None:
            continue
        va = pe.base+row['rva']
        end,boundary = method_span(va,ranges,starts,pointers)
        length = min(end-va,4096)
        instructions = list(cs.disasm(pe.data[row['file_offset']:row['file_offset']+length],va))
        lines, accesses = [], []
        for ins in instructions:
            label = ''
            if ins.mnemonic in ('call','jmp') and ins.op_str.startswith('0x'):
                target = int(ins.op_str,16)
                label = ' ; '+ ' | '.join(names.get(target,[])[:2]) if target in names else ''
            lines.append(f'{ins.address-pe.base:08x}: {ins.mnemonic} {ins.op_str}{label}')
            if '[rcx +' in ins.op_str:
                accesses.append(ins.op_str)
            if ins.mnemonic == 'ret' and not any(i.mnemonic.startswith(('j','call','loop')) for i in instructions[:len(lines)-1]):
                break
        record = dict(**row,boundary=boundary,truncated=end-va>4096,disassembly=lines)
        output.append(record)
        if len(lines) <= 16 and accesses:
            getter_candidates.append(record)
    (directory/'method-map.json').write_text(json.dumps(mapping,ensure_ascii=False),encoding='utf-8')
    (directory/'reviewed-disassembly.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'short-accessors.json').write_text(json.dumps(getter_candidates,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(mapped_methods=len(mapping),non_null_pointers=sum(m['rva'] is not None for m in mapping),
                          reviewed_methods=len(output),short_accessors=len(getter_candidates)),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--directory',type=Path,default=Path('analysis/il2cpp'))
    args = parser.parse_args()
    analyze(args.binary,args.directory)
