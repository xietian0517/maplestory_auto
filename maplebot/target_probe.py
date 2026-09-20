"""Search exact user-provided labels in client files and test memory query access."""
import argparse
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from .memory import ProcessReader


class MemoryBasicInformation(C.Structure):
    _fields_ = [('base', C.c_void_p), ('allocation_base', C.c_void_p),
                ('allocation_protect', W.DWORD), ('partition_id', W.WORD),
                ('region_size', C.c_size_t), ('state', W.DWORD),
                ('protect', W.DWORD), ('type', W.DWORD)]


def search_private_memory(reader, handle, labels, seconds=30, max_bytes=4*1024**3):
    """Bounded exact-byte search. Hits are strings, NOT identified game objects."""
    k = reader.k
    needles = [(label, encoding, label.encode(encoding)) for label in labels for encoding in ('utf-8','utf-16-le')]
    overlap = max(len(n) for _,_,n in needles)-1
    report = dict(scope='committed readable MEM_PRIVATE regions', bytes_read=0, read_errors=0,
                  regions_visited=0, hits=[], completed=False, stop_reason=None)
    seen = set()
    end = time.monotonic()+seconds
    address = 0
    while address < 0x7fffffff0000:
        if time.monotonic() >= end or report['bytes_read'] >= max_bytes:
            report['stop_reason'] = 'time_or_byte_budget'
            return report
        info = MemoryBasicInformation()
        if not k.VirtualQueryEx(handle,address,C.byref(info),C.sizeof(info)):
            error = C.get_last_error()
            report['stop_reason'] = 'address_space_end' if error == 87 else f'VirtualQueryEx error {error}'
            report['completed'] = error == 87 and report['read_errors'] == 0
            return report
        region_base = info.base or 0
        next_address = region_base+info.region_size
        if next_address <= address:
            report['stop_reason'] = 'nonadvancing_region'
            return report
        address = next_address
        report['regions_visited'] += 1
        if info.state != 0x1000 or info.type != 0x20000 or info.protect & 0x100 or info.protect & 0xff not in (2,4,8,0x20,0x40,0x80):
            continue
        pos, tail = region_base, b''
        while pos < next_address:
            if time.monotonic() >= end or report['bytes_read'] >= max_bytes:
                report['stop_reason'] = 'time_or_byte_budget'
                return report
            size = min(1024*1024,next_address-pos,max_bytes-report['bytes_read'])
            try:
                block = reader.read(pos,size)
            except OSError as error:
                report['read_errors'] += 1
                report['last_read_error'] = error.winerror
                tail = b''
                if report['read_errors'] >= 32 and report['bytes_read'] == 0:
                    report['stop_reason'] = 'no_successful_reads_after_32_attempts'
                    return report
                pos += size
                continue
            report['bytes_read'] += len(block)
            combined = tail+block
            for label,encoding,needle in needles:
                offset = 0
                while True:
                    offset = combined.find(needle,offset)
                    if offset < 0:
                        break
                    hit_address = pos-len(tail)+offset
                    key = (label,encoding,hit_address)
                    if key not in seen:
                        seen.add(key)
                        report['hits'].append(dict(label=label,encoding=encoding,address=hex(hit_address)))
                    offset += len(needle)
                    if len(report['hits']) >= 100:
                        report['stop_reason'] = '100_hit_limit'
                        return report
            tail = combined[-overlap:]
            pos += size
    report['completed'] = report['read_errors'] == 0
    report['stop_reason'] = 'address_space_end'
    return report


def probe(pid, client):
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), pid=pid,
                  expected_player='小棒槌啊', expected_map='金银岛/射手村', files={},
                  runtime_player_verified=False, runtime_map_verified=False)
    labels = ('小棒槌啊', '射手村', '金银岛', 'Henesys')
    for filename in ('Maplestory_Classic_Data/il2cpp_data/Metadata/global-metadata.dat', 'GameAssembly.dll'):
        data = (client/filename).read_bytes()
        hits = []
        for label in labels:
            for encoding in ('utf-8','utf-16-le'):
                needle = label.encode(encoding)
                start = 0
                while True:
                    off = data.find(needle,start)
                    if off < 0:
                        break
                    hits.append(dict(label=label,encoding=encoding,file_offset=off))
                    start = off+len(needle)
        report['files'][filename] = hits
    with ProcessReader(pid) as reader:
        k = reader.k
        handle = k.OpenProcess(0x410, False, pid)
        if not handle:
            report['memory_query'] = dict(success=False, stage='OpenProcess', winerror=C.get_last_error())
            return report
        try:
            k.VirtualQueryEx.argtypes = [W.HANDLE, C.c_void_p, C.POINTER(MemoryBasicInformation), C.c_size_t]
            k.VirtualQueryEx.restype = C.c_size_t
            info = MemoryBasicInformation()
            result = k.VirtualQueryEx(handle, None, C.byref(info), C.sizeof(info))
            report['memory_query'] = dict(success=bool(result),
                                          **(dict(first_region_size=info.region_size) if result else dict(winerror=C.get_last_error())))
            if result:
                report['memory_search'] = search_private_memory(reader,handle,labels)
        finally:
            k.CloseHandle(handle)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--client', required=True, type=Path)
    args = parser.parse_args()
    result = probe(args.pid,args.client)
    output = Path('analysis/il2cpp/player-name-probe.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
