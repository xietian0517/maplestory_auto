"""Read-only Windows process access; no injection or memory writes."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import hashlib
import json
from pathlib import Path
import struct


class ModuleEntry(C.Structure):
    _fields_ = [('size', W.DWORD), ('module_id', W.DWORD), ('pid', W.DWORD),
                ('global_usage', W.DWORD), ('process_usage', W.DWORD),
                ('base', C.c_void_p), ('bytes', W.DWORD), ('module', W.HMODULE),
                ('name', W.WCHAR * 256), ('path', W.WCHAR * 260)]


class ProcessReader:
    def __init__(self, pid):
        self.pid = pid
        self.k = C.WinDLL('kernel32', use_last_error=True)
        self.k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.k.OpenProcess.restype = W.HANDLE
        self.k.CloseHandle.argtypes = [W.HANDLE]
        self.k.ReadProcessMemory.argtypes = [W.HANDLE, C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_size_t)]
        self.k.ReadProcessMemory.restype = W.BOOL
        self.k.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
        self.k.CreateToolhelp32Snapshot.restype = W.HANDLE
        for name in ('Module32FirstW', 'Module32NextW'):
            fn = getattr(self.k, name)
            fn.argtypes = [W.HANDLE, C.POINTER(ModuleEntry)]
            fn.restype = W.BOOL
        self.handle = self.k.OpenProcess(0x0010, False, pid)
        if not self.handle:
            raise C.WinError(C.get_last_error())

    def close(self):
        if self.handle:
            self.k.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def read(self, address, size):
        if not self.handle:
            raise RuntimeError('process reader is closed')
        if address <= 0 or not 0 < size <= 1024*1024:
            raise ValueError('invalid address or read size (max 1 MiB)')
        data, count = C.create_string_buffer(size), C.c_size_t()
        if not self.k.ReadProcessMemory(self.handle, address, data, size, C.byref(count)):
            raise C.WinError(C.get_last_error())
        if count.value != size:
            raise RuntimeError(f'partial read: {count.value}/{size}')
        return data.raw

    def scalar(self, address, kind='f32'):
        fmt = {'f32':'<f', 'i32':'<i', 'u32':'<I', 'u64':'<Q'}[kind]
        return struct.unpack(fmt, self.read(address, struct.calcsize(fmt)))[0]

    def resolve(self, address, offsets, pointer_size=8):
        """For each offset: address = read_pointer(address) + offset."""
        if pointer_size not in (4, 8):
            raise ValueError('pointer_size must be 4 or 8')
        for offset in offsets:
            pointer = self.scalar(address, 'u64' if pointer_size == 8 else 'u32')
            if pointer == 0:
                raise ValueError('null pointer in chain')
            address = pointer + offset
        return address

    def modules(self):
        snapshot = self.k.CreateToolhelp32Snapshot(0x08 | 0x10, self.pid)
        if snapshot == C.c_void_p(-1).value:
            raise C.WinError(C.get_last_error())
        try:
            entry = ModuleEntry(size=C.sizeof(ModuleEntry))
            if not self.k.Module32FirstW(snapshot, C.byref(entry)):
                raise C.WinError(C.get_last_error())
            result = []
            while True:
                result.append(dict(name=entry.name, base=entry.base, size=entry.bytes))
                if not self.k.Module32NextW(snapshot, C.byref(entry)):
                    error = C.get_last_error()
                    if error != 18:
                        raise C.WinError(error)
                    break
            return result
        finally:
            self.k.CloseHandle(snapshot)


def probe(pid, client):
    report = dict(pid=pid, client=str(client), files={}, read_access=None,
                  game_state_decoded=False, missing=['player layout', 'monster collection layout', 'map geometry layout'])
    for name in ['Maplestory_Classic.exe', 'GameAssembly.dll', 'Maplestory_Classic_Data/il2cpp_data/Metadata/global-metadata.dat']:
        path = client/name
        if path.exists():
            with path.open('rb') as f:
                prefix = f.read(8)
                f.seek(0)
                sha = hashlib.file_digest(f, 'sha256').hexdigest()
            report['files'][name] = dict(size=path.stat().st_size, sha256=sha, header=prefix.hex())
    try:
        with ProcessReader(pid) as reader:
            report['read_handle_opened'] = True
            report['stage'] = 'enumerate_modules'
            modules = reader.modules()
            report['stage'] = 'read_module_headers'
            report['modules'] = [m for m in modules if m['name'].lower() in ('gameassembly.dll','unityplayer.dll','maplestory_classic.exe')]
            for module in report['modules']:
                try:
                    module['header_read'] = reader.read(module['base'], 2).hex()
                    report['read_access'] = bool(report['read_access']) or module['header_read'] == '4d5a'
                except OSError as error:
                    module['error'] = str(error)
            report['stage'] = 'probe_complete'
    except OSError as error:
        report['error'] = str(error)
        report['winerror'] = error.winerror
    return report


def main():
    parser = argparse.ArgumentParser(description='怀旧版只读内存兼容性诊断（尚未实现游戏对象解码）')
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('captures/memory-probe.json'))
    args = parser.parse_args()
    report = probe(args.pid, args.client)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
