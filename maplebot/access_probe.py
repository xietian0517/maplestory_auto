"""Compare standard Windows query paths without changing process permissions."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json


def probe(pid):
    k = C.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.CloseHandle.argtypes = [W.HANDLE]
    k.QueryFullProcessImageNameW.argtypes = [W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)]
    k.K32EnumProcessModulesEx.argtypes = [W.HANDLE, C.POINTER(W.HMODULE), W.DWORD, C.POINTER(W.DWORD), W.DWORD]
    a = C.WinDLL('advapi32', use_last_error=True)
    a.OpenProcessToken.argtypes = [W.HANDLE, W.DWORD, C.POINTER(W.HANDLE)]
    a.GetTokenInformation.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)]
    report = dict(pid=pid, current_process_elevated=bool(C.windll.shell32.IsUserAnAdmin()), attempts=[])
    for rights in (0x1000, 0x1010, 0x410):
        h = k.OpenProcess(rights, False, pid)
        result = dict(rights=hex(rights), opened=bool(h))
        report['attempts'].append(result)
        if not h:
            result['winerror'] = C.get_last_error()
            continue
        try:
            name, length = C.create_unicode_buffer(32768), W.DWORD(32768)
            ok = k.QueryFullProcessImageNameW(h, 0, name, C.byref(length))
            result['image_query'] = name.value if ok else dict(winerror=C.get_last_error())
            token = W.HANDLE()
            if a.OpenProcessToken(h, 8, C.byref(token)):
                try:
                    elevated, needed = W.DWORD(), W.DWORD()
                    if a.GetTokenInformation(token, 20, C.byref(elevated), C.sizeof(elevated), C.byref(needed)):
                        result['target_elevated'] = bool(elevated.value)
                    else:
                        result['token_error'] = C.get_last_error()
                finally:
                    k.CloseHandle(token)
            else:
                result['token_error'] = C.get_last_error()
            if rights == 0x410:
                modules, needed = (W.HMODULE * 2048)(), W.DWORD()
                ok = k.K32EnumProcessModulesEx(h, modules, C.sizeof(modules), C.byref(needed), 3)
                result['psapi'] = dict(success=bool(ok))
                if ok:
                    result['psapi']['module_count'] = needed.value // C.sizeof(W.HMODULE)
                else:
                    result['psapi']['winerror'] = C.get_last_error()
        finally:
            k.CloseHandle(h)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(probe(args.pid), ensure_ascii=False, indent=2))
