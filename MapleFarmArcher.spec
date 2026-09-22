# 独立射手程序，内置本次参考截图裁出的识别模板。
a = Analysis(['rope_archer_gui.py'], pathex=[], binaries=[],
             datas=[('templates/rope_archer', 'templates/rope_archer')],
             hiddenimports=[], hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=[], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='MapleFarmArcher',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          upx_exclude=[], runtime_tmpdir=None, console=False, uac_admin=True,
          disable_windowed_traceback=False, argv_emulation=False,
          target_arch=None, codesign_identity=None, entitlements_file=None)
