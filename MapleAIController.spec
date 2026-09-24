# Build with .venv/Scripts/python.exe -m PyInstaller --noconfirm MapleAIController.spec
a = Analysis(['game_ai_gui.py'], pathex=[], binaries=[], datas=[], hiddenimports=['mss.windows'],
             hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['torch','onnxruntime','rapidocr'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='MapleAIController-v0.2.0',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, disable_windowed_traceback=False, uac_admin=True)
