# .venv/Scripts/python.exe -m PyInstaller --noconfirm MapleDemoRecorder.spec
from PyInstaller.utils.hooks import collect_data_files
a = Analysis(['game_demo_gui.py'], pathex=[], binaries=[], datas=collect_data_files('imageio_ffmpeg'), hiddenimports=['mss.windows'],
             hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=['torch','onnxruntime','rapidocr'], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,a.binaries,a.datas,[],name='MapleDemoRecorder-v0.1.1',
          debug=False,bootloader_ignore_signals=False,strip=False,upx=False,
          console=False,disable_windowed_traceback=False,uac_admin=True)
