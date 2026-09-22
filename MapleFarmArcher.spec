# 独立射手程序，内置本次参考截图裁出的识别模板。
from PyInstaller.utils.hooks import collect_data_files

# 大模型放在 exe 旁 models/name_ocr/，源码与发行包使用相同路径。
ocr_data = collect_data_files('rapidocr', includes=['config.yaml', 'default_models.yaml'])
a = Analysis(['rope_archer_gui.py'], pathex=[], binaries=[],
             datas=[('templates/rope_archer', 'templates/rope_archer')] + ocr_data,
             hiddenimports=['rapidocr.main', 'onnxruntime'], hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['torch', 'paddle', 'openvino', 'tensorrt', 'MNN'], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='MapleFarmArcher',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          upx_exclude=[], runtime_tmpdir=None, console=False, uac_admin=True,
          disable_windowed_traceback=False, argv_emulation=False,
          target_arch=None, codesign_identity=None, entitlements_file=None)
