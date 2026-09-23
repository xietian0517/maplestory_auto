from PyInstaller.utils.hooks import collect_data_files

ocr_data = collect_data_files('rapidocr', includes=['config.yaml', 'default_models.yaml'])
a = Analysis(['unified_gui.py'], pathex=[], binaries=[],
             datas=[('templates/rope_archer', 'templates/rope_archer'),
                    ('templates/platform_guard_example', 'templates/platform_guard_example')] + ocr_data,
             hiddenimports=['rapidocr.main', 'onnxruntime'], hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['torch', 'paddle', 'openvino', 'tensorrt', 'MNN'], noarchive=False, optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='MapleFarmUnified',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          upx_exclude=[], runtime_tmpdir=None, console=False, uac_admin=True,
          disable_windowed_traceback=False, argv_emulation=False,
          target_arch=None, codesign_identity=None, entitlements_file=None)
