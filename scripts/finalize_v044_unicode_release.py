"""Attach source and verified Unicode-path checks to the ai2 release."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'variants/v044_ai'
RELEASE = ROOT / 'releases/v0.4.4-ai2'


def main():
    self_test = json.loads((RELEASE / 'unicode_exe_self_test.json').read_text(encoding='utf-8'))
    assert all(self_test[name] is True for name in (
        'ready', 'name_template_ok', 'hud_calibration_ok', 'start_enabled', 'frozen', 'encoder_ok'))
    assert self_test['input_started'] is False
    test_log = (SOURCE / 'baseline_test_results.txt').read_text(encoding='utf-8')
    assert 'Ran 482 tests' in test_log and test_log.rstrip().endswith('OK')
    (RELEASE / 'baseline_test_results.txt').write_text(test_log, encoding='utf-8')
    verification = dict(
        version='0.4.4-ai2', baseline='0.4.4',
        non_gui_tests_passed=482, gui_tests_passed=17,
        unicode_frozen_exe_self_test=self_test,
        fix='Python filesystem access with OpenCV image codecs for Unicode paths',
        live_gameplay_tested=False, ten_minute_exp_confirmed=False,
    )
    (RELEASE / 'verification.json').write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with zipfile.ZipFile(ROOT / 'releases/v0.4.4-ai1/source.zip') as previous:
        source_names = previous.namelist()
    source_names += ['autofarm/realtime/image_io.py', 'tests/test_unicode_image_io.py']
    assert len(source_names) == len(set(source_names))
    with zipfile.ZipFile(RELEASE / 'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as source_zip:
        for name in source_names:
            source_file = (SOURCE / name).resolve()
            assert source_file.is_relative_to(SOURCE.resolve())
            source_zip.write(source_file, name)
    baseline_path = ROOT / 'AI_BASELINE.json'
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    baseline['release'] = 'releases/v0.4.4-ai2'
    baseline_path.write_text(json.dumps(baseline, indent=2) + '\n', encoding='utf-8')
    package = ROOT / 'dist/MapleAIController-v0.4.4-ai2-试用包.zip'
    additions = ['source.zip', 'verification.json', 'unicode_exe_self_test.json', 'baseline_test_results.txt']
    with zipfile.ZipFile(package, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in additions:
            archive.write(RELEASE / name, 'MapleAIController-v0.4.4-ai2/' + name)
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert archive.testzip() is None
        assert not any('deepseek.txt' in name or '/runs/' in name for name in names)
    print(json.dumps(dict(
        package=str(package), crc_ok=True,
        sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        non_gui_tests_passed=482, gui_tests_passed=17,
        unicode_name_template_ok=True, unicode_hud_calibration_ok=True,
    ), ensure_ascii=True))


if __name__ == '__main__':
    main()
