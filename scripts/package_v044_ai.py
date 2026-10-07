"""Package the v0.4.4 fork without credentials or development-run artifacts."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'variants/v044_ai'
sys.path.insert(0,str(SOURCE))
from game_ai_gui import prepare_session,VERSION
from autofarm.realtime.semantic import atomic_json


def main():
    release=ROOT/'releases'/('v'+VERSION);release.mkdir(parents=True,exist_ok=False)
    exe=release/f'MapleAIController-v{VERSION}.exe'
    shutil.copy2(SOURCE/'dist'/exe.name,exe)
    preset=prepare_session(ROOT/'dist/captures/monkey_forest_v044',release/'captures/monkey_forest_v044')
    atomic_json(preset/'ai_strategy.json',dict(version=1,enabled=True,credential_file=str(ROOT/'dist/deepseek.txt')))
    calibrated=ROOT/'releases/v0.5.6/captures/monkey_forest_v046'
    shutil.copy2(calibrated/'feedback_config.json',preset/'feedback_config.json')
    shutil.copytree(calibrated/'feedback_calibration',preset/'feedback_calibration')
    instructions='''0.4.4 基础版 + AI 策略 2 · 中文路径修复

修复名字模板和经验校准图片在中文文件夹路径下无法读取的问题；截图保存也支持中文路径。

沿用0.4.4原版控制、攻击、爬绳、感知、恢复和返程。新增DeepSeek后台选择练级锚点。
这是“AI策略 + 0.4.4本地战斗”混合模式，本地战斗目标选择仍由原版逻辑负责。

解压完整目录，双击EXE，载入“当前角色·猴子森林预设”。
勾选“DeepSeek后台调整练级位置”开启AI；取消勾选可运行原版决策作为对照。
Key优先读取本次程序内存；本机配置使用用户指定的E:\\work\\mxd\\dist\\deepseek.txt。
也可点击“填写DeepSeek Key”，Key仅存在本次程序内存，不保存。
API调用包含当前游戏图像，会产生API费用。没有有效回复时原版本地控制照常运行。

保持游戏前台，关闭其他挂机程序。点击开始后切回游戏；F11或停止按钮立即停止。
每30秒最多发起一次策略请求；候选锚点沿用原版距离、占用和最多两段路线条件。
跨层动作进行中不更换锚点。回复跨场景、过期或目标条件变化会被丢弃。
strategy_requests保存模型输入、图片和回复，status.json中ai_strategy显示连接和采纳情况。

本版本未声明已经提升经验效率；真实成绩以独立计分报告为准。
'''
    (release/'使用说明.txt').write_text(instructions,encoding='utf-8')
    shutil.copy2(SOURCE/'baseline_provenance.json',release/'baseline_provenance.json')
    provenance=json.loads((SOURCE/'baseline_provenance.json').read_text())
    critical=['control','climbing','combat','calibration','perception','recovery','parking']
    hashes={name:hashlib.sha256((SOURCE/f'autofarm/realtime/{name}.py').read_bytes()).hexdigest() for name in critical}
    assert all(hashes[name]==provenance['files'][f'autofarm/realtime/{name}.py'] for name in critical)
    report=dict(baseline='0.4.4',critical_sources_byte_identical=True,core_sha256=hashes,
        exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),live_performance_accepted=False)
    atomic_json(release/'build_verification.json',report)
    package=ROOT/'dist'/f'MapleAIController-v{VERSION}-试用包.zip'
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(release.rglob('*')):
            if p.is_file():z.write(p,Path(f'MapleAIController-v{VERSION}')/p.relative_to(release))
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert not any('deepseek.txt' in n or '/runs/' in n for n in z.namelist())
    print(json.dumps(dict(release=str(release),exe=str(exe),package=str(package),**report)))


if __name__=='__main__':main()
