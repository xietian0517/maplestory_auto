"""Create a separate EXE trial folder with only calibrated scene assets and docs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from game_ai_gui import prepare_session,VERSION
from autofarm.realtime.policy import PolicyConfig
from autofarm.realtime.semantic import atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene',default='captures/policy_stage3_trial_20261006')
    parser.add_argument('--allow-transfers',action='store_true')
    parser.add_argument('--thinking',action='store_true')
    parser.add_argument('--map-transit',action='store_true')
    parser.add_argument('--hierarchical',action='store_true')
    args=parser.parse_args()
    source=ROOT/'dist'/f'MapleAIController-v{VERSION}.exe'
    if not source.is_file():raise FileNotFoundError(source)
    release=ROOT/'releases'/f'v{VERSION}'
    release.mkdir(parents=True,exist_ok=False)
    exe=release/source.name;shutil.copy2(source,exe)
    preset=prepare_session(ROOT/args.scene,release/'captures'/'monkey_forest_v046')
    config=PolicyConfig.parse(dict(version=1,provider='deepseek',mode='active',
        interval=1.,timeout=30.,response_ttl=30.,fallback='wait',allow_transfers=args.allow_transfers,
        deepseek_thinking=args.thinking,allow_map_transit=args.map_transit,
        hierarchical=args.hierarchical,
        max_output_tokens=8192 if args.thinking else 4096))
    atomic_json(preset/'ai_policy.json',config.data())
    shutil.copy2(ROOT/'AI_POLICY_USAGE.md',release/'AI_POLICY_USAGE.md')
    instructions=f'''冒险岛 AI 行动决策试用版 {VERSION} · DeepSeek

先解压整个试用包，保持 EXE 与 captures 文件夹放在一起。
停止并关闭旧版程序，双击 {exe.name}，允许管理员运行。

1. 点击“载入当前角色·猴子森林预设”。仅适用于已有地图与校准角色。
2. 点击“AI 配置”，服务商选 DeepSeek，模型保留 deepseek-flash。
3. 粘贴 DeepSeek API Key，点击“应用并检测”。
   检测包含一张合成图片和最小等待动作请求，会产生少量 API 费用，不发送游戏按键。
   Key 仅本次程序使用，不写入文件。关闭程序后需要重新填写。
4. 检测通过后，“行动决策”选“AI 主动”，首轮时长 60 秒。
   已有预设可保持“交给当前对话识图”；无需重新识图。
   新地图选择“程序在线识图”也使用 DeepSeek，但仍需相应地图和角色校准。
5. 点击“开始挂机”，3 秒内切回游戏，保持前台；F11 或“停止”停止。

检测失败时，窗口显示错误原因、HTTP 状态及处理建议。
EXE 旁 action_api_diagnostic.json 保存检测结果，不含 Key。
每次运行新建 captures/runs/run_...；policy_requests 中的 error.json 保存安全错误分类。
Key 无效、余额不足、无模型权限等错误会暂停重试，修正后重新开始运行。

模型可选择等待、同平台站位、朝向和攻击目标。
AI 跨平台转移配置：{'已开启，仅限校准任务平台及可达路线' if args.allow_transfers else '关闭'}；主动模式默认不自动抢占 Buff。
未得到有效模型动作时等待；AI 影子只记录建议，规则基线不属于 AI 接管。
分层计划：{'开启：AI指定目的平台、站位、方向和攻击预算；到达后直接攻击，累计25秒实际攻击后请求复盘，等回复时继续执行' if args.hierarchical else '关闭：使用逐动作决策'}。
连续攻击由 AI 指定；逐帧检查游戏前台、站稳条件及血量。计划攻击预算最长120秒。
DeepSeek 思考模式：{'开启' if args.thinking else '关闭'}。
经过已校准地图平台：{'允许；可能经过有怪物的平台，生命检查仍启用' if args.map_transit else '仅限原安全平台'}。
真实运行是否达标请查看附带实测报告；接口连通和短时出手不等于十分钟经验达标。
技术说明见 AI_POLICY_USAGE.md。
'''
    (release/'使用说明.txt').write_text(instructions,encoding='utf-8')
    package=ROOT/'dist'/f'MapleAIController-v{VERSION}-试用包.zip'
    files=[exe,release/'使用说明.txt',release/'AI_POLICY_USAGE.md',*sorted(preset.rglob('*'))]
    with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as output:
        for path in files:
            if path.is_file():output.write(path,Path(f'MapleAIController-v{VERSION}')/path.relative_to(release))
    with zipfile.ZipFile(package) as check:
        assert check.testzip() is None
        assert not any('preferences' in name or '/runs/' in name for name in check.namelist())
        count=len(check.namelist())
    report=dict(version=VERSION,provider=config.provider,model=config.model,
        exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),zip_sha256=hashlib.sha256(package.read_bytes()).hexdigest(),
        zip_file_count=count,zip_bytes=package.stat().st_size,zip_crc_ok=True,
        authenticated_api_test=False,game_inputs_sent=False)
    atomic_json(release/'build_verification.json',report)
    print(json.dumps(dict(release=str(release),exe=str(exe),package=str(package),**report)))


if __name__=='__main__':main()
