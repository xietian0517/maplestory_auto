import argparse
import json
from pathlib import Path
import time
import cv2
import mss
import numpy as np
from .core import Planner
from .vision import Vision, crop
from .windows import Desktop, KEYS


def read_config(path):
    config = json.loads(path.read_text(encoding='utf-8'))
    if not config['window_title'].strip():
        raise ValueError('window_title 不能为空')
    if not 1 <= config['fps'] <= 30 or not 0.5 <= config['threshold'] <= 1:
        raise ValueError('fps 应为 1~30，threshold 应为 0.5~1')
    skills = config['profiles'][config['profile']]['skills']
    if not skills or len({s['name'] for s in skills}) != len(skills):
        raise ValueError('至少需要一个技能，技能名称不能重复')
    for skill in skills:
        if skill['key'] not in KEYS or skill['key'] in ('f8', 'f9'):
            raise ValueError('技能按键无效，F8/F9 为保留热键')
        if not 0.01 <= skill['hold'] <= 0.5 or skill['cooldown'] < 0 or skill.get('recovery', 0) < 0:
            raise ValueError('技能持续时间应为 0.01~0.5 秒，冷却和后摇不能为负')
        if skill['range_x'] <= 0 or skill['range_y'] < 0:
            raise ValueError('技能范围无效')
    nav = config['navigation']
    if not 0.01 <= nav['move_pulse'] <= 0.5 or nav['stuck_seconds'] <= 0 or nav['waypoint_tolerance'] <= 0 or nav['same_floor_tolerance'] < 0:
        raise ValueError('导航参数无效')
    for point in nav['waypoints']:
        if not all(k in KEYS and k not in ('f8', 'f9') for k in point.get('vertical_keys', [])):
            raise ValueError('路线按键无效')
        float(point['x']), float(point['y'])
    return config


def save_image(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode('.png', frame)[1].tofile(str(path))


def annotate(scene, player, monsters, action):
    view = scene.copy()
    for entity, color in [(m, (0, 0, 255)) for m in monsters] + ([(player, (0,255,0))] if player else []):
        cv2.circle(view, (int(entity.x), int(entity.y)), 12, color, 2)
    cv2.putText(view, action.reason, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return view


def calibrate(args, path, config):
    frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('无法读取截图')
    # Templates are cropped from the full client screenshot; detector coordinates
    # are local to scene_roi, so sprite positions are never stored here.
    roi = tuple(map(int, cv2.selectROI('Select ROI - ENTER confirm / ESC cancel', frame, False)))
    cv2.destroyAllWindows()
    if roi[2] == 0 or roi[3] == 0:
        print('已取消，配置未修改')
        return
    if args.kind in ('scene', 'minimap'):
        config[args.kind+'_roi'] = roi
    else:
        folder = path.parent / 'assets'
        output = folder / f'{args.kind}_{time.time_ns()}.png'
        save_image(output, crop(frame, roi))
        config[args.kind+'_templates'].append(output.relative_to(path.parent).as_posix())
    config['client_size'] = [frame.shape[1], frame.shape[0]]
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'已保存 {args.kind}: {roi}')


def main():
    parser = argparse.ArgumentParser(description='冒险岛视觉识别与可配置技能控制')
    parser.add_argument('--config', default='config.json')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    snap = sub.add_parser('capture')
    snap.add_argument('--output', default='captures/game.png')
    snap.add_argument('--delay', type=float, default=3)
    cal = sub.add_parser('calibrate')
    cal.add_argument('kind', choices=['scene', 'minimap', 'player', 'monster'])
    cal.add_argument('--image', default='captures/game.png')
    replay = sub.add_parser('analyze')
    replay.add_argument('--image', default='captures/game.png')
    replay.add_argument('--output', default='captures/analysis.png')
    run = sub.add_parser('run')
    run.add_argument('--live', action='store_true')
    run.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    path = Path(args.config).resolve()
    if args.command == 'init':
        if path.exists():
            raise ValueError('配置已存在，不覆盖')
        path.write_text((Path(__file__).resolve().parent.parent/'config.example.json').read_text(encoding='utf-8'), encoding='utf-8')
        print(f'配置已创建: {path}')
        return
    config = read_config(path)
    if args.command == 'calibrate':
        calibrate(args, path, config)
        return
    if args.command == 'capture':
        desktop = Desktop(config['window_title'])
        print(f'{args.delay} 秒后截图，请将游戏置于前台', flush=True)
        time.sleep(max(0, args.delay))
        if not desktop.foreground():
            raise RuntimeError('请先将游戏置于前台')
        with mss.mss() as screen:
            save_image(Path(args.output), np.array(screen.grab(desktop.region()))[:, :, :3])
        print(args.output)
        return
    if not config['player_templates'] or not config['monster_templates']:
        raise ValueError('请先标定 player 和 monster 模板')
    vision, planner = Vision(config, path.parent), Planner(config)
    if args.command == 'analyze':
        frame = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('无法读取截图')
        scene, player, monsters, dot, map_view = vision.read(frame)
        action = planner.decide(player, monsters, dot, time.monotonic())
        save_image(Path(args.output), annotate(scene, player, monsters, action))
        if map_view is not None:
            save_image(Path(args.output).with_name('minimap-analysis.png'), map_view)
        print(json.dumps(dict(player=vars(player) if player else None,
                              monsters=[vars(m) for m in monsters],
                              minimap_player=vars(dot) if dot else None,
                              action=vars(action)), ensure_ascii=False, indent=2))
        return
    desktop = Desktop(config['window_title'])
    enabled, previous_f8 = not args.live, False
    print('F8 开始/暂停；F9 退出。' + (' 实际按键模式，初始暂停。' if args.live else ' 仅分析，不发送按键。'), flush=True)
    last_status = None
    try:
        with mss.mss() as screen:
            while not desktop.pressed('f9'):
                start = time.monotonic()
                f8 = desktop.pressed('f8')
                if f8 and not previous_f8:
                    enabled = not enabled
                    planner.reset()
                    desktop.release()
                previous_f8 = f8
                if not desktop.foreground():
                    if args.live:
                        enabled = False
                    desktop.release()
                    planner.reset()
                    time.sleep(0.1)
                    continue
                region = desktop.region()
                if config.get('client_size') and [region['width'], region['height']] != config['client_size']:
                    raise RuntimeError('窗口尺寸与标定不一致，请恢复尺寸或重新标定')
                frame = np.array(screen.grab(region))[:, :, :3]
                scene, player, monsters, dot, map_view = vision.read(frame)
                action = planner.decide(player, monsters, dot, time.monotonic()) if enabled else None
                if action:
                    status = (action.reason, action.skill, len(monsters))
                    if status != last_status:
                        print(status, flush=True)
                        last_status = status
                    if action.keys:
                        action_start = time.monotonic()
                        if not args.live or desktop.execute(action):
                            planner.commit(action, action_start)
                        else:
                            enabled = False
                            planner.reset()
                if args.preview:
                    from .core import Action
                    cv2.imshow('MapleBot vision', annotate(scene, player, monsters, action or Action(reason='paused')))
                    if map_view is not None:
                        cv2.imshow('Minimap edges (local coordinates)', map_view)
                    cv2.waitKey(1)
                time.sleep(max(0, 1/config['fps']-(time.monotonic()-start)))
    finally:
        desktop.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(f'错误: {error}') from error
