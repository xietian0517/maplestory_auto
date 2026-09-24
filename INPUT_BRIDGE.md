# 冒险岛输入桥接 v0.1.0

让 Codex 使用 Computer Use 看游戏画面，再由独立的管理员输入程序执行一次短按键。
程序复用本项目的 `WinApi.send_key`，不启动原挂机助手的路线、识怪或自动攻击逻辑。

## 中午验证

1. 打开冒险岛、进入角色，退出原挂机助手。
2. 双击 `F:\mxd\dist\MapleInputBridge-v0.1.0.exe`，在 Windows 管理员提示中选择允许。
   只需给这个输入桥接授权，无需反复重启 Codex。
3. 窗口应显示“桥接运行中 · 管理员输入已就绪”。
4. 点击“右移 0.2 秒”，在 3 秒倒计时内切回游戏，观察角色是否移动。
5. 分别验证“跳跃 Alt”和“攻击 Shift”。每次点击只执行一次；每次都要切回游戏。
6. 三项验证完成后，回到当前 Codex 任务回复结果。后续由 Codex 观察画面、逐次发送操作。

如果你一直停留在桥接窗口，测试会被拒绝，这是前台保护生效。
“已松键”表示输入接口已执行完毕；角色是否移动、跳跃或命中仍须看实际画面。

**F11 随时停止。** 也可以点击“立即停止”或关闭窗口。
停止后需要再次点击“启动桥接”才能接受新的操作。

## 运行边界

- 只允许窗口标题“冒险岛怀旧服”，且进程路径必须是：
  `F:\mxd\冒险岛online\mxdclassic\Maplestory_Classic.exe`。
- 每次动作最多 2 秒、最多 3 个键；仅移动、跳跃和攻击键可组合。
- 游戏失去焦点、按 F11、接口异常或持键到期时，尝试立即松键。
- 5 分钟没有成功动作、或本次会话达到 15 分钟时自动停止服务。
  窗口保留，方便重新启动会话。
- 本机回环地址、随机端口和每次启动生成的令牌；拒绝未授权请求、浏览器跨站请求、
  过期请求和重复动作编号。
- 没有远程监听、任意文本输入、命令执行、客户端修改或登录功能。
- 窗口必须在当前交互桌面上；不要同时使用另一套挂机程序。

如果游戏路径改变，需要更新 `game_input_bridge.py` 中的 `GAME` 并重新打包。
如果窗口无法启动，保留弹出的错误内容，回到本任务排查。

## Codex 调用

GUI 启动后，普通权限的 Python 客户端即可向已授权的桥接发送请求：

```powershell
python game_input_bridge.py health
python game_input_bridge.py action --hwnd <最新游戏窗口ID> --keys shift --ms 100
python game_input_bridge.py action --hwnd <最新游戏窗口ID> --keys right --ms 200
python game_input_bridge.py stop
```

窗口 ID 必须来自 Computer Use 当前列出的窗口；不要长期保存或猜测。
每次操作后重新截图确认结果。超时表示结果未知，先观察画面，不要直接重发。
组合示例为 `--keys right alt --ms 100`，仍受同一目标、前台和时长限制。

会话连接信息保存在 `%LOCALAPPDATA%\MapleInputBridge\session.json`，包含临时令牌，
不要贴进聊天或上传。服务正常停止时会清理自己的会话文件。

## 源码与测试

- `autofarm/input_bridge.py`：限时输入、松键监护、前台检查、动作校验。
- `autofarm/bridge_service.py`：本地认证接口、会话到期、客户端。
- `game_input_bridge.py`：管理员状态检查、固定游戏进程验证、单实例和验证窗口。
- `MapleInputBridge.spec`：独立 EXE 打包，不改原助手发行包。

```powershell
python -m unittest discover -s tests -p 'test_*bridge*.py' -v
python -m PyInstaller --noconfirm MapleInputBridge.spec
```

自动测试使用假的输入设备，不向实际游戏发键。
当前 23 项自动测试通过，覆盖输入逻辑、本地通信和验证窗口的倒计时/取消行为；
真实管理员启动和游戏响应留待用户手动验证。
