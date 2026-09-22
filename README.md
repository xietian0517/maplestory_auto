# 冒险岛怀旧服挂机脚本

新增独立方案 **`rope_archer`（绳边射手 v1.4.0）**：右侧有猴子时持续按住 Shift，无怪松开；支持光圈辅助识别，击退后持续按住方向键回到平台内侧，移动按剩余距离提前松键。
直接运行 `dist\MapleFarmArcher.exe` 或 `rope_archer_gui.py`，使用独立配置，不覆盖旧方案。
模板、边界、使用步骤与限制见 [射手模板说明](templates/rope_archer/README.md)。截图只能降低走落风险，无法保证被怪物击退时绝不掉落。

仓库包含 v1.4.0 的 Windows 发行包。克隆仓库或下载并解压整个仓库后，直接打开
[`dist/MapleFarmArcher.exe`](dist/MapleFarmArcher.exe)，无需安装 Python。
`dist/rope_archer_config.json` 是射手界面配置，`dist/templates/rope_archer/` 包含平台、人物、猴子和光圈图片及识别参数；移动发行包时请保留这些相对路径。
源码运行的射手配置是根目录 `rope_archer_config.json`，原有名字牌图片在 `assets/`。
修改配置后，运行中的程序会保存对应目录的配置文件。

从源码构建：安装 `requirements.txt` 中的依赖及 PyInstaller，然后执行
`python -m PyInstaller --noconfirm MapleFarmArcher.spec`。打包前可执行
`python -m unittest discover -s tests -q` 进行离线测试，测试不会控制游戏。

模拟玩家的键盘输入，按设定节奏跳起来攻击；同时每隔几秒截一张游戏画面，用**名字牌**认出主角
在平台的左半边还是右半边，决定接下来往哪边打，避免越打越靠边掉下去。

只用两种手段：**SendInput 发按键** + **截图做模板匹配**。不读内存、不注入、不改客户端。

## 目前实现了什么

| 能力 | 说明 |
| --- | --- |
| 模拟键盘输入 | 方向键/跳跃键/攻击键的按下与抬起全走 Windows SendInput，只在游戏窗口处于前台时发送 |
| 节奏化攻击 | 移动时长、出手间隔、按住时长、轮间间隔、喝药间隔都是可调的随机区间 |
| 位置判断 | 每 4 秒（可调）截一张客户区画面，用名字牌模板匹配主角的 x。实测真名牌得分 1.00、同图聊天栏文字只有 0.61，阈值很好分 |
| 攻击方向跟随位置 | 在右半边就一直往左打、在左半边就一直往右打；跨过中线才换方向，不做「左一下右一下」 |
| 不走出平台 | 离平台两端不足设定像素时算贴边，按倍数多挪一段回中间，挪完立刻重新确认 |
| 安全兜底 | 启动即暂停；游戏不在前台自动挂起、不发键；认不到名字牌就沿用上一次方向，不盲改；退出时抬起所有按键 |

## 还没实现

- 旧方案不识别怪物；`rope_archer` 识别本次模板里的猴子。不寻路、不认地图名、不自动爬绳/走传送门
- 不读游戏内存里的坐标（试过只读内存这条路，被客户端挡住了，已放弃）
- 切地图、改窗口分辨率或缩放后需要重新标定；模板匹配只认标定时的画面尺寸

## 快速开始

### 用打包好的 exe

1. 双击 `dist\MapleFarmVision.exe`。会弹 UAC，点「是」——游戏是管理员权限，本程序权限低的话
   按键会被 Windows（UIPI）静默丢弃，界面看着正常但游戏毫无反应
2. 确认「方案」是 `vision_jump`，「名字模板」指向标定好的名字牌图片
3. 点「试一下识别」先验证：截一张图，弹窗给出主角 x、匹配得分、接下来往哪边打，并画圈的图存到
   `captures\vision_test.png`。这一步不发任何按键
4. 点「启动（挂后台等F12）」→ 切回游戏窗口 → 按 **F12** 开始

### 用源码

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe farm_gui.py     # 界面版，参数在界面上改
.\.venv\Scripts\python.exe farm.py         # 控制台版，参数在 farm.py 顶部的 Config 里
```

源码运行时也要**用管理员权限打开终端**，理由同上。

## 界面与热键

| 操作 | 说明 |
| --- | --- |
| 启动（挂后台等F12） | 绑定窗口、加载模板，然后一直暂停着等 F12 |
| 开始挂机 / 暂停 | 和 F12 等效，切换运行状态 |
| 停止 | 和 F11 等效，退出并抬起所有按键 |
| 截屏标定 | 3 秒后截客户区，先后框选名字牌和平台范围，结果存进 `gui_config.json` |
| 试一下识别 | 只截一张图试识别，不发送按键 |

热键：**F12 开始/暂停，F11 退出**。运行中游戏窗口必须在最前面；切出去会自动挂起不发键，切回来按 F12 继续。
日志区会打印每次判断，例如：

```
[截图] 名字牌 x=688（得分 1.00）在右半边 → 换方向，一直往左打
[移动] 左 89ms   [第1轮]左跳攻(1/1)
[截图] 名字牌 x=688（得分 1.00）在右半边 → 继续一直往左打
```

## 标定：告诉脚本哪个是你的名字

位置判断靠名字牌，所以先要给一次标定：

1. 游戏用**窗口模式**、窗口完整可见，标定后别改分辨率/缩放
2. 点「截屏标定」，3 秒内切回游戏；它会自动截客户区
3. 先用鼠标框住**自己的名字牌文字**（例如 `CatApril`），回车确认
4. 再框住**平台可站立范围**（木板的左右两端），回车确认；按 Esc 可跳过，
   跳过时就按整个画面判断左右（中心=画面中线，兜底也够用）
5. 点「试一下识别」确认得分够高（>0.8 比较稳），有问题就重标一次

名字牌模板就是标定那一刻截图里的一块像素，所以换机器、改分辨率、改名之后都要重标。
标定产物放在程序目录的 `assets\` 下，不进版本库。

## 挂机方案（plans）

在界面「方案」里选，或在 `farm.py` 的 `Config.plan` 里改：

| 方案 | 说明 |
| --- | --- |
| `vision_jump` | **推荐**。截图判断左右 → 单方向持续攻击 → 贴边回中间。需要标定名字牌 |
| `random_jump` | 纯计时版：先后手随机，小概率多打一下 / 纯跳一下 / 发呆一会儿 |
| `fixed_jump` | 纯计时版：固定先右后左各一次，打完回原位 |
| `static_cast` | 站桩连打，每 N 下左右微调一次（示范怎么拼别的打法） |

纯计时版不需要标定，也不会截图；没标定名字牌时 `vision_jump` 会自动退回 `random_jump`。

## 参数

界面里都能调，等价于 `gui_config.json` / `Config` 里的同名字段。时间类参数都是 `(最小, 最大)` 秒的随机区间。

**位置判断（vision_jump 专用）**

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `vision_enabled` | 开 | 关掉就退回纯计时打法，完全不截图 |
| `name_template` | `assets/name_CatApril.png` | 名字牌模板路径（相对程序目录） |
| `match_threshold` | `0.75` | 匹配阈值：真名牌≈1.0，聊天栏文字≈0.6 |
| `vision_interval` | `4.0` 秒 | 隔多久截一张图。越大扫得越远，但也越容易冲过中线才掉头 |
| `platform_left` / `platform_right` | 空 | 平台两端（客户区像素），标定得到；空=按整个画面判断 |
| `edge_margin` | `60` px | 离两端不足这么多算贴边，方向一律指回中间 |
| `rescue_scale` | `2.0` 倍 | 贴边那一轮的移动时长放大倍数（净往中间挪） |

**节奏与按键**

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `attack_key` / `jump_key` | `shift` / `alt` | 攻击键 / 跳跃键 |
| `attacks_per_side` | `1` | 每一次移动后打几下 |
| `move_secs` | 40~120 ms | 单次移动时长，决定每轮挪多远 |
| `jump_hold_secs` | 40~80 ms | 跳跃键按住时长 |
| `jump_rise_secs` | 30~50 ms | 起跳后等多久再出招（太快打地面就调大） |
| `key_hold_secs` | 30~80 ms | 攻击键按住时长 |
| `attack_gap_secs` | 250~450 ms | 两次跳攻之间的间隔（含落地） |
| `settle_secs` | 30~60 ms | 移动后停稳等待 |
| `switch_gap_secs` | 350~550 ms | 轮与轮之间的间隔 |
| `potion_key` | 空（不喝药） | 喝药键，填了才生效 |
| `potion_every` / `potion_pause_secs` | 15 下 / 0.6~1.2 秒 | 每多少下攻击喝一瓶、喝完停多久 |
| `extra_attack_prob` / `hop_prob` / `idle_prob` | 15% / 5% / 8% | 仅 `random_jump`：多打一下 / 纯跳 / 发呆的概率 |
| `micro_move_every` | 25 | 仅 `static_cast`：每多少下做一次左右微调 |

## 工作原理与限制

- **为什么用名字牌而不是人物模型**：名字牌是静态白字，不像角色贴图那样受动作帧、朝向、换装影响，
  一次模板匹配就能定位；同一张图里聊天栏的白字只有 0.6 分左右，不会误命中
- **方向规则**：判定在哪半边就往反方向打，跨过中线才换；所以每一步都朝远离她那一端的方向走，
  越打越靠中间，再加上贴边回中的保护，不会走出平台
- **截图要求游戏在前台且不被遮挡**（本来发按键也要求前台）；认不到名字牌时沿用上一次方向，
  不会因为一次识别失败乱改方向
- **只管左右不管上下**：掉到下层平台后不会自己爬回来
- 客户端不处理 PostMessage 队列消息，只认 SendInput 系统级注入
- 名字牌被特效遮挡、或贴到屏幕边缘被裁掉时可能匹配不到

这类自动化通常违反在线游戏的用户协议，账号风险自负，请只在自己的机器和账号上使用。

## 打包

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\pyinstaller.exe --noconfirm MapleFarmVision.spec
```

- spec 里 `uac_admin=True`（启动自动请求管理员权限）、`console=False`
- 图像匹配要带上 opencv+numpy，所以 exe 约 63 MB；`dist/`、`build/` 都不进版本库

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

14 个用例：名字牌定位与坐标、左右判断与贴边、方案行为（一轮只走一个方向、同侧不换向、跨中线才换）。

## 仓库结构

```
farm.py / farm_gui.py    挂机入口：控制台版 / 界面版
autofarm/
  winapi.py              Win32 封装：找窗口、SendInput、全局热键、权限与客户区查询
  bot.py                 运行控制：F12/F11、失焦挂起、可打断等待、按键兜底释放
  blocks.py              积木：移动、跳攻、连打、喝药、等待
  plans.py               方案拼装：vision_jump / random_jump / fixed_jump / static_cast
  vision.py              截图 + 名字牌匹配 + 左右判断
assets/                  标定产物（名字牌模板等），不进版本库
MapleFarmVision.spec     PyInstaller 打包配置（MapleFarmGUI.spec 只差一个 exe 名字）
tests/                   单元测试
```
