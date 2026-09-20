# 冒险岛视觉脚本

怀旧版内存读取开发与实测阻塞情况见 [MEMORY.md](MEMORY.md)。目前已实现底层只读模块和诊断工具，尚未取得游戏对象数据，视觉版仍是现有运行入口。

Windows / Python 3.10+。截图 → 模板检测 → 寻怪/巡逻 → 按职业选择技能 → Windows SendInput。

这是需要客户端标定的可运行基础版。没有附带真实游戏素材，也没有完成实机验证；不能开箱即用识别所有地图或所有职业。已实现角色、怪物位置检测、小地图边缘轮廓与黄色玩家点检测、同层寻怪、配置路线移动、通用技能配置。地图轮廓只是边缘图，不等于完整地图几何、碰撞信息或自动跨平台寻路；当前不识别地图名称和地图 ID。切换地图后先暂停并重新标定。

## 启动

当前目录已安装 `.venv` 依赖，并已创建 `config.json`，可以直接开始编辑配置和标定。只有配置不存在时才运行下面的初始化命令：

```powershell
cd E:\work\mxd
.\.venv\Scripts\python.exe -m maplebot init
```

新机器安装：`python -m venv .venv`，然后 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`。

编辑 `config.json` 的 `window_title`，填实际游戏窗口标题的一段唯一文字（支持中文）。使用窗口模式，保持分辨率、缩放不变，窗口需要完整可见。按以下步骤准备素材：

```powershell
# 3 秒内把游戏切到前台，保存客户区截图
.\.venv\Scripts\python.exe -m maplebot capture
# 每次框选后按 Enter 确认，Esc 取消
.\.venv\Scripts\python.exe -m maplebot calibrate scene
.\.venv\Scripts\python.exe -m maplebot calibrate minimap
.\.venv\Scripts\python.exe -m maplebot calibrate player
.\.venv\Scripts\python.exe -m maplebot calibrate monster
```

- `scene`：只框游戏战斗画面，尽量排除小地图、聊天栏、快捷栏。
- `minimap`：只框小地图的地图内容，排除标题、按钮和边框。
- `player`：紧密框选自己的角色或独有名牌。建议使用角色整体，模板中心作为攻击距离的参考点。名牌模板需要相应调整技能距离。
- `monster`：紧密框选怪物。同一种怪物可重复添加不同动画帧；多种怪物分别添加。左右镜像自动支持。
- 多次 `capture --output captures/another.png` 后，可用 `calibrate player --image captures/another.png` 添加其他动作帧。使用相同窗口尺寸。

先离线检查，再打开实时预览：

```powershell
.\.venv\Scripts\python.exe -m maplebot analyze
.\.venv\Scripts\python.exe -m maplebot run --preview
```

`analyze` 保存 `captures/analysis.png` 和小地图轮廓图，同时输出位置与动作 JSON。绿色表示角色，红色表示怪物。实时预览要放在游戏窗口之外，不能遮住游戏。默认运行只模拟决策，不发送按键。

确认识别后：

```powershell
.\.venv\Scripts\python.exe -m maplebot run --live
```

实际按键模式初始暂停。切到游戏后 **F8 开始/暂停，F9 退出**。失去游戏焦点会松开按键并暂停，回到游戏后需重新按 F8。角色未检测到或有多个近似匹配时不操作；移动卡住会停止，F8 暂停再开启可以重置。异常退出会释放本程序按下的键。

## 职业和攻击技能

将 `profile` 改成 `mage`、`warrior`、`archer` 或自行新增的职业名称。示例仅演示键位和规则，**所有距离、冷却和技能名称均需按客户端实测校准**，并非游戏数据库。SHIFT/CTRL 只是用户指定的键位。

每个技能支持：

| 字段 | 含义 |
| --- | --- |
| `name` / `key` | 唯一技能名 / 游戏内绑定按键 |
| `range_x` / `range_y` | 战斗画面内两个模板中心之间允许的横向/纵向像素距离 |
| `directional` | 是否必须先面向目标 |
| `priority` | 数值越大越先尝试 |
| `cooldown` | 两次技能启动的最短间隔，秒 |
| `hold` | 按住技能键的时间，秒，最多 0.5 |
| `recovery` | 松键后的动作等待时间，秒 |

通用逻辑会寻找范围内可命中的目标，按优先级释放冷却结束的技能；有目标在范围内但技能未就绪时等待。没有读取 MP、技能等级、实际施放成功与服务端冷却，因此当前冷却属于本地计时。支持 `a`~`z`、`0`~`9`、`shift`、`ctrl`、`alt`、`space` 和四个方向键。F8/F9 保留。

## 巡逻与上下移动

同层有怪物时向怪物移动。没有同层目标时，按 `navigation.waypoints` 循环巡逻。坐标来自**小地图 ROI 左上角**，不是战斗画面坐标。玩家点默认用黄色 HSV 区间识别；若小地图有多个同色标记，脚本暂停路线移动，需要调整颜色范围/ROI。

例如先走到梯子处，再上爬，最后返回。将下例坐标替换成你的地图坐标：

```json
"waypoints": [
  {"x": 40, "y": 80},
  {"x": 40, "y": 35, "vertical_keys": ["up"]},
  {"x": 110, "y": 35},
  {"x": 110, "y": 80, "vertical_keys": ["down", "space"]}
]
```

程序先横向对齐，再发送目标点指定的 `vertical_keys`；组合键同步按住。`space` 必须在游戏中绑定跳跃，`down+space` 的效果也取决于客户端。路线应逐段标定，不会自动识别梯子、传送门或计算跳跃轨迹。没有路线时，无目标就等待。

## 限制与验证

模板匹配受动画、遮挡、特效、换装和缩放影响；多补充真实帧模板并通过离线图检查。纵向距离只是同层的近似判断，不能判断障碍物和平台连通性。截图窗口被其他窗口遮挡也会影响识别。部分客户端可能不接受 SendInput 或截图结果为空，本程序不包含驱动或兼容绕过；出现这些情况需先诊断客户端兼容性。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试覆盖职业选择、技能方向/优先级/冷却、角色丢失、上下路线、卡住停止，以及合成图片的多目标检测、坐标偏移和歧义拒绝。测试不代表真实客户端识别效果或按键已经验证。

实现参考：[OpenCV 模板匹配](https://docs.opencv.org/4.x/de/da9/tutorial_template_matching.html)、[MSS 截图示例](https://python-mss.readthedocs.io/latest/examples.html)。
