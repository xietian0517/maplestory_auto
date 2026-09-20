# 怀旧版内存读取：当前进度

## 最新结果

静态元数据与原生类型表分析已完成一轮，见 [结构报告](analysis/il2cpp/REPORT.md) 和 [位置更新链路](analysis/il2cpp/METHOD_FINDINGS.md)。已恢复字段类型与编译时偏移，并发现与 Transform.set_position 相连的候选路径，业务身份和实时地址尚未确认。

进一步实测：脚本与游戏都已是管理员权限；PSAPI 模块枚举同样失败。VirtualQueryEx 可查询内存区域，但实际 ReadProcessMemory 连续32次错误5、成功读取0字节。下面的初期记录保留排查过程，最新结论以此段和报告为准。

## 初期诊断记录

目标目录：`D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic`。

已添加 `maplebot.memory.ProcessReader`：只读进程访问、模块枚举、精确字节读取、标量读取、32/64 位指针链解析、异常及句柄释放。没有写内存或注入代码。

**尚未完成游戏数据解码和内存驱动自动攻击。** 不需要框选怪物是这个方案的目标，但不能把一个通用 ReadProcessMemory 封装当作已经能读取本客户端怪物列表。

本机诊断发现：

- 正在运行的是 `Maplestory_Classic.exe`，本次 PID 为 `12704`；重启后 PID 会改变。
- 安装目录包含 `GameAssembly.dll` 及 IL2CPP 元数据，文件指纹已记录在 `captures/memory-probe.json`。
- `OpenProcess(PROCESS_VM_READ)` 成功。
- `CreateToolhelp32Snapshot` 枚举模块时返回 Windows 错误 5（拒绝访问）。没有取得模块基址，未执行游戏模块内存读取；因此报告的 `read_access: null` 表示未验证，而不是读取成功或已确认 ReadProcessMemory 被拒绝。
- 尚未确定角色对象、怪物容器、地图平台/梯子数据的字段布局与运行时定位方式。安装文件本身不提供实时怪物坐标。

## 重新诊断

```powershell
cd E:\work\mxd
Get-Process Maplestory_Classic | Select-Object Id
.\.venv\Scripts\python.exe -m maplebot.memory --pid 12704 --client 'D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic'
```

将 PID 换成实际值。命令不发送游戏按键，输出 JSON 报告。当前会话无法确定错误 5 是进程权限还是客户端保护造成的；如游戏以管理员身份运行，可手动在同等权限的终端重试诊断。即使取得基址，仍需继续定位并验证真实游戏对象布局，不能直接启用自动攻击。

后续接入应将玩家、怪物、地图几何统一为世界坐标，单独校准技能范围；现有视觉版的屏幕像素距离不能直接复用。地图切换和实体容器变化时还需要快照一致性检查。

底层模块通过独立子进程的真实跨进程读取测试以及指针链、空指针、无效地址、关闭句柄测试。这些测试不代表对游戏进程读取已成功。

API 依据：[Microsoft ReadProcessMemory](https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-readprocessmemory)、[模块快照](https://learn.microsoft.com/en-us/windows/win32/toolhelp/snapshots-of-the-system)。
