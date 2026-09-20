# 怀旧版静态分析入口

结果见 [IL2CPP 结构报告](analysis/il2cpp/REPORT.md) 和 [位置更新链路](analysis/il2cpp/METHOD_FINDINGS.md)。这些是文件静态分析结果，当前游戏内存读取仍未成功。

在项目目录执行以下命令可重新生成结果，输入文件只读：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-analysis.txt
.\.venv\Scripts\python.exe -m maplebot.metadata --metadata 'D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic\Maplestory_Classic_Data\il2cpp_data\Metadata\global-metadata.dat' --binary 'D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic\GameAssembly.dll'
.\.venv\Scripts\python.exe -m maplebot.method_analysis --binary 'D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic\GameAssembly.dll'
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

解析器支持该客户端的 v39 布局，方法重点类型索引绑定已记录的文件指纹，不会把旧索引用到未知新版本。原始字段名、类型名与语义推测分开保存。

`python -m maplebot.target_probe --pid <实际PID> --client <客户端目录>` 用固定目标“小棒槌啊”和“射手村”等进行明文检索，运行时搜索限定已提交可读私有区域，最多30秒或4GiB；连续32次没有任何成功读取就停止。它只保存命中位置和统计，不保存进程内存转储，命中字符串本身也不等于确认角色对象。
