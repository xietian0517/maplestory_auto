"""Small diagnostic UI for the experimental GPT/local controller."""
import ctypes
import json
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

from autofarm.realtime.model import MotionProfile
from autofarm.realtime.perception import LatestCapture
from autofarm.realtime.runtime import run
from autofarm.realtime.semantic import make_request,OpenAIPlanner
from autofarm.winapi import WinApi


class Window:
    def __init__(self,root):
        self.root=root; self.worker=None; self.messages=queue.Queue(); self.folder=None
        self.root.title('冒险岛 AI 实时控制 · 实验版 0.2')
        self.root.geometry('850x590'); self.root.protocol('WM_DELETE_WINDOW',self.close)
        outer=ttk.Frame(root,padding=18); outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='GPT 地图理解 + 本地实时控制',font=('Microsoft YaHei UI',18,'bold')).pack(anchor='w')
        ttk.Label(outer,text='实验版：支持 AI 场景交接与快速跟踪；跨地图泛化尚待验证。').pack(anchor='w',pady=8)
        controls=ttk.Frame(outer); controls.pack(fill='x')
        ttk.Button(controls,text='新地图截图',command=self.snapshot).pack(side='left')
        ttk.Button(controls,text='载入场景文件夹',command=self.load).pack(side='left',padx=8)
        ttk.Button(controls,text='GPT 在线识图',command=self.plan).pack(side='left')
        self.folder_text=tk.StringVar(value='先采集地图，再由当前对话或在线模型生成 scene.json')
        ttk.Label(outer,textvariable=self.folder_text,wraplength=800).pack(anchor='w',pady=10)
        options=ttk.Frame(outer); options.pack(fill='x')
        self.seconds=tk.StringVar(value='60'); self.navigate=tk.BooleanVar(value=True)
        self.record=tk.BooleanVar(value=True)
        ttk.Label(options,text='本轮时长（秒）').pack(side='left')
        ttk.Entry(options,textvariable=self.seconds,width=8).pack(side='left',padx=6)
        ttk.Checkbutton(options,text='导航与自动运动测量',variable=self.navigate).pack(side='left',padx=12)
        ttk.Checkbutton(options,text='保存回放（最多 160 帧）',variable=self.record).pack(side='left')
        buttons=ttk.Frame(outer); buttons.pack(fill='x',pady=12)
        ttk.Button(buttons,text='只观察测速',command=lambda:self.start(False)).pack(side='left')
        ttk.Button(buttons,text='开始按键测试',command=lambda:self.start(True)).pack(side='left',padx=8)
        ttk.Button(buttons,text='测试爬绳',command=lambda:self.start(True,True)).pack(side='left',padx=8)
        ttk.Button(buttons,text='停止 / F11',command=self.stop).pack(side='left')
        self.status=tk.StringVar(value='未运行')
        ttk.Label(outer,textvariable=self.status,font=('Microsoft YaHei UI',11),wraplength=800).pack(anchor='w')
        self.log=tk.Text(outer,height=14,wrap='word'); self.log.pack(fill='both',expand=True,pady=10)
        ttk.Label(outer,text='启动后有 3 秒切回游戏；关闭原挂机助手。切走窗口自动松键。\n有效 API 密钥只从本机 OPENAI_API_KEY 读取；不在界面输入或保存密钥。').pack(anchor='w')
        self.root.after(250,self.poll)

    def task(self,fn):
        if self.worker and self.worker.is_alive(): self.messages.put('已有任务运行，请先停止。'); return
        def target():
            try: fn()
            except Exception as e: self.messages.put(type(e).__name__+': '+str(e))
        self.worker=threading.Thread(target=target,daemon=True); self.worker.start()

    def snapshot(self):
        if self.worker and self.worker.is_alive(): self.messages.put('请先停止当前任务。'); return
        base=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent
        self.folder=base/'captures'/('ai_'+time.strftime('%Y%m%d_%H%M%S'))
        self.folder.mkdir(parents=True,exist_ok=True); self.folder_text.set(str(self.folder))
        def work():
            self.messages.put('3 秒后截图，请切回游戏。'); time.sleep(3)
            api=WinApi(); hwnd,_=api.find_window('冒险岛怀旧服')
            with LatestCapture(api,hwnd) as cap:
                frame=cap.next(timeout=5)
                if not frame or not frame.foreground: raise RuntimeError('请将游戏置于前台')
                make_request(frame.image,self.folder)
            self.messages.put('已保存 request.json 和地图图片；等待 AI 生成同目录 scene.json。')
        self.task(work)

    def load(self):
        if self.worker and self.worker.is_alive(): self.messages.put('请先停止当前任务。'); return
        value=filedialog.askdirectory(title='选择含 request.json 和 scene.json 的文件夹')
        if value: self.folder=Path(value); self.folder_text.set(value)

    def plan(self):
        if not self.folder: return
        self.task(lambda:(OpenAIPlanner().plan(self.folder),self.messages.put('在线 GPT 场景已接收并校验。')))

    def start(self,live,climb=False):
        if self.worker and self.worker.is_alive(): self.messages.put('已有任务运行。'); return
        if not self.folder: self.messages.put('请先采集或载入地图。'); return
        try: seconds=float(self.seconds.get())
        except ValueError: self.messages.put('请输入有效时长。'); return
        if not 1<=seconds<=600: self.messages.put('时长范围为 1–600 秒。'); return
        folder=self.folder; navigate=self.navigate.get(); record=self.record.get()
        if record and (folder/'replay'/'frames.jsonl').exists():
            # Keep earlier evidence and a fresh STOP lifecycle for every run.
            import shutil
            from autofarm.realtime.semantic import load_request
            req,_=load_request(folder)
            new=folder.parent/(folder.name+'_run_'+time.strftime('%H%M%S'))
            new.mkdir(parents=True,exist_ok=False)
            for name in ('request.json','scene.json',req['image']):
                if (folder/name).exists(): shutil.copy2(folder/name,new/name)
            folder=new; self.folder=new; self.folder_text.set(str(new))
        if (folder/'STOP').exists(): (folder/'STOP').unlink()
        def work():
            api=WinApi(); hwnd,_=api.find_window('冒险岛怀旧服'); adapter=None
            if live:
                from game_input_bridge import GameAdapter
                adapter=GameAdapter(); adapter.validate_target(hwnd)
            # Re-measure each live run: haste may have expired since last time.
            motion=None
            self.messages.put('3 秒后开始，请关闭原助手并切回游戏。'); time.sleep(3)
            report=run(api,hwnd,folder,seconds,30,live,adapter,navigate,motion,climb=climb,record=record)
            self.messages.put(json.dumps(report,ensure_ascii=False,indent=2))
        self.task(work)

    def stop(self):
        if self.folder: (self.folder/'STOP').write_text('stop',encoding='utf-8')
        self.messages.put('已请求停止，控制循环会松开按键。')

    def poll(self):
        while not self.messages.empty():
            self.log.insert('end',self.messages.get()+'\n'); self.log.see('end')
        if self.folder and (self.folder/'status.json').exists():
            try:
                s=json.loads((self.folder/'status.json').read_text(encoding='utf-8'))
                latency=s.get('frame_to_decision_ms') or {}
                running=bool(self.worker and self.worker.is_alive())
                self.status.set(f'{"运行中" if running else "已结束"} · {s["mode"]} · 有效 {s["valid_decision_hz"]} 次/秒 · '
                    f'P99 {latency.get("p99","—")} ms · {s["reason"]} · 怪物 {s["monsters"]}')
            except (OSError,ValueError,KeyError): pass
        self.root.after(250,self.poll)

    def close(self):
        if not getattr(self,'closing',False): self.stop(); self.closing=True
        if self.worker and self.worker.is_alive(): self.root.after(100,self.close)
        else: self.root.destroy()


if __name__=='__main__':
    ctypes.windll.user32.SetProcessDPIAware()
    root=tk.Tk(); Window(root); root.mainloop()
