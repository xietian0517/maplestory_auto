"""Standalone human demonstration recorder. Never starts game automation."""
import ctypes
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

from autofarm.realtime.demonstration import record_demo, VERSION
from autofarm.winapi import WinApi


def app_directory():
    return Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent


def check_window(api,hwnd):
    if api.is_iconic(hwnd): raise ValueError('请先恢复游戏窗口，再开始录制。')
    r=api.client_rect(hwnd);u=api.u
    left,top,width,height=(u.GetSystemMetrics(i) for i in (76,77,78,79))
    if not (left<=r['left'] and top<=r['top'] and r['left']+r['width']<=left+width
            and r['top']+r['height']<=top+height):
        raise ValueError('游戏窗口有一部分在屏幕外。请把游戏和小地图完整放进屏幕。')


class DemoWindow:
    def __init__(self,root):
        self.root=root;self.worker=None;self.stop_event=threading.Event();self.events=queue.Queue()
        self.folder=None;self.closing=False;self.poll_id=None;self.report=None
        self.destination=app_directory()/'demonstrations'
        root.title('冒险岛 · 人工示范录制器 '+VERSION);root.geometry('820x690');root.minsize(760,660)
        root.protocol('WM_DELETE_WINDOW',self.close)
        outer=ttk.Frame(root,padding=20);outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='你来操作，我只记录',font=('Microsoft YaHei UI',21,'bold')).pack(anchor='w')
        ttk.Label(outer,text='独立录制器 · 不挂机 · 不发送按键 · 无需地图 JSON',font=('Microsoft YaHei UI',11)).pack(anchor='w',pady=(4,15))
        steps=('① 停止旧挂机程序，把游戏窗口和小地图完整放在屏幕内。\n'
               '② 点击开始录制，在 5 秒倒计时内切回游戏。\n'
               '③ 正常打怪、换平台、抓绳；10 分钟后自动结束，F11 可提前停止。')
        ttk.Label(outer,text=steps,font=('Microsoft YaHei UI',12),justify='left').pack(anchor='w',pady=(0,15))
        options=ttk.Frame(outer);options.pack(fill='x')
        self.minutes=tk.StringVar(value='10');self.fps=tk.StringVar(value='60')
        ttk.Label(options,text='录制时长（分钟）').pack(side='left')
        self.duration=ttk.Combobox(options,textvariable=self.minutes,values=('2','5','10','20'),width=5,state='readonly');self.duration.pack(side='left',padx=8)
        ttk.Label(options,text='目标帧率').pack(side='left',padx=(16,0))
        self.rate=ttk.Combobox(options,textvariable=self.fps,values=('60','30'),width=5,state='readonly');self.rate.pack(side='left',padx=8)
        buttons=ttk.Frame(outer);buttons.pack(fill='x',pady=15)
        self.start_button=ttk.Button(buttons,text='开始录制我的操作',command=self.start);self.start_button.pack(side='left')
        self.stop_button=ttk.Button(buttons,text='停止并保存',command=self.stop,state='disabled');self.stop_button.pack(side='left',padx=10)
        self.choose_button=ttk.Button(buttons,text='选择保存位置',command=self.choose);self.choose_button.pack(side='left')
        self.title=tk.StringVar(value='准备就绪 · 等你开始录制')
        self.detail=tk.StringVar(value='先停止挂机，再手动示范。录完后把保存目录发给当前对话。')
        self.stats=tk.StringVar(value='目标 60 FPS；界面会显示实际帧率、写入帧数和丢帧。')
        ttk.Label(outer,textvariable=self.title,font=('Microsoft YaHei UI',15,'bold'),foreground='#166534').pack(anchor='w',pady=(8,6))
        ttk.Label(outer,textvariable=self.detail,wraplength=745,justify='left').pack(anchor='w')
        self.progress=ttk.Progressbar(outer,mode='determinate',maximum=100);self.progress.pack(fill='x',pady=12)
        ttk.Label(outer,textvariable=self.stats,wraplength=745,justify='left').pack(anchor='w')
        self.feedback=tk.StringVar(value='经验栏：每秒 5 张无损截图＋首尾快照；攻击期间另存原图样本。尚未自动计算伤害或经验。')
        ttk.Label(outer,textvariable=self.feedback,wraplength=745,justify='left').pack(anchor='w',pady=(8,0))
        self.location=tk.StringVar(value='保存位置：'+str(self.destination))
        ttk.Label(outer,textvariable=self.location,wraplength=745,justify='left').pack(anchor='w',pady=12)
        after=ttk.Frame(outer);after.pack(fill='x')
        self.open_button=ttk.Button(after,text='打开录像目录',command=self.open_folder,state='disabled');self.open_button.pack(side='left')
        self.review_button=ttk.Button(after,text='回看录像与按键',command=self.review,state='disabled');self.review_button.pack(side='left',padx=10)
        self.copy_button=ttk.Button(after,text='复制完成消息',command=self.copy,state='disabled');self.copy_button.pack(side='left')
        ttk.Label(outer,text='只记录游戏在前台时的方向键、Shift / Alt / Ctrl / Space 和 Home / End / Ins / Del / PgUp / PgDn。\n'
                  '切到其他窗口会暂停画面和游戏键采集，倒计时继续。按键时间由目标 240 Hz 轮询观测。',
                  wraplength=745,justify='left',foreground='#555').pack(anchor='w',pady=18)
        self.poll()

    def busy(self): return bool(self.worker and self.worker.is_alive())

    def choose(self):
        if self.busy(): return
        self.destination.mkdir(parents=True,exist_ok=True)
        path=filedialog.askdirectory(initialdir=str(self.destination),title='选择人工示范保存位置')
        if path: self.destination=Path(path);self.location.set('保存位置：'+str(self.destination))

    def start(self):
        if self.busy(): return
        minutes=int(self.minutes.get());fps=int(self.fps.get());self.stop_event.clear();self.report=None
        self.folder=self.destination/('demo_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
        self.location.set('本次目录：'+str(self.folder));self.progress['value']=0
        for button in (self.start_button,self.choose_button,self.open_button,self.review_button,self.copy_button):button.configure(state='disabled')
        self.duration.configure(state='disabled');self.rate.configure(state='disabled');self.stop_button.configure(state='normal')
        def work():
            try:
                for n in range(5,0,-1):
                    self.events.put(dict(phase='countdown',remaining=n))
                    if self.stop_event.wait(1):self.events.put(dict(phase='cancelled'));return
                api=WinApi();hwnd,_=api.find_window('冒险岛怀旧服');check_window(api,hwnd)
                self.folder.mkdir(parents=True,exist_ok=False)
                record_demo(api,hwnd,self.folder,seconds=minutes*60,fps=fps,stop=self.stop_event,notify=self.events.put)
            except Exception as exc:self.events.put(dict(phase='error',error=str(exc)))
            finally:self.events.put(dict(phase='worker_finished'))
        self.worker=threading.Thread(target=work,name='human-demonstration',daemon=False);self.worker.start()

    def stop(self):
        if not self.busy():return
        self.stop_event.set();self.title.set('正在停止并保存录像');self.detail.set('正在写完当前片段与日志。')

    def show_status(self,s):
        phase=s['phase']
        if phase=='worker_finished':
            self.start_button.configure(state='normal');self.choose_button.configure(state='normal')
            self.stop_button.configure(state='disabled');self.duration.configure(state='readonly');self.rate.configure(state='readonly')
            exists=bool(self.folder and self.folder.exists())
            self.open_button.configure(state='normal' if exists else 'disabled')
            self.copy_button.configure(state='normal' if exists else 'disabled')
            self.review_button.configure(state='normal' if exists and (self.folder/'review.html').exists() else 'disabled')
        elif phase=='countdown':
            self.title.set(f'{s["remaining"]} 秒后开始 · 请切回游戏');self.detail.set('由你操作角色，录制器不会替你按键。')
        elif phase=='recording':
            elapsed=s['elapsed_seconds'];total=s['requested_seconds'];self.progress['value']=min(100,elapsed/total*100)
            self.title.set('正在录制你的操作' if s['foreground'] else '已暂停采集 · 游戏不在前台')
            self.detail.set(f'已用 {int(elapsed)//60:02d}:{int(elapsed)%60:02d} / {int(total)//60:02d}:00；有效前台 {s["foreground_seconds"]:.1f} 秒。F11 提前结束。')
            self.stats.set(f'实际采集 {s["recent_capture_fps"]:.1f} FPS / 目标 {s["target_fps"]} FPS · 已写入 {s["frames_written"]} 帧\n'
                           f'编码丢帧 {s["encoder_drops"]} · 捕获跳帧 {s["capture_mailbox_gaps"]} · 按键轮询 {s.get("observed_key_poll_hz",0):.0f} Hz')
            self.feedback.set(f'已保存经验栏原图 {s.get("hud_samples",0)} 张 · 攻击原图 {s.get("combat_samples",0)} 张。伤害与经验尚待识别校验。')
        elif phase=='saving':self.title.set('正在保存最后片段与回看页面')
        elif phase=='cancelled':self.title.set('已取消，未开始录制')
        else:
            self.report=s
            if phase=='error':self.title.set('录制未完成');self.detail.set(s.get('error','未知错误'))
            else:
                self.title.set('录制已保存 · 可以把目录发给我');self.detail.set('点击“复制完成消息”，回到当前对话粘贴。我会分析你的导航、抓绳和打怪动作。')
                if phase=='completed':self.progress['value']=100
            if 'frames_written' in s:
                self.stats.set(f'保存 {s["frames_written"]} 帧 · 前台平均采集 {s["average_capture_fps_foreground"]:.1f} FPS\n'
                               f'前台 {s.get("foreground_seconds",0):.1f} 秒 · 编码丢帧 {s["encoder_drops"]} · 质量详情见 report.json')

    def poll(self):
        self.poll_id=None
        while not self.events.empty():self.show_status(self.events.get())
        if not self.closing:self.poll_id=self.root.after(150,self.poll)

    def open_folder(self):
        if self.folder and self.folder.exists():os.startfile(str(self.folder))

    def review(self):
        if self.folder and (self.folder/'review.html').exists():os.startfile(str(self.folder/'review.html'))

    def copy(self):
        if not self.folder:return
        text=f'人工示范已录好，请检查录制质量，结合录像、按键、combat 攻击原图和 hud 经验栏分析我的导航、抓绳和打怪；以短期有效伤害、长期净经验增长核对收益，无法确认时保留未知，再提出有证据的改进；不要自动启动游戏按键。目录：{self.folder.resolve()}'
        self.root.clipboard_clear();self.root.clipboard_append(text);self.detail.set('完成消息已复制，回到当前对话粘贴即可。')

    def close(self):
        if not self.closing:
            self.closing=True;self.stop()
            if self.poll_id:self.root.after_cancel(self.poll_id);self.poll_id=None
        if self.busy():self.root.after(100,self.close)
        else:self.root.destroy()


if __name__=='__main__':
    ctypes.windll.user32.SetProcessDPIAware()
    root=tk.Tk();DemoWindow(root);root.mainloop()
