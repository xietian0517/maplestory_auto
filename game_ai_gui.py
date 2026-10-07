"""Guided screenshot -> scene understanding -> ready -> play workflow."""
import ctypes
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

# Development goal sessions use a fresh controller process so source fixes can
# be tested while the UI retains its in-memory credentials. No credential file.
if __name__=='__main__' and len(sys.argv)==3 and sys.argv[1]=='--source-job':
    import runpy
    task_path=Path(sys.argv[2]).resolve()
    task_data=json.loads(task_path.read_text(encoding='utf-8'))
    source_root=Path(task_data['source_root']).resolve()
    if not task_path.is_relative_to(source_root) or not (source_root/'game_ai.py').is_file():
        raise ValueError('Invalid development source job')
    sys.path.insert(0,str(source_root))
    runpy.run_path(str(source_root/'scripts'/'goal_worker.py'),run_name='__main__')
    sys.exit(0)

from autofarm.realtime.perception import LatestCapture
from autofarm.realtime.control import Controller
from autofarm.realtime.runtime import attack_hold_seconds, run, run_duration_seconds
from autofarm.realtime.semantic import make_request, load_request, atomic_json
from autofarm.realtime.session import SessionMonitor
from autofarm.winapi import WinApi
from autofarm.realtime.model import MotionProfile

VERSION='0.6.1'

def duration_seconds(value,unit='秒'):
    seconds=float(value)*{'秒':1,'分钟':60,'小时':3600}[unit]
    return run_duration_seconds(seconds)


def prepare_session(folder,output,climb=False):
    import shutil
    req,_=load_request(folder)
    output.mkdir(parents=True,exist_ok=False)
    names=['request.json','scene.json',req['image'],'identity_poses.npy','identity_name.png',
           'monster_templates.npz','world_geometry.json','minimap_calibration.json','parking.json','health.json','initial_motion.json']
    if not climb:names+=['navigation_policy.json','platform_intent.json','farm_plan.json','ai_policy.json','feedback_config.json']
    for name in names:
        if (folder/name).exists():shutil.copy2(folder/name,output/name)
    if not climb and (folder/'feedback_calibration').is_dir():
        shutil.copytree(folder/'feedback_calibration',output/'feedback_calibration')
    return output


def app_directory():
    return Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent


class Window:
    def __init__(self, root):
        self.root=root; self.worker=None; self.folder=None; self.events=queue.Queue()
        self.operation=None; self.phase='idle'; self.phase_note=''; self.notice=''
        self.cancel=threading.Event(); self.task_id=0; self.closing=False
        self.monitor=SessionMonitor(); self.runtime_status=None; self.status_stamp=None
        self.action_model=None
        self.action_provider=None
        self.poll_id=None; self.progress_running=False; self.last_directory=None
        self.base=app_directory(); self.preferences=self.base/'ai_gui_preferences.json'
        try:
            saved=json.loads(self.preferences.read_text(encoding='utf-8'))
            candidate=Path(saved.get('last_scene_directory',''))
            if candidate.is_dir() and candidate.is_absolute(): self.last_directory=candidate
        except (OSError,ValueError,TypeError): pass
        root.title('冒险岛 AI 挂机 · 行动决策试用版 '+VERSION)
        root.geometry('980x700'); root.minsize(940,680); root.protocol('WM_DELETE_WINDOW',self.close)
        outer=ttk.Frame(root,padding=16); outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='准备好地图，再开始挂机',font=('Microsoft YaHei UI',18,'bold')).pack(anchor='w')
        ttk.Label(outer,text='按下面 3 步操作。截图后还要识图；显示“准备就绪”时才能开始挂机。').pack(anchor='w',pady=(4,12))
        cards=ttk.Frame(outer); cards.pack(fill='x'); frames=[]
        self.steps=[tk.StringVar() for _ in range(3)]
        for i,title in enumerate(('① 游戏截图','② AI 识图','③ 开始挂机')):
            cards.columnconfigure(i,weight=1,uniform='step')
            card=ttk.LabelFrame(cards,text=title,padding=10)
            card.grid(row=0,column=i,sticky='nsew',padx=(0 if i==0 else 8,0))
            ttk.Label(card,textvariable=self.steps[i],wraplength=275,justify='left').pack(anchor='w',pady=(0,8))
            frames.append(card)
        self.snapshot_button=ttk.Button(frames[0],text='拍摄新地图（3 秒后截图）',command=self.snapshot)
        self.snapshot_button.pack(fill='x')
        self.load_button=ttk.Button(frames[0],text='使用已有场景文件夹',command=self.load)
        self.load_button.pack(fill='x',pady=(6,0))
        self.preset_button=ttk.Button(frames[0],text='载入当前角色·猴子森林预设',command=self.load_preset)
        self.preset_button.pack(fill='x',pady=(6,0))
        self.source=tk.StringVar(value='conversation'); self.source_buttons=[]
        for value,label in (('conversation','交给当前对话识图'),('online','程序在线识图（需 API 配置）')):
            radio=ttk.Radiobutton(frames[1],text=label,variable=self.source,value=value,command=self.refresh)
            radio.pack(anchor='w'); self.source_buttons.append(radio)
        self.ai_button=ttk.Button(frames[1],command=self.ai_action); self.ai_button.pack(fill='x',pady=(6,0))
        self.start_button=ttk.Button(frames[2],text='开始挂机',command=lambda:self.start(True)); self.start_button.pack(fill='x')
        self.observe_button=ttk.Button(frames[2],text='只观察（不按键）',command=lambda:self.start(False))
        self.observe_button.pack(fill='x',pady=(6,0))
        self.stop_button=ttk.Button(frames[2],text='停止 / F11',command=self.stop); self.stop_button.pack(fill='x',pady=(6,0))
        self.finish_button=ttk.Button(frames[2],text='结束刷怪并回安全点',command=self.finish)
        self.finish_button.pack(fill='x',pady=(6,0))
        status=ttk.LabelFrame(outer,text='当前进度',padding=10); status.pack(fill='x',pady=10)
        self.state_title=tk.StringVar(); self.state_detail=tk.StringVar(); self.next_step=tk.StringVar()
        self.title_label=ttk.Label(status,textvariable=self.state_title,font=('Microsoft YaHei UI',13,'bold'))
        self.title_label.pack(anchor='w')
        ttk.Label(status,textvariable=self.state_detail,wraplength=895,justify='left').pack(anchor='w',pady=(4,0))
        ttk.Label(status,textvariable=self.next_step,wraplength=895,foreground='#185797').pack(anchor='w',pady=(4,0))
        self.progress=ttk.Progressbar(status,mode='indeterminate'); self.progress.pack(fill='x',pady=(6,0))
        self.folder_text=tk.StringVar(value='场景目录：尚未选择')
        ttk.Label(outer,textvariable=self.folder_text,wraplength=910).pack(anchor='w')
        options=ttk.Frame(outer); options.pack(fill='x',pady=(8,2))
        self.seconds=tk.StringVar(value='600'); self.navigate=tk.BooleanVar(value=True)
        self.record=tk.BooleanVar(value=True); self.minimap=tk.BooleanVar(value=True)
        ttk.Label(options,text='挂机时长').pack(side='left')
        ttk.Entry(options,textvariable=self.seconds,width=7).pack(side='left',padx=6)
        self.duration_unit=tk.StringVar(value='秒')
        ttk.Combobox(options,textvariable=self.duration_unit,values=('秒','分钟','小时'),
                     state='readonly',width=4).pack(side='left')
        ttk.Label(options,text='（0=不限时；支持小数）').pack(side='left',padx=6)
        ttk.Label(options,text='攻击按住(秒)').pack(side='left',padx=(14,0))
        self.attack_hold=tk.StringVar(value=f'{Controller().attack_hold_seconds:g}')
        ttk.Entry(options,textvariable=self.attack_hold,width=6).pack(side='left',padx=6)
        ttk.Label(options,text='（0=关闭，默认 0.30）').pack(side='left')
        options=ttk.Frame(outer);options.pack(fill='x',pady=(2,2))
        self.policy_mode=tk.StringVar(value='配置文件')
        ttk.Label(options,text='行动决策').pack(side='left')
        ttk.Combobox(options,textvariable=self.policy_mode,values=('配置文件','规则基线','AI 影子','AI 主动'),
                     state='readonly',width=9).pack(side='left',padx=4)
        ttk.Button(options,text='AI 配置',command=self.api_settings).pack(side='left',padx=4)
        ttk.Checkbutton(options,text='自动导航',variable=self.navigate).pack(side='left',padx=8)
        ttk.Label(options,text='人物定位：小地图黄点（必需）').pack(side='left',padx=8)
        ttk.Checkbutton(options,text='完整录像与按键日志',variable=self.record).pack(side='left',padx=8)
        self.climb_button=ttk.Button(options,text='爬绳测试',command=lambda:self.start(True,True)); self.climb_button.pack(side='right')
        self.stats=tk.StringVar(value='运行后显示人物、小地图和动作状态。')
        ttk.Label(outer,textvariable=self.stats,wraplength=910).pack(anchor='w',pady=4)
        self.log=tk.Text(outer,height=5,wrap='word',state='disabled'); self.log.pack(fill='both',expand=True,pady=(4,6))
        ttk.Label(outer,text='试用版尚未达到人工效率。宠物负责回血；F11 立即停止。完整录像约 1.5 GB/10 分钟。').pack(anchor='w')
        self.refresh(); self.poll_id=root.after(250,self.poll)

    def set_action_api(self,key,model,provider='openai'):
        from autofarm.realtime.policy import PolicyConfig,key_variable
        model=model.strip()
        PolicyConfig.parse(dict(version=1,model=model,provider=provider))
        key=key.strip()
        if key and any(c.isspace() or ord(c)<32 for c in key):
            raise ValueError('API Key 不能包含空白或换行')
        variable=key_variable(provider)
        if not key and not os.environ.get(variable):
            raise ValueError('请填写 API Key')
        if key:os.environ[variable]=key
        self.action_model=model
        self.action_provider=provider
        self.notice='AI 行动配置已生效；密钥仅保留到程序关闭。'
        self.refresh()

    def api_settings(self):
        if self.busy():
            self.notice='请先停止当前任务，再更改 AI 配置。';self.refresh();return
        from autofarm.realtime.policy import load_policy_config,PROVIDER_DEFAULTS
        try:config=load_policy_config(self.folder,model=self.action_model,provider=self.action_provider) if self.folder else None
        except (ValueError,OSError,KeyError):config=None
        initial_provider=config.provider if config else self.action_provider or 'deepseek'
        model=config.model if config else self.action_model or PROVIDER_DEFAULTS[initial_provider]
        dialog=tk.Toplevel(self.root);dialog.title('AI 配置');dialog.transient(self.root)
        panel=ttk.Frame(dialog,padding=16);panel.pack(fill='both',expand=True)
        ttk.Label(panel,text='服务商').grid(row=0,column=0,sticky='w')
        selected_provider=tk.StringVar(value='DeepSeek' if initial_provider=='deepseek' else 'OpenAI')
        choices=ttk.Combobox(panel,textvariable=selected_provider,values=('DeepSeek','OpenAI'),state='readonly',width=49)
        choices.grid(row=1,column=0,sticky='ew',pady=(4,12))
        ttk.Label(panel,text='API Key（仅本次程序使用）').grid(row=2,column=0,sticky='w')
        key=tk.StringVar();entry=ttk.Entry(panel,textvariable=key,show='*',width=52)
        entry.grid(row=3,column=0,sticky='ew',pady=(4,12))
        ttk.Label(panel,text='已配置所选服务商的环境变量时可留空。密钥不写入文件。').grid(row=4,column=0,sticky='w')
        ttk.Label(panel,text='模型名称（行动决策与在线识图）').grid(row=5,column=0,sticky='w',pady=(12,4))
        selected=tk.StringVar(value=model)
        ttk.Entry(panel,textvariable=selected,width=52).grid(row=6,column=0,sticky='ew')
        def changed_provider(event=None):
            selected.set(PROVIDER_DEFAULTS['deepseek' if selected_provider.get()=='DeepSeek' else 'openai'])
            key.set('')
        choices.bind('<<ComboboxSelected>>',changed_provider)
        message=tk.StringVar()
        ttk.Label(panel,textvariable=message,foreground='#a00000',wraplength=430).grid(row=7,column=0,sticky='w',pady=8)
        def save(detect=False):
            try:self.set_action_api(key.get(),selected.get(),'deepseek' if selected_provider.get()=='DeepSeek' else 'openai')
            except ValueError as exc:message.set(str(exc));return
            key.set('');dialog.destroy()
            if detect:self.check_action_api()
        ttk.Label(panel,text='检测会进行一次最小动作请求，产生少量 API 费用；不发送游戏按键。',
                  wraplength=430).grid(row=8,column=0,sticky='w',pady=(0,8))
        buttons=ttk.Frame(panel);buttons.grid(row=9,column=0,sticky='e')
        ttk.Button(buttons,text='应用并检测',command=lambda:save(True)).pack(side='left',padx=6)
        ttk.Button(buttons,text='应用',command=save).pack(side='left')
        dialog.grab_set();entry.focus_set()

    def diagnose_action_api(self):
        from autofarm.realtime.policy import load_policy_config,PolicyConfig,create_action_policy
        from autofarm.realtime.policy_api import PolicyAPIError
        config=load_policy_config(self.folder,model=self.action_model,provider=self.action_provider) if self.folder else PolicyConfig.parse(
            dict(version=1,provider=self.action_provider or 'deepseek',**({'model':self.action_model} if self.action_model else {})))
        provider=create_action_policy(config)
        try:
            details=provider.diagnose()
            result=dict(ok=True,code='ok',description='Key、模型访问与结构化动作请求均通过。',
                        remedy='可以选择 AI 主动，开始短时观察或挂机。',**details)
        except PolicyAPIError as error:result=dict(ok=False,**error.data())
        except Exception as error:
            result=dict(ok=False,code='unknown_error',error_type=type(error).__name__,
                        description='检测发生未分类错误。',remedy='查看错误类型；不会记录原始错误详情。')
        result.update(model=config.model,provider=config.provider,automatic_inputs=False)
        atomic_json(self.base/'action_api_diagnostic.json',result)
        return result

    def check_action_api(self):
        if self.busy():return
        from autofarm.realtime.policy import key_variable,load_policy_config
        provider=self.action_provider or (load_policy_config(self.folder).provider if self.folder else 'deepseek')
        if not os.environ.get(key_variable(provider)):
            self.notice='请先在 AI 配置中填写 Key。';self.refresh();return
        def work(cancel,emit):
            emit('phase',('diagnostic','正在检测 Key、模型访问和结构化动作请求，不发送游戏按键。'))
            result=self.diagnose_action_api()
            emit('diagnostic',result)
        self.task('diagnostic',work)

    def scene_api_config(self):
        from autofarm.realtime.policy import load_policy_config,PolicyConfig
        if self.folder:
            return load_policy_config(self.folder,model=self.action_model,provider=self.action_provider)
        return PolicyConfig.parse(dict(version=1,provider=self.action_provider or 'deepseek',
            **({'model':self.action_model} if self.action_model else {})))

    def scene_api_ready(self):
        from autofarm.realtime.policy import key_variable
        try:return bool(os.environ.get(key_variable(self.scene_api_config().provider)))
        except (ValueError,OSError,KeyError,TypeError):return False

    def scene_planner(self):
        from autofarm.realtime.scene_api import create_scene_planner
        config=self.scene_api_config()
        return create_scene_planner(config.provider,config.model)

    def busy(self):
        return self.operation is not None or bool(self.worker and self.worker.is_alive())

    def write_log(self,text):
        self.log.configure(state='normal'); self.log.insert('end',text+'\n'); self.log.see('end'); self.log.configure(state='disabled')

    def select_folder(self,folder):
        self.folder=Path(folder).resolve(); self.last_directory=self.folder; self.notice=''; self.runtime_status=None
        self.folder_text.set('场景目录：'+str(self.folder))
        try: atomic_json(self.preferences,dict(last_scene_directory=str(self.folder)))
        except OSError as exc: self.write_log('无法保存上次目录：'+str(exc))
        self.refresh(force=True)

    def initial_directory(self):
        # First use opens the actual screenshot root beside this EXE. Later use
        # remembers the selected session, including custom locations.
        for candidate in (self.folder,self.last_directory):
            if candidate and candidate.is_dir(): return candidate
        default=self.base/'captures'; default.mkdir(parents=True,exist_ok=True)
        return default

    def task(self,kind,fn):
        if self.busy(): return
        self.task_id+=1; ident=self.task_id
        self.operation=kind; self.phase=kind; self.phase_note=''; self.notice=''; self.cancel.clear(); self.runtime_status=None
        self.status_stamp=self.monitor.stamp(self.folder/'status.json') if self.folder else None
        def emit(event,value): self.events.put((ident,event,value))
        def target():
            try: fn(self.cancel,emit)
            except Exception as exc:
                from autofarm.realtime.policy_api import PolicyAPIError
                if isinstance(exc,PolicyAPIError):
                    details=exc.data()
                    status=' HTTP '+str(details['http_status']) if details['http_status'] else ''
                    emit('error',details['description']+status+' '+details['remedy'])
                else:emit('error',type(exc).__name__+': '+str(exc))
            finally: emit('finished',None)
        self.worker=threading.Thread(target=target,daemon=True); self.worker.start(); self.refresh()

    @staticmethod
    def countdown(cancel,emit,action):
        for left in (3,2,1):
            emit('phase',('countdown',f'{left} 秒后{action}，请切回游戏。'))
            if cancel.wait(1): return False
        return not cancel.is_set()

    def snapshot(self):
        if self.busy(): return
        folder=self.base/'captures'/('ai_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
        folder.mkdir(parents=True,exist_ok=False); self.select_folder(folder)
        online=self.source.get()=='online'
        def work(cancel,emit):
            if not self.countdown(cancel,emit,'截图'): return
            emit('phase',('capture','正在截取游戏画面。'))
            api=WinApi(); hwnd,_=api.find_window('冒险岛怀旧服')
            with LatestCapture(api,hwnd) as cap:
                frame=cap.next(timeout=5)
                if cancel.is_set(): return
                if not frame or not frame.foreground: raise RuntimeError('截图未完成，请将游戏置于前台后重试。')
                make_request(frame.image,folder)
            emit('log','截图完成。已保存截图请求，接下来需要 AI 识图。')
            if online and self.scene_api_ready() and not cancel.is_set():
                emit('phase',('planning','AI 正在识别人物、平台、绳索和怪物；等待在线模型返回。'))
                self.scene_planner().plan(folder); emit('log','在线识图结果已返回，正在检查能否用于挂机。')
        self.task('snapshot',work)

    def load(self):
        if self.busy(): return
        value=filedialog.askdirectory(title='选择场景文件夹',initialdir=str(self.initial_directory()))
        if value: self.select_folder(value)

    def load_preset(self):
        if self.busy():return
        folder=self.base/'captures'/'monkey_forest_v046'
        if folder.is_dir():self.select_folder(folder)
        else:self.notice='未找到附带地图，请保持 EXE 与 captures 文件夹放在一起。';self.refresh()

    def finish(self):
        if self.operation!='run' or not self.folder:return
        if not (self.folder/'parking.json').exists():return
        (self.folder/'PARK').write_text('finish and park',encoding='utf-8')
        self.notice='已请求回安全点。请切回游戏，等待“安全停靠”；F11 可立即取消。';self.refresh()

    def ai_action(self):
        if self.source.get()=='online': self.plan()
        else: self.copy_request()

    def copy_request(self):
        state=self.monitor.refresh(self.folder,force=True)
        if self.busy() or not state.request_valid: return
        folder=self.planning_folder()
        prompt=(f'请读取这个场景目录的 request.json 和对应截图，识别主角、怪物、平台和绳索，'
                f'生成匹配本次请求的 scene.json 并校验，不启动游戏按键。目录：{folder.resolve()}')
        self.root.clipboard_clear(); self.root.clipboard_append(prompt)
        self.notice='识图请求已复制。回到当前对话粘贴发送；识图完成后，本窗口会自动显示“准备就绪”。'
        self.write_log(self.notice); self.refresh()

    def plan(self):
        state=self.monitor.refresh(self.folder,force=True)
        if self.busy() or not state.request_valid: return
        if not self.scene_api_ready():
            self.notice='在线识图尚未配置。可改选“交给当前对话识图”，复制请求后发送。'; self.refresh(); return
        folder=self.planning_folder()
        def work(cancel,emit):
            emit('phase',('planning','AI 正在识别人物、平台、绳索和怪物；等待在线模型返回。'))
            self.scene_planner().plan(folder); emit('log','在线识图结果已返回，正在检查能否用于挂机。')
        self.task('plan',work)

    def planning_folder(self):
        refresh=self.folder/'refresh'
        return refresh if (refresh/'request.json').exists() else self.folder

    def start(self,live,climb=False):
        if self.busy(): return
        state=self.monitor.refresh(self.folder,force=True)
        if not state.ready:
            self.notice='还不能开始：请先完成截图和 AI 识图，等待“准备就绪”。'; self.refresh(); return
        try:
            seconds=duration_seconds(self.seconds.get(),self.duration_unit.get())
        except ValueError:
            self.notice='请填写有效时长（至少1秒），或填0不限时；可选择秒、分钟、小时。'; self.refresh(); return
        try:
            attack_hold=attack_hold_seconds(float(self.attack_hold.get()))
        except ValueError:
            self.notice='攻击按住时长请填 0（关闭）或 0.05~1.0 秒，常用默认值 0.30。'; self.refresh(); return
        folder=self.folder; navigate=self.navigate.get(); record=self.record.get(); minimap=self.minimap.get()
        if not navigate and ((folder/'farm_plan.json').exists() or (folder/'navigation_policy.json').exists()):
            self.notice='当前预设包含平台策略，请勾选“自动导航”。';self.refresh();return
        online=self.source.get()=='online'
        if online and not self.scene_api_ready():
            self.notice='在线模式尚未配置，请选择“交给当前对话识图”后使用已有场景。'; self.refresh(); return
        from autofarm.realtime.policy import load_policy_config,key_variable
        action_mode={'配置文件':None,'规则基线':'rule','AI 影子':'shadow','AI 主动':'active'}[self.policy_mode.get()]
        try:
            action_config=load_policy_config(folder,action_mode,self.action_model,self.action_provider)
            if climb and action_config.mode!='rule':raise ValueError('爬绳测试请使用规则基线')
            if action_config.mode!='rule' and not os.environ.get(key_variable(action_config.provider)):
                raise ValueError('AI 行动决策尚未配置。请点击“AI 配置”填写 API Key，再选择“AI 主动”或“AI 影子”。')
        except (ValueError,KeyError,OSError) as exc:
            self.notice=str(exc);self.refresh();return
        try:
            new=self.base/'captures'/'runs'/('run_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000))
            folder=prepare_session(folder,new,climb);self.select_folder(folder)
        except (OSError,ValueError,KeyError) as exc:
            self.notice='准备运行目录失败：'+str(exc); self.refresh(); return
        self.run_mode='挂机' if live else '观察'
        def work(cancel,emit):
            if not self.countdown(cancel,emit,'开始'+self.run_mode): return
            api=WinApi(); hwnd,_=api.find_window('冒险岛怀旧服'); adapter=None
            if live:
                from game_input_bridge import GameAdapter
                adapter=GameAdapter(); adapter.validate_target(hwnd)
            if cancel.is_set(): return
            emit('phase',('running','正在连接游戏画面，等待第一帧运行状态。'))
            if (self.base/'goal_backend.json').exists():
                report=self.run_goal_backend(folder,seconds,live,online,climb,navigate,record,minimap,
                                             attack_hold,action_mode,cancel)
                emit('log','本轮结束。'+json.dumps(report,ensure_ascii=False))
                emit('status',json.loads((folder/'status.json').read_text(encoding='utf-8')))
                return
            motion_path=folder/'initial_motion.json'
            motion=MotionProfile.parse(json.loads(motion_path.read_text(encoding='utf-8'))) if motion_path.exists() else None
            parking=live and (folder/'parking.json').exists()
            report=run(api,hwnd,folder,seconds,30,live,adapter,navigate,motion,
                       online=online,climb=climb,record=record and not (live and parking),minimap=minimap,
                       evaluation=record and parking,park_on_finish=parking,hybrid_attacks=True,
                       attack_hold=attack_hold,policy_mode=action_mode,policy_model=self.action_model,
                       policy_provider=self.action_provider)  # 站立射击；跳A 需显式 --jump-attacks 才启用
            emit('log','本轮结束。'+json.dumps(report,ensure_ascii=False))
            emit('status',json.loads((folder/'status.json').read_text(encoding='utf-8')))
        self.task('run',work)

    def run_goal_backend(self,folder,seconds,live,online,climb,navigate,record,minimap,attack_hold,action_mode,cancel):
        import subprocess
        config=json.loads((self.base/'goal_backend.json').read_text(encoding='utf-8'))
        source_root=Path(config['source_root']).resolve()
        if not folder.resolve().is_relative_to(source_root) or not (source_root/'game_ai.py').is_file():
            raise ValueError('目标测试源码路径无效')
        job=folder/'goal_job.json'
        atomic_json(job,dict(source_root=str(source_root),folder=str(folder.resolve()),seconds=seconds,
            live=live,online=online,climb=climb,navigate=navigate,record=record,minimap=minimap,
            attack_hold=attack_hold,policy_mode=action_mode,policy_model=self.action_model,
            policy_provider=self.action_provider))
        command=([sys.executable,'--source-job',str(job)] if getattr(sys,'frozen',False)
                 else [sys.executable,str(source_root/'scripts'/'goal_worker.py'),str(job)])
        with (folder/'backend_console.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=source_root,stdout=log,stderr=log,
                env=dict(os.environ,PYTHONIOENCODING='utf-8'),creationflags=subprocess.CREATE_NO_WINDOW)
            while child.poll() is None:
                if cancel.wait(.2):(folder/'STOP').write_text('GUI stop',encoding='utf-8')
            if child.returncode:raise RuntimeError('目标测试进程未完成，详情见 backend_console.log')
        return json.loads((folder/'report.json').read_text(encoding='utf-8'))

    def stop(self):
        if not self.busy(): return
        self.cancel.set()
        if self.operation=='run' and self.folder: (self.folder/'STOP').write_text('stop',encoding='utf-8')
        planning=self.phase=='planning' or self.operation=='plan'
        self.phase='stopping'
        self.phase_note=('已请求停止。在线识图需等返回或超时，期间不会开始挂机。' if planning else '已请求停止，正在结束操作并释放按键。')
        self.refresh()

    @staticmethod
    def describe_runtime(s):
        reason=s.get('reason','')
        action=s.get('action_policy') or {}
        if action.get('last_model_error'):
            details=action.get('last_model_error_details') or {}
            description=details.get('description',action['last_model_error'])
            status=details.get('http_status')
            title='AI 行动请求失败：'+description+(f'（HTTP {status}）' if status else '')
            remedy=details.get('remedy','停止后进入 AI 配置，点击“应用并检测”。')
            if action.get('model_blocked'):remedy+=' 已暂停模型重试，修正后重新开始。'
            return title,remedy
        if reason.startswith('policy_'):
            return 'AI 行动：'+reason,'行动意图、执行反馈和拒绝原因已写入本轮日志。'
        health=s.get('health') or {}
        cause={'health_unreadable_return':'血量持续无法识别，已触发自动返程。',
               'low_health_return':'已确认血量过低，已触发自动返程。'}.get(health.get('return_reason'),'')
        if s.get('phase')=='parked':return cause+'已安全停靠，所有按键已释放。','可以切出游戏。'
        if s.get('phase')=='parking_unconfirmed':return cause+'已松键，但未完成安全停靠确认。','请查看游戏实际位置，需要时手动停靠。'
        if s.get('phase')=='parking':return cause+'正在结束刷怪并返回安全平台。','请保持游戏前台；到达后还会确认周围怪物和位置稳定。'
        if s.get('phase')=='health_stop':return '已确认血量为0，本轮已停止并释放按键。','请检查角色状态。'
        if reason=='health_confirming':return '正在确认血量数字或新的满血值，暂时松键。','连续确认后继续挂机；持续无法识别会返回安全点。'
        if reason=='attack_reacquire_wait':return '攻击目标短暂不可见，原地复核。','最多等待 0.18 秒；看到目标才继续攻击。'
        if reason=='anchor_wait_respawn':return '在已选刷怪位置等待附近目标。','短时无目标后继续探索。'
        if reason in ('jump_brake','drop_brake'):return '正在停稳，避免滑过起跳或下落位置。','确认落点后继续。'
        if reason=='focus_lost': return '已暂停按键：游戏不在前台。','切回游戏后自动继续。'
        if reason in ('waiting_for_gpt','camera_or_map_changed','window_resized'):
            if reason=='window_resized' or reason=='waiting_for_gpt':
                return '等待当前视野的识图结果，尚未确认地图发生变化。','停止后点击识图按钮更新当前视野；原场景仍保留。'
            return '画面与场景暂时未对齐，不代表已经换地图。',('正在尝试原场景重新定位，并等待在线识图结果。' if s.get('model_connection')=='online' else '正在尝试原场景重新定位；若持续未恢复，停止后复制识图请求，发给当前对话更新视野。')
        if reason in ('player_not_found','identity_confirming'): return '正在重新确认主角位置。','等待人物定位恢复。'
        if reason in ('airborne_or_floor_unknown','navigation_no_safe_recovery'):
            return '等待落地或确认脚下平台，暂不尝试无目标跳跃。','若持续无法确认，需要更新当前视野。'
        if reason=='firing_position_wait':
            return '暂未找到可达、无近怪且能远程攻击的站位。','正在观察怪物位置；不会主动寻路进入怪堆。'
        if reason=='engagement_reacquire_wait':
            return '目标短暂消失，留在原位确认，避免打到一半就走。','未把目标消失直接判定为击杀。'
        if reason=='firing_position_align':
            return '正在平台内调整射击位置，覆盖附近目标。','到位后转向连续攻击。'
        if reason.startswith('calibrat'): return '正在测量移动和跳跃能力。','测量后继续导航与打怪。'
        if reason.startswith(('rope_','climb_')):
            actions={'rope_approach':'正在走到绳子下方。','rope_brake':'正在停稳，避免起跳后横向滑过绳子。',
                     'rope_align':'正在微调绳下位置。','rope_catch':'正在原地跳跃抓绳。',
                     'rope_ascend':'正在向上爬绳。','rope_confirm_landing':'正在确认已站上平台。',
                     'rope_wait_for_landing':'抓绳未成功，等待落地后重新对齐。','rope_catch_failed':'本次抓绳多次失败，重新规划。'}
            actions.update(rope_probe_attachment='正在确认是否挂住绳子，尚未确认上爬。',
                           rope_resume_unconfirmed='没有观察到沿绳上升，停止本次尝试。',
                           rope_no_vertical_progress='爬绳没有上升进展，停止并重新判断。',
                           rope_not_above_player='绳索没有向上延伸，取消向上爬绳。')
            rope=s.get('rope') or {}
            attempts=rope.get('attempts',0)
            return actions.get(reason,'正在爬绳或确认落脚。'),f'本轮已尝试抓绳 {attempts}/3 次；持续检查落脚位置。'
        if 'attack' in reason or 'shift' in s.get('keys',[]): return '正在识别目标并执行攻击。','持续检查目标与朝向。'
        if reason=='stale_frame': return '正在等待更新的游戏画面。','当前画面过旧，暂不继续原动作。'
        return '正在寻找怪物、探索平台或调整站位。','本地持续跟踪人物、小地图和路线。'

    def refresh(self,force=False):
        old_ready=self.monitor.state.ready
        state=self.monitor.refresh(self.folder,force=force)
        if state.ready and not old_ready: self.notice=''; self.write_log('识图结果已通过检查，现在可以开始挂机。')
        busy=self.busy(); online=self.source.get()=='online'; has_key=self.scene_api_ready()
        def enable(button,value): button.configure(state='normal' if value else 'disabled')
        for button in (self.snapshot_button,self.load_button,self.preset_button,*self.source_buttons): enable(button,not busy)
        enable(self.ai_button,not busy and state.request_valid and (not online or has_key))
        self.ai_button.configure(text='开始在线识图 / 重试' if online else '复制识图请求，发给当前对话')
        for button in (self.start_button,self.observe_button,self.climb_button): enable(button,not busy and state.ready and (not online or has_key))
        enable(self.stop_button,busy)
        enable(self.finish_button,self.operation=='run' and getattr(self,'run_mode','')=='挂机'
               and self.folder is not None and (self.folder/'parking.json').exists())
        self.steps[0].set('已完成：游戏截图已保存。' if state.request_valid else '待完成：拍摄当前地图，或载入已有场景。')
        self.steps[1].set('已完成：识图结果已通过检查。' if state.ready else '待完成：选择一种识图方式。截图不代表识图已完成。')
        self.steps[2].set('已就绪：现在可以点击开始挂机。' if state.ready and not busy else '运行中：下方显示当前动作。' if self.operation=='run' else '等待前两步完成，按钮会自动启用。')
        if busy:
            title={'snapshot':'正在准备截图','plan':'AI 正在识图','run':'正在准备运行','diagnostic':'正在检测 AI 连接'}.get(self.operation,'正在处理')
            detail=self.phase_note; next_step=''
            if self.phase=='planning':
                title='第 2 步 · AI 正在识图'; self.steps[1].set('处理中：等待在线模型返回。')
                next_step='识图完成并检查通过后，“开始挂机”才会亮起；不会自动开始按键。'
            elif self.phase=='countdown': title='倒计时 · 请切回游戏'; next_step='可点击“停止”取消。'
            elif self.phase=='stopping': title='正在停止'
            elif self.phase=='running':
                title='正在'+getattr(self,'run_mode','运行'); next_step='按 F11 或点击“停止”结束本轮。'
                if self.runtime_status:
                    detail,action=self.describe_runtime(self.runtime_status); next_step=action+' '+next_step
                    if self.runtime_status.get('planner_error'):
                        detail='当前视野的在线识图失败。'; next_step='停止后重试识图，或改用当前对话。'
                    if getattr(self,'run_mode','')=='观察': detail='只观察，不发送按键。'+detail.replace('执行攻击','判断攻击目标')
        elif state.ready:
            title='准备就绪 · 可以开始挂机'; detail='截图与识图结果已匹配。'+state.detail
            next_step='下一步：点击“开始挂机”，在 3 秒内切回游戏。'
            if online and not has_key:
                title='场景已就绪 · 在线模式尚未配置'; next_step='选择“交给当前对话识图”，即可使用已有场景开始挂机。'
            if self.runtime_status and self.runtime_status.get('phase') in ('parked','parking_unconfirmed','health_stop'):
                title,detail=self.describe_runtime(self.runtime_status)
        elif not state.request_valid:
            title='第 1 步 · 请先截图' if state.stage=='empty' else '截图请求不可用 · 请重新截图'
            detail='AI 尚未开始识图。先把游戏切到要挂机的地图。'
            next_step='下一步：点击“拍摄新地图”，3 秒内切回游戏；或选择已有场景文件夹。'
        else:
            title='第 2 步 · 等待 AI 识图' if state.stage=='waiting' else '识图结果未通过检查 · 还不能挂机'
            if online:
                detail='截图已完成，在线识图尚未开始。' if has_key else '截图已完成；本机未配置在线 API，程序尚未调用 AI。'
                next_step='下一步：点击“开始在线识图 / 重试”。' if has_key else '选择“交给当前对话识图”，复制识图请求并发送。'
            else:
                detail='当前对话模式：需要你发送识图请求，程序不会在后台自动调用 AI。'
                next_step='下一步：点击“复制识图请求”，回到当前对话粘贴发送。完成后自动显示“准备就绪”。'
            if state.stage=='scene_invalid': detail='结果不完整、置信度不足，或与当前截图不匹配。需要重新识图。'
        self.state_title.set(title); self.state_detail.set(detail); self.next_step.set(self.notice or next_step)
        self.title_label.configure(foreground='#17723b' if state.ready and not busy else '#185797')
        animate=busy and self.phase in ('snapshot','plan','capture','planning','countdown')
        if animate and not self.progress_running:
            self.progress.configure(mode='indeterminate'); self.progress.start(15)
        elif not animate and self.progress_running: self.progress.stop()
        self.progress_running=animate
        if not animate:
            self.progress.configure(mode='determinate')
            self.progress['value']=100 if state.ready else 35 if state.request_valid else 0
        if self.runtime_status and self.operation=='run':
            s=self.runtime_status; mini=s.get('minimap') or {}
            names={'tracking':'已定位，辅助导航','learning_scale':'已定位，学习比例','disabled':'已关闭',
                   'not_found':'未找到，请保持小地图完整可见','confirming':'确认中','marker_missing':'黄点暂不可见','marker_ambiguous':'黄点不唯一'}
            self.stats.set(f'人物：{"可见" if s.get("player_visible") else "定位中"}  |  怪物：{s.get("monsters",0)}  |  '
                           f'小地图：{names.get(mini.get("state"),"重新定位中")}  |  有效决策：{s.get("valid_decision_hz",0):.1f} 次/秒'
                           +f'  |  行动来源：{(s.get("action_policy") or {}).get("last_source","规则")}')
        elif not busy: self.stats.set('未运行，不发送按键。')

    def poll(self):
        self.poll_id=None
        while not self.events.empty():
            ident,event,value=self.events.get()
            if ident!=self.task_id: continue
            if event=='phase' and not self.cancel.is_set(): self.phase,self.phase_note=value
            elif event=='log': self.write_log(value)
            elif event=='diagnostic':
                status=' HTTP '+str(value['http_status']) if value.get('http_status') else ''
                self.notice=('AI 检测通过：' if value['ok'] else 'AI 检测失败：')+value['description']+status+' '+value['remedy']
                self.write_log(self.notice)
            elif event=='status':self.runtime_status=value;self.notice=''
            elif event=='error': self.notice='本次操作未完成：'+value; self.write_log(self.notice)
            elif event=='finished':
                kind=self.operation; self.operation=None; self.phase='idle'; self.phase_note=''
                if self.cancel.is_set() and not self.notice: self.notice='已停止，可以按当前进度继续操作。'
                elif kind=='run' and not self.notice: self.notice='本轮已结束。场景有效时，可以再次点击“开始挂机”。'
                self.refresh(force=True)
        if self.folder and self.operation=='run' and self.phase=='running':
            path=self.folder/'status.json'; stamp=self.monitor.stamp(path)
            if stamp and stamp!=self.status_stamp:
                try: self.runtime_status=json.loads(path.read_text(encoding='utf-8')); self.status_stamp=stamp
                except (OSError,ValueError): pass
        self.refresh()
        if not self.closing: self.poll_id=self.root.after(250,self.poll)

    def close(self):
        if not self.closing:
            self.closing=True; self.stop()
            if self.poll_id: self.root.after_cancel(self.poll_id); self.poll_id=None
        if self.worker and self.worker.is_alive(): self.root.after(100,self.close)
        else: self.progress.stop(); self.root.destroy()


if __name__=='__main__':
    ctypes.windll.user32.SetProcessDPIAware()
    root=tk.Tk()
    if len(sys.argv)==3 and sys.argv[1]=='--self-test':
        root.withdraw();window=Window(root);window.load_preset();root.update()
        from autofarm.realtime.demo_video import H264Writer
        import numpy as np
        video=Path(sys.argv[2]).with_suffix('.mp4')
        writer=H264Writer(video,30,(64,64));writer.write(np.zeros((64,64,3),np.uint8));writer.release()
        from autofarm.realtime.actions import ActionIntent
        from autofarm.realtime.pipeline import DecisionPipeline
        from autofarm.realtime.pipeline_replay import replay_trace
        from autofarm.realtime.policy import PolicyConfig
        from autofarm.realtime.model import Actor,Box,Observation,Platform
        import tempfile
        with tempfile.TemporaryDirectory() as trial:
            policy=DecisionPipeline(Controller(),PolicyConfig(mode='active',hierarchical=True),folder=trial,replay=True)
            o=Observation(1,1,Actor(Box(285,52,315,100),.99),[],[Platform('floor',0,700,100)])
            advice=ActionIntent('smoke','ai','smoke',1,0,'smoke_scene',1,16,'farm','floor',300,'right',60)
            for index in range(15):
                stamp=1+index*.05;o.captured_at=stamp;o.frame_id=index+1
                d=policy.step(o,stamp,scene_id='smoke_scene',delivery=advice if index==0 else None)
                policy.commit(d,False,stamp,o,simulated=True)
            policy.close(1.8)
            pipeline_ok=d.keys==frozenset({'shift'}) and replay_trace(trial)['identical']
        atomic_json(Path(sys.argv[2]),dict(version=VERSION,ready=window.monitor.state.ready,
            start_enabled=not window.start_button.instate(['disabled']),
            default_seconds=window.seconds.get(),default_attack_hold=window.attack_hold.get(),
            input_started=window.worker is not None,
            preset=str(window.folder),frozen=bool(getattr(sys,'frozen',False)),encoder_ok=video.stat().st_size>0,
            action_policy_modes=['rule','shadow','active'],policy_pipeline_ok=pipeline_ok,
            action_api_configured=window.scene_api_ready(),action_api_providers=['openai','deepseek'],
            preset_provider=window.scene_api_config().provider,preset_model=window.scene_api_config().model))
        window.close()
    else:
        window=Window(root)
        if '--goal-trial' in sys.argv:
            window.load_preset();window.seconds.set('600');window.duration_unit.set('秒')
            window.policy_mode.set('AI 主动');window.record.set(True)
            window.notice='10 分钟目标测试已准备。先配置并检测 DeepSeek；检测完成后由操作者开始。'
            window.refresh();root.after(200,window.api_settings)
        root.mainloop()
