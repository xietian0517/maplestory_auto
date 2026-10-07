"""Cached validation of the screenshot/scene handoff used by the GUI."""
from dataclasses import dataclass
from pathlib import Path
import json
from .semantic import load_request, load_scene


@dataclass(frozen=True)
class SessionState:
    stage: str = 'empty'
    detail: str = ''
    request_valid: bool = False
    ready: bool = False
    image: str = ''


class SessionMonitor:
    def __init__(self):
        self.folder=None; self.signature=None; self.state=SessionState()

    @staticmethod
    def stamp(path):
        try:
            s=path.stat(); return s.st_mtime_ns,s.st_size
        except OSError: return None

    def refresh(self,folder,force=False):
        if folder is None:
            self.folder=None; self.signature=None; self.state=SessionState(); return self.state
        folder=Path(folder)
        image=self.state.image if folder==self.folder else ''
        signature=(self.stamp(folder/'request.json'),self.stamp(folder/'scene.json'),
                   self.stamp(folder/'minimap_calibration.json'),
                   self.stamp(folder/image) if image else None)
        if not force and folder==self.folder and signature==self.signature and self.state.stage!='request_invalid': return self.state
        self.folder=folder
        try: request,_=load_request(folder)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            stage='empty' if not (folder/'request.json').exists() else 'request_invalid'
            self.state=SessionState(stage,str(exc))
        else:
            image=request['image']
            if not (folder/'scene.json').exists():
                self.state=SessionState('waiting',request_valid=True,image=image)
            else:
                try:
                    scene,seed=load_scene(folder)
                    if scene.confidence<.75: raise ValueError('识图置信度不足，请重新识图或截图。')
                    path=folder/'minimap_calibration.json'
                    if not path.exists():raise ValueError('缺少小地图黄点坐标校准，请载入已校准预设。')
                    data=json.loads(path.read_text(encoding='utf-8'))
                    if data.get('request_id')!=scene.request_id:raise ValueError('小地图校准与当前场景不匹配。')
                    from .minimap import MinimapNavigator
                    MinimapNavigator().configure(seed,data)
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    self.state=SessionState('scene_invalid',str(exc),True,image=image)
                else:
                    self.state=SessionState('ready',f'{scene.map_name} · {len(scene.platforms)} 个平台 · {len(scene.ropes)} 根绳索',True,True,image)
        self.signature=(self.stamp(folder/'request.json'),self.stamp(folder/'scene.json'),
                        self.stamp(folder/'minimap_calibration.json'),
                        self.stamp(folder/self.state.image) if self.state.image else None)
        return self.state
