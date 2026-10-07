"""Read-only DeepSeek image/JSON diagnostic. Never imports a game input adapter."""
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'variants/v044_ai'))
import cv2
from autofarm.realtime.strategy import AnchorPolicy
from autofarm.realtime.policy_api import PolicyAPIError
from autofarm.realtime.semantic import atomic_json


def main():
    paths=list((ROOT/'dist/captures/runs').glob('*/latest.png'))
    path=max(paths,key=lambda p:p.stat().st_mtime)
    image=cv2.imread(str(path))
    if image is None:raise ValueError('No recorded game image for diagnostic')
    request=dict(request_id=uuid.uuid4().hex,issued_at=time.perf_counter(),scene_id='read_only_diagnostic',map_epoch=0,
        state=dict(baseline='0.4.4',current_anchor=None,offered_anchors=[],feedback=None,
                   diagnostic='Recorded game image; no permitted goal changes. Return keep. No inputs will be sent.'),image=image)
    try:
        reply=AnchorPolicy(ROOT/'dist/deepseek.txt').propose(request)
        result=dict(ok=True,image_input_tested=True,structured_reply=True,action=reply['action'],
                    input_started=False,baseline='0.4.4',provider='deepseek',model='deepseek-flash')
    except Exception as exc:
        result=dict(ok=False,input_started=False,error=exc.data() if type(exc) is PolicyAPIError else dict(code=type(exc).__name__))
    atomic_json(ROOT/'releases/v0.4.4-ai1/action_api_diagnostic.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
