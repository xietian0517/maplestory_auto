"""Offline DeepSeek transport, image, credentials, grounding and diagnostics checks."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

import cv2
import numpy as np

from autofarm.realtime.policy import (PolicyConfig,PolicyRequest,DeepSeekActionPolicy,
    OpenAIActionPolicy,create_action_policy,load_policy_config)
from autofarm.realtime.policy_api import PolicyAPIError
from autofarm.realtime.scene_api import DeepSeekPlanner,create_scene_planner
from autofarm.realtime.semantic import make_request,atomic_json,PlannerError


def completion(value,finish='stop'):
    return io.StringIO(json.dumps(dict(choices=[dict(finish_reason=finish,
        message=dict(role='assistant',content=json.dumps(value)))])))


class ActionOpener:
    def __init__(self):self.requests=[];self.reply=None;self.finish='stop'
    def open(self,http,timeout):
        self.requests.append(http)
        if http.get_method()=='GET':return io.StringIO('{"data":[{"id":"deepseek-flash"}]}')
        self.body=json.loads(http.data)
        request=json.loads(self.body['messages'][1]['content'][0]['text'])
        reply=dict(request_id=request['request_id'],frame_id=request['frame_id'],map_epoch=request['map_epoch'],
            action='wait',target='',world_x=None,direction='',duration=.05,reason='offline')
        if self.reply:reply.update(self.reply)
        return completion(reply,self.finish)


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,{'DEEPSEEK_API_KEY':'deepseek-offline-secret',
            'OPENAI_API_KEY':'openai-offline-secret'});self.env.start()
        self.request=PolicyRequest('request','scene',42,3,1,15,{'available_actions':['wait']},
            (np.zeros((32,32,3),np.uint8),))
    def tearDown(self):self.env.stop()

    def test_provider_defaults_old_config_and_explicit_override(self):
        self.assertEqual(PolicyConfig.parse({'version':1}).provider,'openai')
        cfg=PolicyConfig.parse({'version':1,'provider':'deepseek'})
        self.assertEqual(cfg.model,'deepseek-flash');self.assertIsInstance(create_action_policy(cfg),DeepSeekActionPolicy)
        self.assertIsInstance(create_action_policy(PolicyConfig()),OpenAIActionPolicy)
        with tempfile.TemporaryDirectory() as folder:
            atomic_json(Path(folder)/'ai_policy.json',dict(version=1,model='gpt-6-astra'))
            selected=load_policy_config(folder,provider='deepseek')
            self.assertEqual(selected.model,'deepseek-flash')
        with self.assertRaises(ValueError):PolicyConfig.parse(dict(version=1,provider='unknown'))

    def test_image_request_endpoint_key_isolation_and_bound_reply(self):
        opener=ActionOpener();intent=DeepSeekActionPolicy(opener=opener).propose(self.request)
        http=opener.requests[0]
        self.assertEqual(http.full_url,'https://api.deepseek.com/chat/completions')
        self.assertEqual(http.get_header('Authorization'),'Bearer deepseek-offline-secret')
        self.assertNotIn('secret',http.data.decode())
        self.assertEqual(opener.body['thinking'],{'type':'disabled'})
        self.assertEqual(opener.body['response_format'],{'type':'json_object'})
        self.assertTrue(opener.body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,'))
        self.assertEqual(intent.frame_id,42);self.assertEqual(intent.map_epoch,3);self.assertEqual(intent.action,'wait')

    def test_other_provider_key_cannot_authorize_deepseek(self):
        opener=ActionOpener()
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':''}),self.assertRaises(PolicyAPIError) as caught:
            DeepSeekActionPolicy(opener=opener).propose(self.request)
        self.assertEqual(caught.exception.code,'missing_key');self.assertEqual(opener.requests,[])

    def test_optional_thinking_config_is_explicit_and_final_action_only(self):
        opener=ActionOpener()
        config=PolicyConfig.parse(dict(version=1,provider='deepseek',deepseek_thinking=True))
        policy=create_action_policy(config,opener=opener)
        self.assertEqual(policy.propose(self.request).action,'wait')
        self.assertEqual(opener.body['thinking'],{'type':'enabled'})
        self.assertEqual(opener.body['reasoning_effort'],'low')
        with self.assertRaises(ValueError):PolicyConfig.parse(dict(version=1,deepseek_thinking=1))

    def test_stale_reply_extra_fields_and_truncated_output_are_rejected(self):
        for reply,finish,code in [({'frame_id':41},'stop','invalid_action'),
                ({'request_id':'wrong'},'stop','invalid_action'),({'map_epoch':4},'stop','invalid_action'),
                ({'extra':1},'stop','invalid_action'),({},'length','output_token_limit')]:
            with self.subTest(reply=reply,finish=finish):
                opener=ActionOpener();opener.reply=reply;opener.finish=finish
                with self.assertRaises(PolicyAPIError) as caught:
                    DeepSeekActionPolicy(opener=opener).propose(self.request)
                self.assertEqual(caught.exception.code,code)

    def test_no_image_support_rejected_before_request(self):
        opener=ActionOpener()
        with self.assertRaises(PolicyAPIError) as caught:
            DeepSeekActionPolicy('deepseek-chat',opener=opener).propose(self.request)
        self.assertEqual(caught.exception.code,'unsupported_vision_model');self.assertFalse(opener.requests)

    def test_diagnostic_checks_image_and_json_without_game_inputs(self):
        opener=ActionOpener();report=DeepSeekActionPolicy(opener=opener).diagnose()
        self.assertEqual([r.get_method() for r in opener.requests],['GET','POST'])
        self.assertTrue(report['image_input_tested']);self.assertTrue(report['structured_action'])
        self.assertFalse(report['automatic_inputs'])

    def test_payment_error_retains_status_but_never_remote_secret(self):
        class Failing:
            def open(self,http,timeout):
                raise urllib.error.HTTPError(http.full_url,402,'deepseek-offline-secret',{},
                    io.BytesIO(b'{"error":{"message":"deepseek-offline-secret","code":"secret"}}'))
        with self.assertRaises(PolicyAPIError) as caught:
            DeepSeekActionPolicy(opener=Failing()).propose(self.request)
        details=caught.exception.data()
        self.assertEqual(details['code'],'insufficient_quota');self.assertEqual(details['http_status'],402)
        self.assertFalse(details['retryable']);self.assertNotIn('secret',json.dumps(details))

    def test_scene_planner_uses_deepseek_and_validates_grounding(self):
        fixtures=Path(__file__).parent/'fixtures'/'realtime'
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);request=make_request(cv2.imread(str(fixtures/'seed.png')),folder)
            scene=json.loads((fixtures/'scene.json').read_text(encoding='utf-8'))
            scene['request_id']=request['request_id']
            class Opener:
                def open(inner,http,timeout):
                    inner.http=http;return completion(scene)
            opener=Opener();planner=create_scene_planner('deepseek',opener=opener)
            self.assertIsInstance(planner,DeepSeekPlanner);planner.plan(folder)
            self.assertEqual(opener.http.get_header('Authorization'),'Bearer deepseek-offline-secret')
            self.assertEqual(json.loads((folder/'scene.json').read_text(encoding='utf-8'))['request_id'],request['request_id'])
            (folder/'scene.json').unlink();scene['request_id']='stale'
            with self.assertRaises(PolicyAPIError):planner.plan(folder)
            self.assertFalse((folder/'scene.json').exists())

    def test_scene_superseded_during_request_is_not_written(self):
        fixtures=Path(__file__).parent/'fixtures'/'realtime'
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);image=cv2.imread(str(fixtures/'seed.png'));request=make_request(image,folder)
            scene=json.loads((fixtures/'scene.json').read_text(encoding='utf-8'));scene['request_id']=request['request_id']
            class Opener:
                def open(inner,http,timeout):
                    make_request(image,folder);return completion(scene)
            with self.assertRaises(PlannerError):DeepSeekPlanner(opener=Opener()).plan(folder)
            self.assertFalse((folder/'scene.json').exists())


if __name__=='__main__':unittest.main()
