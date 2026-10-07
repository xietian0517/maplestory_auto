"""API failures are diagnosable without retaining secret-bearing error bodies."""
from concurrent.futures import Future
import io
import json
from pathlib import Path
import ssl
import tempfile
import urllib.error
import urllib.request
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from autofarm.realtime.policy import AsyncPolicy, OpenAIActionPolicy, PolicyConfig, PolicyRequest
from autofarm.realtime.policy_api import PolicyAPIError,request_json
from autofarm.realtime.pipeline import ArchivedPolicy


class FailingOpener:
    def __init__(self,status=429,code='insufficient_quota',param='model'):
        self.status,self.code,self.param=status,code,param

    def open(self,request,timeout):
        raw=json.dumps({'error':{'code':self.code,'type':'insufficient_quota',
            'message':'Echoed private-key-value with transport details','param':self.param}}).encode()
        raise urllib.error.HTTPError(request.full_url,self.status,'private-key-value',{},io.BytesIO(raw))


class APIErrorTests(unittest.TestCase):
    def test_http_classification_retains_only_safe_codes_and_parameters(self):
        for status,code in ((401,'invalid_api_key'),(429,'insufficient_quota'),
            (429,'project_spend_limit_exceeded'),(404,'model_not_found'),
            (403,'unsupported_country_region_territory')):
            with self.subTest(code=code),self.assertRaises(PolicyAPIError) as caught:
                request_json(urllib.request.Request('https://api.openai.com/v1/models'),12,
                             FailingOpener(status,code))
            data=caught.exception.data()
            self.assertEqual(data['code'],code);self.assertEqual(data['http_status'],status)
            self.assertFalse(data['retryable']);self.assertNotIn('private-key-value',json.dumps(data))
        error=PolicyAPIError('private-key-value',401,'private-key-value')
        self.assertEqual(error.code,'unknown_error');self.assertIsNone(error.param)

    def test_network_errors_are_classified_without_exception_text(self):
        for exc,code in ((TimeoutError('private-key-value'),'timeout'),
                         (urllib.error.URLError(ssl.SSLError('private-key-value')),'tls_error'),
                         (urllib.error.URLError('private-key-value'),'connection_error')):
            with self.subTest(code=code),self.assertRaises(PolicyAPIError) as caught:
                request_json(urllib.request.Request('https://api.openai.com/v1/models'),12,
                             SimpleNamespace(open=lambda *args,**kwargs:(_ for _ in ()).throw(exc)))
            self.assertEqual(caught.exception.code,code)
            self.assertNotIn('private-key-value',json.dumps(caught.exception.data()))

    def test_quota_failure_stops_automatic_retry_and_is_archived(self):
        with tempfile.TemporaryDirectory() as root:
            config=PolicyConfig(mode='active')
            future=Future()
            pool=SimpleNamespace(submit=lambda *args:future,shutdown=lambda **kwargs:None)
            worker=AsyncPolicy(SimpleNamespace(propose=lambda request:None),config,pool)
            o=SimpleNamespace(frame_id=1,map_epoch=0)
            worker.submit(o,1,'scene',{})
            policy=OpenAIActionPolicy(opener=FailingOpener())
            archived=ArchivedPolicy(policy,root)
            with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key-value'}):
                try:archived.propose(worker.request)
                except PolicyAPIError as error:future.set_exception(error)
            self.assertIsNone(worker.poll(1.1));self.assertEqual(worker.blocked_error,'insufficient_quota')
            self.assertFalse(worker.submit(o,100,'scene',{}))
            self.assertEqual(worker.events[-1]['error']['http_status'],429)
            paths=list(Path(root).glob('*/error.json'));self.assertEqual(len(paths),1)
            self.assertEqual(json.loads(paths[0].read_text(encoding='utf-8'))['code'],'insufficient_quota')
            for p in Path(root).rglob('*.json'):self.assertNotIn('private-key-value',p.read_text(encoding='utf-8'))
            worker.close()

    def test_incomplete_output_reports_token_budget(self):
        body=dict(status='incomplete',incomplete_details=dict(reason='max_output_tokens'),output=[])
        opener=SimpleNamespace(open=lambda *a,**kw:io.StringIO(json.dumps(body)))
        policy=OpenAIActionPolicy(opener=opener)
        req=PolicyRequest('r','scene',1,0,1,20,{})
        with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key-value'}),self.assertRaises(PolicyAPIError) as caught:
            policy.propose(req)
        self.assertEqual(caught.exception.code,'output_token_limit')

    def test_minimum_diagnostic_checks_actual_structured_action_and_billing(self):
        class Opener:
            def __init__(self):self.methods=[]
            def open(self,http,timeout):
                self.methods.append(http.get_method())
                if http.get_method()=='GET':return io.StringIO('{"id":"gpt-6-astra"}')
                self.body=json.loads(http.data)
                prompt=self.body['input'][0]['content'][0]['text']
                request=json.loads(prompt[prompt.index('\n{')+1:])
                reply=dict(request_id=request['request_id'],frame_id=request['frame_id'],
                    map_epoch=request['map_epoch'],action='wait',target='',world_x=None,
                    direction='',duration=.05,reason='offline diagnostic')
                return io.StringIO(json.dumps(dict(status='completed',output=[dict(type='message',
                    content=[dict(type='output_text',text=json.dumps(reply))])])) )
        opener=Opener();policy=OpenAIActionPolicy(opener=opener)
        with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key-value'}):result=policy.diagnose()
        self.assertEqual(opener.methods,['GET','POST']);self.assertTrue(result['structured_action'])
        self.assertFalse(result['automatic_inputs']);self.assertEqual(opener.body['reasoning']['effort'],'low')
        self.assertEqual(opener.body['max_output_tokens'],4096)
        class QuotaOpener(Opener):
            def open(self,http,timeout):
                if http.get_method()=='GET':return super().open(http,timeout)
                return FailingOpener().open(http,timeout)
        with patch.dict('os.environ',{'OPENAI_API_KEY':'private-key-value'}),self.assertRaises(PolicyAPIError) as caught:
            OpenAIActionPolicy(opener=QuotaOpener()).diagnose()
        self.assertEqual(caught.exception.code,'insufficient_quota')


if __name__=='__main__':unittest.main()
