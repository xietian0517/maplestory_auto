"""Offline recording tests. Fake API cannot send game input."""
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autofarm.realtime.demonstration import KeyTrace,KeySampler,VideoSegments,record_demo,write_review
from autofarm.realtime.demo_analysis import input_intervals,analyze
from autofarm.realtime.perception import Frame


class KeyTraceTests(unittest.TestCase):
    def test_focus_boundary_does_not_invent_key_release(self):
        trace=KeyTrace();events=trace.observe(0,True,{'left'})
        self.assertEqual([e['event'] for e in events],['focus_gained','key_sync'])
        events=trace.observe(.01,True,{'left','alt','a'})
        self.assertEqual(events,[dict(t=.01,event='key_down',key='alt')])
        events=trace.observe(.02,False,{'left','alt','shift'})
        self.assertNotIn('key_up',[e['event'] for e in events])
        self.assertEqual([e['key'] for e in events if e['event']=='key_cancel'],['alt','left'])
        self.assertFalse(trace.snapshot(.021)['keys'])

    def test_edges_keep_short_taps_and_snapshot_uses_past_sample(self):
        trace=KeyTrace();trace.observe(0,True,set())
        self.assertEqual(trace.observe(.004,True,{'alt'})[0]['event'],'key_down')
        self.assertEqual(trace.observe(.012,True,set())[0]['event'],'key_up')
        self.assertEqual(trace.snapshot(.009)['keys'],['alt'])
        self.assertIsNone(trace.snapshot(-1)['keys'])

    def test_polling_gap_is_reported_without_fabricating_events(self):
        trace=KeyTrace();trace.observe(0,True,{'shift'})
        events=trace.observe(.2,True,{'shift'})
        self.assertEqual(events,[dict(t=.2,event='sampling_gap',seconds=.2)])
        self.assertEqual(trace.summary()['maximum_key_poll_gap_ms'],200)

    def test_simultaneous_keys_generate_combo_and_no_success_claim(self):
        events=[dict(t=0,event='focus_gained'),dict(t=.1,event='key_down',key='alt'),
                dict(t=.1,event='key_down',key='up'),dict(t=.2,event='key_up',key='alt'),
                dict(t=.4,event='recording_end')]
        rows=input_intervals(events)
        combo=next(r for r in rows if 'rope_catch_attempt_input' in r['tags'])
        self.assertAlmostEqual(combo['seconds'],.1)
        self.assertEqual(combo['result'],'unknown')
        self.assertTrue(rows[-1]['boundary_censored'])

    def test_sampler_reads_only_foreground_whitelist_and_f11_stops(self):
        class API:
            foreground=0
            reads=[]
            def get_foreground(self):return self.foreground
            def is_iconic(self,hwnd):return False
            def async_pressed(self,key):self.reads.append(key);return key=='f11'
        api=API();stop=threading.Event()
        with tempfile.TemporaryDirectory() as d:
            with KeySampler(api,1,d,time.perf_counter(),stop) as sampler:
                time.sleep(.025);self.assertEqual(api.reads,[])
                api.foreground=1;self.assertTrue(stop.wait(1))
            self.assertIsNone(sampler.error)
            rows=[json.loads(x) for x in (Path(d)/'inputs.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['event'],'recording_end')
            self.assertEqual(sampler.stop_reason,'F11')


class RecordingTests(unittest.TestCase):
    def test_segment_indexes_and_actual_timestamps_survive_resize_and_gap(self):
        with tempfile.TemporaryDirectory() as d:
            with VideoSegments(d,60,segment_seconds=1) as writer:
                for i,(t,size) in enumerate(((0,(64,48)),(.016,(64,48)),(1.2,(64,48)),(1.22,(80,64)))):
                    w,h=size;image=np.full((h,w,3),i*50,np.uint8)
                    self.assertTrue(writer.offer(image,dict(frame_id=i,capture_started=t,capture_finished=t+.001,keys=[])))
            self.assertIsNone(writer.error);self.assertEqual(writer.written,4)
            rows=[json.loads(x) for x in (Path(d)/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['video_frame'] for r in rows],[0,1,0,0])
            self.assertEqual(rows[2]['capture_started'],1.2)
            for segment in writer.segments:
                video=cv2.VideoCapture(str(Path(d)/segment['file']))
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),segment['frames'])
                self.assertTrue(video.read()[0]);video.release()
            write_review(d);self.assertTrue((Path(d)/'review.html').exists())

    def test_backpressure_is_counted(self):
        with tempfile.TemporaryDirectory() as d:
            writer=VideoSegments(d,60,queue_size=1);image=np.zeros((48,64,3),np.uint8)
            self.assertTrue(writer.offer(image,{}));self.assertFalse(writer.offer(image,{}))
            self.assertEqual(writer.dropped,1)

    def test_encoder_failure_is_visible(self):
        class BrokenWriter:
            def isOpened(self):return False
            def release(self):pass
        with tempfile.TemporaryDirectory() as d,patch('autofarm.realtime.demonstration.H264Writer',return_value=BrokenWriter()):
            with VideoSegments(d,60) as writer:
                writer.offer(np.zeros((48,64,3),np.uint8),dict(capture_started=0))
            self.assertIn('MP4',writer.error);self.assertEqual(writer.written,0)

    def test_hud_endpoints_and_attack_lossless_samples_are_timestamped(self):
        with tempfile.TemporaryDirectory() as d:
            images=[]
            with VideoSegments(d,60,queue_size=20) as writer:
                for i,t in enumerate((0,.1,.2,.5,.7,1.0,1.4,2.0)):
                    image=np.full((120,160,3),i*20,np.uint8);images.append(image)
                    writer.offer(image,dict(frame_id=i,capture_started=t,capture_finished=t+.001,
                                            keys=['shift'] if t<=.5 else []))
            self.assertIsNone(writer.error)
            hud=[json.loads(s) for s in (Path(d)/'hud.jsonl').read_text().splitlines()]
            combat=[json.loads(s) for s in (Path(d)/'combat.jsonl').read_text().splitlines()]
            self.assertEqual(hud[0]['kind'],'initial');self.assertEqual(hud[-1]['kind'],'final')
            self.assertEqual(hud[-1]['t'],2.0)
            np.testing.assert_array_equal(cv2.imread(str(Path(d)/hud[-1]['image'])),images[-1][-110:])
            self.assertEqual([r['t'] for r in combat],[0,.5,1.0])
            self.assertTrue(all(not r['damage_confirmed'] for r in combat))
            np.testing.assert_array_equal(cv2.imread(str(Path(d)/combat[1]['image'])),images[3])

    def test_complete_recorder_with_fake_game_and_readback_analysis(self):
        stop=threading.Event()
        class API:
            def get_foreground(self):return 7
            def is_iconic(self,hwnd):return False
            def async_pressed(self,key):return False
            def send_key(self,*args):raise AssertionError('Recorder must never send input')
        class Capture:
            backend='offline_test';error=None
            def __init__(self,*args,**kwargs):self.i=0;assert kwargs['foreground_only']
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def next(self,*args,**kwargs):
                time.sleep(.01);self.i+=1;t=time.perf_counter()
                if self.i>8:stop.set();return None
                return Frame(self.i,t,t,np.full((48,64,3),self.i*20,np.uint8),True,(0,0,64,48))
        with tempfile.TemporaryDirectory() as d:
            report=record_demo(API(),7,d,seconds=1,stop=stop,capture_factory=Capture,lock_factory=nullcontext)
            self.assertEqual(report['phase'],'stopped');self.assertFalse(report['automatic_input'])
            self.assertEqual(report['frames_written'],8);self.assertNotIn('analysis_error',report)
            results=analyze(d,extract=False)
            self.assertTrue(all(v['valid'] for v in results['video_checks']))
            self.assertIsNone(results['confirmed_kills'])
            self.assertTrue((Path(d)/'review_data.js').exists())
            with self.assertRaises(ValueError):record_demo(API(),7,d,seconds=1,lock_factory=nullcontext)

    def test_active_automation_refuses_recording(self):
        class Locked:
            def __enter__(self):raise RuntimeError('Another AI controller is already running')
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory() as d:
            report=record_demo(None,7,d,seconds=1,lock_factory=Locked)
            self.assertEqual(report['phase'],'error');self.assertIn('旧挂机程序',report['error'])
            self.assertFalse((Path(d)/'inputs.jsonl').exists())


if __name__=='__main__':unittest.main()
