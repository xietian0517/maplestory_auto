"""Duration regressions use a simulated clock and capture; never send input."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from autofarm.realtime.perception import Frame
from autofarm.realtime.runtime import run,run_duration_seconds


class DurationTests(unittest.TestCase):
    def test_invalid_duration_is_rejected_before_capture(self):
        for seconds in (-1,.5,float('nan'),float('inf'),True,'600'):
            with self.subTest(seconds=seconds),self.assertRaises(ValueError):
                run_duration_seconds(seconds)

    def test_long_and_continuous_runs_reach_explicit_stop(self):
        for seconds in (1800,5400,0):
            with self.subTest(seconds=seconds),tempfile.TemporaryDirectory() as tmp:
                folder=Path(tmp);clock=[100.];image=np.zeros((32,32,3),np.uint8)
                class Capture:
                    backend='simulated';error=None
                    def __init__(self,*args):self.i=0
                    def __enter__(self):return self
                    def __exit__(self,*args):pass
                    def next(self,*args):
                        self.i+=1;clock[0]+=301
                        if self.i==5:
                            (folder/'STOP').write_text('explicit stop');return None
                        return Frame(self.i,clock[0],clock[0],image,True,(0,0,32,32))
                class Api:
                    def get_foreground(self):return 7
                with patch('autofarm.realtime.runtime.LatestCapture',Capture), \
                        patch('autofarm.realtime.runtime.time.perf_counter',side_effect=lambda:clock[0]):
                    run(Api(),7,folder,seconds=seconds)
                report=json.loads((folder/'report.json').read_text())
                self.assertEqual(report['phase'],'file_stop')
                self.assertEqual(report['frames'],4)
                self.assertGreater(report['elapsed_seconds'],600)
                self.assertEqual(report['active_input_frames'],0)


if __name__=='__main__':unittest.main()
