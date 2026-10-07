import unittest
import numpy as np
from autofarm.realtime.model import Actor,Box,MotionProfile,Observation,Platform,Rope
from autofarm.realtime.parking import SafeParking,health_signature,validate_parking
from pathlib import Path
import cv2
from types import SimpleNamespace


class ParkingTests(unittest.TestCase):
    def setUp(self):
        self.floor=Platform('refuge',100,300,200)
        self.image=cv2.imread(str(Path(__file__).parent/'fixtures/realtime/live_seed.png'))
        self.roi=(490,730,627,744)
        self.parker=SafeParking(MotionProfile(200,110,200,True),['refuge'],self.roi,quiet_seconds=.5)

    def observation(self,t,x=200,monsters=None):
        return Observation(1,t,Actor(Box(x-15,150,x+15,200),.99),
            monsters=monsters or [],platforms=[self.floor])

    def test_arrival_requires_continuous_quiet_health_observations(self):
        for t in (1,1.1,1.2,1.3,1.4):
            d=self.parker.decide(self.observation(t),t,self.image)
            self.assertFalse(d.keys);self.assertFalse(self.parker.done)
        self.assertEqual(self.parker.decide(self.observation(1.51),1.51,self.image).reason,'parking_confirmed')
        self.assertTrue(self.parker.result()['confirmed'])

    def test_health_loss_or_camera_gap_resets_quiet_confirmation(self):
        for cause in ('damage','gap'):
            with self.subTest(cause=cause):
                self.setUp()
                for t in (1,1.1,1.2,1.3):self.parker.decide(self.observation(t),t,self.image)
                image=self.image.copy()
                if cause=='damage':image[732:741,500:505]=0;t=1.4
                else:t=2
                self.parker.decide(self.observation(t),t,image)
                self.assertFalse(self.parker.done)
                self.assertNotEqual(self.parker.quiet_since,1)

    def test_nearby_monster_disqualifies_refuge(self):
        enemy=Actor(Box(225,160,255,200),.99)
        d=self.parker.decide(self.observation(1,monsters=[enemy]),1,self.image)
        self.assertEqual(d.reason,'parking_no_reachable_refuge')
        self.assertFalse(self.parker.done);self.assertFalse(d.keys)

    def test_monster_on_adjacent_lower_platform_disqualifies_refuge(self):
        # Old +/-100px, 65px check missed ranged attacks arriving from below.
        enemy=Actor(Box(485,220,515,270),.99)
        o=self.observation(1);o.navigation_targets=[enemy]
        d=self.parker.decide(o,1,self.image)
        self.assertEqual(d.reason,'parking_no_reachable_refuge')
        self.assertFalse(self.parker.done)

    def test_unknown_health_and_stale_player_never_confirm_safety(self):
        for t in (1,1.1,1.2,1.3,1.4,1.5,1.6):
            self.parker.decide(self.observation(t),t,np.zeros_like(self.image))
        self.assertFalse(self.parker.done)
        d=self.parker.decide(self.observation(2),2.2,self.image)
        self.assertFalse(d.keys);self.assertFalse(self.parker.done)

    def test_moves_to_refuge_center_before_waiting(self):
        d=self.parker.decide(self.observation(1,x=130),1,self.image)
        self.assertEqual(d.keys,{'right'});self.assertFalse(self.parker.done)

    def test_position_jitter_is_allowed_but_accumulated_drift_is_not(self):
        for t,x in [(1,200),(1.1,201),(1.2,199),(1.3,201),(1.4,200),(1.51,201)]:
            o=self.observation(t,x=x);o.player=Actor(o.player.box,.99,vx=60,vy=35)
            self.parker.decide(o,t,self.image)
        self.assertTrue(self.parker.done)
        self.setUp()
        for t,x in [(1,200),(1.1,202),(1.2,204),(1.3,206),(1.4,207),(1.51,208)]:
            self.parker.decide(self.observation(t,x=x),t,self.image)
        self.assertFalse(self.parker.done)

    def test_close_centering_releases_before_delayed_motion_estimate(self):
        self.assertEqual(self.parker.decide(self.observation(1,x=182),1,self.image).keys,{'right'})
        for t in (1.033,1.1,1.2,1.3):
            self.assertFalse(self.parker.decide(self.observation(t,x=182),t,self.image).keys)
        self.assertFalse(self.parker.done)

    def test_parking_uses_mapped_route_without_attacking(self):
        o=self.observation(1,x=200)
        o.player=Actor(Box(185,0,215,50),.99)
        o.platforms=[Platform('start',100,300,50),self.floor]
        o.monsters=[Actor(Box(215,0,245,50),.99)]
        d=self.parker.decide(o,1,self.image)
        self.assertNotIn('shift',d.keys)
        self.assertEqual(self.parker.target,'refuge')
        self.assertIn(d.reason,('drop','approach_launch'))

    def test_refuges_are_explicitly_bound_to_scene_request(self):
        scene=SimpleNamespace(request_id='current',platforms=[self.floor],width=640,height=400)
        data=dict(request_id='old',safe_platforms=['refuge'],health_text_roi=[0,0,100,20])
        with self.assertRaises(ValueError):validate_parking(data,scene)
        data['request_id']='current'
        self.assertEqual(validate_parking(data,scene)[0],['refuge'])
        data['safe_platforms']=['invented']
        with self.assertRaises(ValueError):validate_parking(data,scene)

    def test_temporary_route_failure_expires_without_a_controller_step(self):
        o=self.observation(1);o.player=Actor(Box(185,0,215,50),.99)
        o.platforms=[Platform('start',100,300,50),self.floor]
        self.parker.epoch=o.map_epoch;self.parker.base.last_epoch=o.map_epoch
        self.parker.base.fail_edge(('start','refuge'),1)
        self.assertEqual(self.parker.decide(o,1,self.image).reason,'parking_no_reachable_refuge')
        o.captured_at=2.1
        d=self.parker.decide(o,2.1,self.image)
        self.assertNotEqual(d.reason,'parking_no_reachable_refuge')
        self.assertEqual(self.parker.target,'refuge')
        self.assertNotIn('shift',d.keys)

    def test_repeated_failed_jump_stays_blocked_during_parking(self):
        o=self.observation(1);o.player=Actor(Box(185,0,215,50),.99)
        o.platforms=[Platform('start',100,300,50),self.floor]
        self.parker.epoch=o.map_epoch;self.parker.base.last_epoch=o.map_epoch
        for _ in range(2):self.parker.base.fail_edge(('start','refuge'),1,'jump')
        o.captured_at=10
        self.assertEqual(self.parker.decide(o,10,self.image).reason,'parking_no_reachable_refuge')

    def test_fresh_parking_on_rope_preserves_bounded_attachment_probe(self):
        o=Observation(1,1,Actor(Box(185,210,215,260),.99),
            platforms=[self.floor,Platform('ground',100,300,400)],ropes=[Rope(200,200,380)])
        self.assertEqual(self.parker.decide(o,1,self.image).reason,'rope_observe_attachment')
        o.captured_at=1.17
        d=self.parker.decide(o,1.17,self.image)
        self.assertEqual(d.reason,'rope_probe_attachment');self.assertEqual(d.keys,{'up'})
        climber=self.parker.base.rope_climber
        o.captured_at=1.20
        self.parker.decide(o,1.20,self.image)
        self.assertIs(self.parker.base.rope_climber,climber)
        self.assertFalse(self.parker.done)

    def test_unknown_airborne_position_without_rope_never_sends_blind_keys(self):
        o=Observation(1,1,Actor(Box(185,210,215,260),.99),platforms=[self.floor])
        d=self.parker.decide(o,1,self.image)
        self.assertFalse(d.keys);self.assertFalse(self.parker.done)

if __name__=='__main__':unittest.main()
