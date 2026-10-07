import unittest
from autofarm.realtime.model import Actor,Box
from autofarm.realtime.monster_tracking import associate,RecentTracks


class MonsterCameraMotionTests(unittest.TestCase):
    def test_stationary_monster_keeps_identity_during_large_camera_scroll(self):
        old=Actor(Box(200,300,250,350),.99,track_id=7)
        hit=Actor(old.box.moved(90,-80),.95)
        actors,n=associate([hit],[old],.1,(90,-80),8)
        self.assertEqual((actors[0].vx,actors[0].vy),(0,0))
        self.assertEqual(actors[0].track_id,7);self.assertEqual(n,8)

    def test_world_motion_does_not_change_when_camera_moves(self):
        old=Actor(Box(200,300,250,350),.99,track_id=7)
        still,_=associate([Actor(old.box.moved(10,2),.95)],[old],.1,(0,0),8)
        scroll,_=associate([Actor(old.box.moved(-40,32),.95)],[old],.1,(-50,30),8)
        self.assertEqual((still[0].vx,still[0].vy),(100,20))
        self.assertEqual((scroll[0].vx,scroll[0].vy),(100,20))

    def test_stale_detections_cannot_supply_velocity_or_identity(self):
        old=Actor(Box(200,300,250,350),.99,track_id=7)
        actors,n=associate([old],[old],.5,(0,0),8)
        self.assertEqual(actors[0].track_id,8)
        self.assertEqual((actors[0].vx,actors[0].vy),(0,0))
        self.assertEqual(n,9)

    def test_reviewed_round053_short_gap_reconnects_current_visible_sprite(self):
        memory=RecentTracks()
        old=Actor(Box(1079,338,1128,384),.85,track_id=10)
        memory.update([old],(193.9216,-259.2),20.0403,10)
        self.assertEqual(memory.update([],(191,-258),20.08,11),[])
        new=Actor(Box(1072,342,1121,388),.84,track_id=11)
        result=memory.update([new],(188.0965,-256.0074),20.1304,11)
        self.assertEqual(result[0].track_id,10)
        self.assertEqual(result[0].box,new.box)
        self.assertEqual(result[0].confidence,new.confidence)

    def test_occluded_cached_monster_is_never_returned_as_detection(self):
        memory=RecentTracks();old=Actor(Box(200,300,250,350),.99,track_id=7)
        memory.update([old],(0,0),1,7)
        self.assertEqual(memory.update([],(10,10),1.1,8),[])
        result=memory.update([Actor(old.box,.99,track_id=8)],(0,0),1.3,8)
        self.assertEqual(result[0].track_id,8)

    def test_two_nearby_old_or_new_monsters_are_ambiguous(self):
        old=Actor(Box(200,300,250,350),.99,track_id=7)
        other=Actor(old.box.moved(20,0),.99,track_id=8)
        memory=RecentTracks();memory.update([old,other],(0,0),1,7)
        new=Actor(old.box.moved(10,0),.99,track_id=9)
        self.assertEqual(memory.update([new],(0,0),1.1,9)[0].track_id,9)
        memory=RecentTracks();memory.update([old],(0,0),1,7)
        result=memory.update([new,Actor(old.box.moved(-10,0),.99,track_id=10)],(0,0),1.1,9)
        self.assertEqual([a.track_id for a in result],[9,10])

    def test_existing_visible_assignment_is_not_stolen_and_camera_reset_clears_cache(self):
        old=Actor(Box(200,300,250,350),.99,track_id=7)
        memory=RecentTracks();memory.update([old],(0,0),1,7)
        result=memory.update([old,Actor(old.box.moved(20,0),.99,track_id=8)],(0,0),1.1,8)
        self.assertEqual([a.track_id for a in result],[7,8])
        memory.clear()
        self.assertEqual(memory.update([Actor(old.box,.99,track_id=9)],(0,0),1.2,9)[0].track_id,9)


if __name__=='__main__':unittest.main()
