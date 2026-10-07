"""Exact, session-calibrated HP reads and a latched return request."""
import re

from .demo_feedback import glyphs


class HealthGuard:
    def __init__(self,config,request_id,width,height):
        if config.get('request_id')!=request_id or config.get('size')!=[width,height]:
            raise ValueError('Health calibration belongs to another scene or window size')
        self.roi=config['roi']
        if (len(self.roi)!=4 or any(type(v)!=int for v in self.roi)
                or not 0<=self.roi[0]<self.roi[2]<=width
                or not 0<=self.roi[1]<self.roi[3]<=height):
            raise ValueError('Invalid health text region')
        self.size=(height,width);self.maximum=config['maximum']
        self.text_origin=config['text_origin']
        if type(self.text_origin)!=int or not 0<=self.text_origin<self.roi[2]-self.roi[0]:
            raise ValueError('Invalid health text origin')
        self.fraction=config.get('minimum_fraction',.45)
        self.return_enabled=config.get('return_enabled',True)
        if type(self.return_enabled)!=bool:raise ValueError('Invalid health return switch')
        self.unknown_timeout=config.get('unknown_timeout',2)
        if type(self.maximum)!=int or not 1<=self.maximum<=500000:
            raise ValueError('Invalid calibrated maximum HP')
        if not .1<=self.fraction<=.9 or not 1<=self.unknown_timeout<=10:
            raise ValueError('Invalid health return limits')
        self.alphabet={}
        for item in config['alphabet']:
            shape=tuple(item['shape']);bits=item['bits'];char=item['char']
            if (len(shape)!=2 or any(type(v)!=int or not 1<=v<=20 for v in shape)
                    or len(bytes.fromhex(bits))!=shape[0]*shape[1] or char not in '0123456789/' or len(char)!=1):
                raise ValueError('Invalid health glyph')
            key=(shape,bits)
            if key in self.alphabet and self.alphabet[key]!=char:raise ValueError('Ambiguous health glyph')
            self.alphabet[key]=char
        if set(self.alphabet.values())!=set('0123456789/'):raise ValueError('Incomplete health alphabet')
        self.hp=None;self.reason=None;self.unknown_since=None;self.low_since=None
        self.low_samples=0;self.previous_time=None;self.last_read_reason='not_observed'
        self.maximum_candidate=None;self.maximum_since=None;self.maximum_samples=0

    def read_numbers(self,image):
        """Decode both values only when every glyph and its position match exactly."""
        if image is None or image.shape[:2]!=self.size:return None,None,'window_size_unknown'
        x1,y1,x2,y2=self.roi
        pieces=glyphs(image[y1:y2,x1:x2])
        chars=[self.alphabet.get((shape,bits)) for _,shape,bits in pieces]
        if not chars or any(c is None for c in chars):return None,None,'glyph_unknown'
        text=''.join(chars)
        if not re.fullmatch(r'\d{1,6}/\d{1,6}',text):return None,None,'number_format_unknown'
        # This HUD font uses 6px digit cells (1 is inset 1px) and an 8px slash.
        # Enforce the calibrated layout so an obscured leading/interior digit
        # cannot silently turn 1165 into a plausible but incorrect 165.
        expected=self.text_origin
        for (x,_,_),char in zip(pieces,chars):
            if x!=expected+(1 if char=='1' else 0):return None,None,'glyph_layout_unknown'
            expected+=8 if char=='/' else 6
        current,maximum=map(int,text.split('/'))
        if not 1<=maximum<=500000 or not 0<=current<=maximum:return None,None,'health_range_unknown'
        return current,maximum,'exact_calibrated_glyphs'

    def read(self,image):
        current,maximum,reason=self.read_numbers(image)
        if current is None:return None,reason
        if maximum!=self.maximum:return None,'health_range_unknown'
        return current,'exact_calibrated_glyphs'

    def reset_maximum_confirmation(self):
        self.maximum_candidate=None;self.maximum_since=None;self.maximum_samples=0

    def observe(self,image,now):
        if self.reason:return self.reason
        # A long observation gap cannot count as continuous low-HP evidence.
        if self.previous_time is not None and not 0<now-self.previous_time<=.2:
            self.low_since=None;self.low_samples=0;self.unknown_since=None
            self.reset_maximum_confirmation()
        self.previous_time=now
        self.hp,self.last_read_reason=self.read(image)
        if self.hp is None and self.last_read_reason=='health_range_unknown':
            current,maximum,_=self.read_numbers(image)
            if current is not None and maximum!=self.maximum:
                if maximum!=self.maximum_candidate:
                    self.maximum_candidate=maximum;self.maximum_since=now;self.maximum_samples=0
                self.maximum_samples+=1
                self.last_read_reason='maximum_confirming'
                if self.maximum_samples>=3 and now-self.maximum_since>=.10:
                    self.maximum=maximum;self.hp=current;self.last_read_reason='exact_calibrated_glyphs'
                    self.reset_maximum_confirmation()
            else:self.reset_maximum_confirmation()
        else:self.reset_maximum_confirmation()
        if self.hp is None:
            self.low_since=None;self.low_samples=0
            if self.unknown_since is None:self.unknown_since=now
            if self.return_enabled and now-self.unknown_since>=self.unknown_timeout:self.reason='health_unreadable_return'
        else:
            self.unknown_since=None
            if self.hp==0 or self.return_enabled and self.hp<self.maximum*self.fraction:
                if self.low_since is None:self.low_since=now
                self.low_samples+=1
                if self.low_samples>=3 and now-self.low_since>=.10:
                    self.reason='health_depleted_stop' if self.hp==0 else 'low_health_return'
            else:self.low_since=None;self.low_samples=0
        return self.reason

    def status(self):
        return dict(hp=self.hp,maximum=self.maximum,minimum_fraction=self.fraction if self.return_enabled else None,
                    return_enabled=self.return_enabled,
                    maximum_candidate=self.maximum_candidate,maximum_samples=self.maximum_samples,
                    read_reason=self.last_read_reason,return_reason=self.reason)
