"""Browser-compatible H.264 encoding using the bundled FFmpeg executable."""
from pathlib import Path
import subprocess

from imageio_ffmpeg import get_ffmpeg_exe


class H264Writer:
    def __init__(self,path,fps,size):
        self.path=Path(path);self.size=size;self.closed=False
        self.log=self.path.with_suffix('.encoder.log').open('wb')
        command=[get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-y',
                 '-f','rawvideo','-pixel_format','bgr24','-video_size',f'{size[0]}x{size[1]}',
                 '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
                 '-crf','18','-pix_fmt','yuv420p','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
                 '-movflags','+faststart',str(self.path)]
        try:
            self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                                          stderr=self.log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:
            self.log.close();raise

    def isOpened(self):return not self.closed and self.process.poll() is None

    def write(self,image):
        if image.shape[:2]!=(self.size[1],self.size[0]):raise ValueError('Video frame size changed')
        self.process.stdin.write(image.tobytes())

    def release(self):
        if self.closed:return
        self.closed=True
        try:
            try:self.process.stdin.close()
            except BrokenPipeError:pass
            try:code=self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait();raise OSError('视频编码超时，最后片段可能不完整')
            if code:raise OSError(f'视频编码失败（{code}），详情见 {self.path.with_suffix(".encoder.log").name}')
        finally:self.log.close()
