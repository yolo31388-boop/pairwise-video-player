"""
游戏内视频播放与过场动画系统 - 核心模块

包含五个子系统：
- FrameDecoder        帧解码（硬件加速 + 解码缓冲）
- FrameBuffer         帧缓冲（双缓冲 + 帧队列）
- AudioSync           音频同步（单调时钟 + 偏移校正）
- SubtitleRenderer    字幕渲染（时间轴 + 样式）
- PlaybackController  播放控制（暂停/快进/seeking/速度调节）
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import time


@dataclass
class Config:
    hardware_acceleration: bool = True
    decode_buffer_size: int = 8
    frame_queue_size: int = 4
    sync_threshold: float = 0.04
    sync_correction_rate: float = 0.5
    default_speed: float = 1.0
    target_fps: float = 30.0


@dataclass
class Frame:
    index: int
    timestamp: float
    data: bytes
    width: int = 1920
    height: int = 1080
    hw_accelerated: bool = False
    decode_time: float = 0.0


@dataclass
class SubtitleStyle:
    font: str = "sans-serif"
    size: int = 24
    color: Tuple[int, int, int] = (255, 255, 255)
    bold: bool = False
    outline: bool = True


@dataclass
class Subtitle:
    text: str
    start: float
    end: float
    style: SubtitleStyle = field(default_factory=SubtitleStyle)


class FrameDecoder:
    """帧解码器：实际解码帧数据，支持硬件加速与解码缓冲。"""

    PER_BYTE_COST = 0.0005
    HW_SPEEDUP = 0.1

    def __init__(self, hardware_acceleration: bool = True,
                 buffer_size: int = 8, target_fps: float = 30.0):
        self.hardware_acceleration = hardware_acceleration
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self._buffer: Deque[Frame] = deque(maxlen=buffer_size)
        self.frames_decoded = 0
        self.total_decode_time = 0.0

    def decode_frame(self, encoded: bytes, index: int = 0,
                     timestamp: float = 0.0) -> Frame:
        data = self._decode(encoded)
        cost = len(data) * self.PER_BYTE_COST
        if self.hardware_acceleration:
            cost *= self.HW_SPEEDUP
        frame = Frame(index=index, timestamp=timestamp, data=data,
                      hw_accelerated=self.hardware_acceleration,
                      decode_time=cost)
        self._buffer.append(frame)
        self.frames_decoded += 1
        self.total_decode_time += cost
        return frame

    @staticmethod
    def _decode(encoded: bytes) -> bytes:
        """RLE 解码：(值, 重复次数) 字节对展开为像素数据。"""
        if not encoded:
            return b""
        if len(encoded) % 2 == 0:
            out = bytearray()
            for i in range(0, len(encoded), 2):
                out.extend(bytes([encoded[i]]) * encoded[i + 1])
            return bytes(out)
        return bytes(encoded)

    @property
    def buffered(self) -> List[Frame]:
        return list(self._buffer)

    @property
    def average_decode_time(self) -> float:
        if not self.frames_decoded:
            return 0.0
        return self.total_decode_time / self.frames_decoded

    @property
    def cpu_usage(self) -> float:
        """按目标帧率估算的 CPU 占用（0.0 ~ 1.0）。"""
        return min(1.0, self.average_decode_time * self.target_fps)


class FrameBuffer:
    """帧缓冲：双缓冲（front/back）避免撕裂，帧队列解耦解码与播放。"""

    def __init__(self, queue_size: int = 4):
        self.queue_size = queue_size
        self._queue: Deque[Frame] = deque(maxlen=queue_size)
        self._front: Optional[Frame] = None
        self._back: Optional[Frame] = None

    def push(self, frame: Frame):
        self._queue.append(frame)

    def swap(self) -> bool:
        """从队列取出下一帧到 back，再交换 front/back 完成呈现。"""
        if not self._queue:
            return False
        self._back = self._queue.popleft()
        self._front, self._back = self._back, self._front
        return True

    @property
    def front(self) -> Optional[Frame]:
        return self._front

    @property
    def back(self) -> Optional[Frame]:
        return self._back

    @property
    def queued(self) -> int:
        return len(self._queue)

    def __len__(self) -> int:
        return len(self._queue)


class AudioSync:
    """音频同步：基于单调时钟的播放时钟 + 偏移校正。"""

    def __init__(self, threshold: float = 0.04, correction_rate: float = 0.5):
        self.threshold = threshold
        self.correction_rate = correction_rate
        self._clock = time.monotonic
        self._origin: Optional[float] = None
        self._base = 0.0
        self.corrections = 0
        self.last_offset = 0.0

    def clock(self) -> float:
        now = self._clock()
        if self._origin is None:
            self._origin = now
            return self._base
        return self._base + (now - self._origin)

    def reset_clock(self, base: float = 0.0):
        self._base = base
        self._origin = None

    def audio_sync(self, video_ts: float, audio_ts: float) -> float:
        """音画偏移超过阈值时，将视频时间戳向音频时钟校正。"""
        offset = audio_ts - video_ts
        self.last_offset = offset
        if abs(offset) <= self.threshold:
            return video_ts
        self.corrections += 1
        return video_ts + offset * self.correction_rate


class SubtitleRenderer:
    """字幕渲染：按时间轴检索当前字幕，支持自定义样式。"""

    def __init__(self, default_style: Optional[SubtitleStyle] = None):
        self.default_style = default_style or SubtitleStyle()
        self._subtitles: List[Subtitle] = []

    def add_subtitle(self, text: str, start: float, end: float,
                     style: Optional[SubtitleStyle] = None) -> Subtitle:
        if end <= start:
            raise ValueError("字幕结束时间必须大于开始时间")
        sub = Subtitle(text=text, start=start, end=end,
                       style=style or self.default_style)
        self._subtitles.append(sub)
        self._subtitles.sort(key=lambda s: s.start)
        return sub

    def subtitle_at(self, t: float) -> List[Subtitle]:
        return [s for s in self._subtitles if s.start <= t < s.end]

    def render(self, t: float) -> List[Dict]:
        return [
            {
                "text": s.text,
                "font": s.style.font,
                "size": s.style.size,
                "color": s.style.color,
                "bold": s.style.bold,
                "outline": s.style.outline,
            }
            for s in self.subtitle_at(t)
        ]

    @property
    def timeline(self) -> List[Subtitle]:
        return list(self._subtitles)


class PlaybackController:
    """播放控制：播放/暂停/停止、seeking、快进与速度调节。"""

    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"

    def __init__(self, duration: float = 0.0, speed: float = 1.0):
        self.duration = duration
        self.speed = speed
        self.state = self.STOPPED
        self.position = 0.0

    def play(self) -> str:
        self.state = self.PLAYING
        return self.state

    def pause(self) -> str:
        if self.state == self.PLAYING:
            self.state = self.PAUSED
        return self.state

    def resume(self) -> str:
        if self.state == self.PAUSED:
            self.state = self.PLAYING
        return self.state

    def stop(self) -> str:
        self.state = self.STOPPED
        self.position = 0.0
        return self.state

    def seek(self, t: float) -> float:
        """跳转到指定时间，自动钳制在 [0, duration] 内。"""
        self.position = max(0.0, min(t, self.duration))
        return self.position

    def set_speed(self, speed: float) -> float:
        if speed <= 0:
            raise ValueError("播放速度必须为正数")
        self.speed = speed
        return self.speed

    def fast_forward(self, factor: float = 2.0) -> float:
        if factor <= 1.0:
            raise ValueError("快进倍率必须大于 1")
        self.speed *= factor
        return self.speed

    def update(self, dt: float) -> float:
        if self.state != self.PLAYING:
            return self.position
        self.position += dt * self.speed
        if self.position >= self.duration:
            self.position = self.duration
            self.state = self.STOPPED
        return self.position


class VideoPlayer:
    """游戏内视频播放器：整合解码、缓冲、音画同步、字幕与播放控制。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        cfg = Config()
        for key, value in self.config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        self._cfg = cfg
        self.decoder = FrameDecoder(cfg.hardware_acceleration,
                                    cfg.decode_buffer_size,
                                    cfg.target_fps)
        self.buffer = FrameBuffer(cfg.frame_queue_size)
        self.sync = AudioSync(cfg.sync_threshold, cfg.sync_correction_rate)
        self.subtitles = SubtitleRenderer()
        self.control = PlaybackController(speed=cfg.default_speed)
        self._frames: List[bytes] = []
        self._fps = cfg.target_fps
        self._next_frame = 0
        self._audio_position = 0.0
        self._state: Dict = {}
        self._history: List = []

    def load(self, frames: List[bytes], duration: float,
             fps: float = 30.0, subtitles: Optional[List[Dict]] = None):
        """加载视频帧流、时长与字幕轨。"""
        self._frames = list(frames)
        self._fps = fps
        self.control.duration = duration
        self._next_frame = 0
        self._audio_position = 0.0
        for sub in subtitles or []:
            self.subtitles.add_subtitle(**sub)

    def decode_frame(self, encoded: bytes, index: int = 0,
                     timestamp: float = 0.0) -> Frame:
        return self.decoder.decode_frame(encoded, index, timestamp)

    def frame_buffer(self) -> FrameBuffer:
        return self.buffer

    def audio_sync(self, video_ts: float, audio_ts: float) -> float:
        return self.sync.audio_sync(video_ts, audio_ts)

    def subtitle_renderer(self) -> SubtitleRenderer:
        return self.subtitles

    def playback_control(self) -> PlaybackController:
        return self.control

    def play(self) -> str:
        self.sync.reset_clock(self.control.position)
        return self.control.play()

    def pause(self) -> str:
        return self.control.pause()

    def seek(self, t: float) -> float:
        return self.control.seek(t)

    def set_speed(self, speed: float) -> float:
        return self.control.set_speed(speed)

    def fast_forward(self, factor: float = 2.0) -> float:
        return self.control.fast_forward(factor)

    @property
    def current_frame(self) -> Optional[Frame]:
        return self.buffer.front

    def _frame_timestamp(self, index: int) -> float:
        return index / self._fps if self._fps else 0.0

    def update(self, dt: float):
        position = self.control.update(dt)
        if self.control.state != PlaybackController.PLAYING:
            return position
        self._audio_position += dt * self.control.speed
        lookahead = self.buffer.queue_size / self._fps if self._fps else 0.0
        while (self._next_frame < len(self._frames)
               and self._frame_timestamp(self._next_frame) <= position + lookahead):
            frame = self.decoder.decode_frame(
                self._frames[self._next_frame],
                index=self._next_frame,
                timestamp=self._frame_timestamp(self._next_frame))
            self.buffer.push(frame)
            self._next_frame += 1
        self.buffer.swap()
        self.sync.audio_sync(position, self._audio_position)
        self._state["sync_offset"] = self.sync.last_offset
        return position

    def reset(self):
        cfg = self._cfg
        self.decoder = FrameDecoder(cfg.hardware_acceleration,
                                    cfg.decode_buffer_size,
                                    cfg.target_fps)
        self.buffer = FrameBuffer(cfg.frame_queue_size)
        self.sync = AudioSync(cfg.sync_threshold, cfg.sync_correction_rate)
        self.subtitles = SubtitleRenderer()
        self.control = PlaybackController(speed=cfg.default_speed)
        self._frames = []
        self._next_frame = 0
        self._audio_position = 0.0
        self._state = {}
        self._history = []
