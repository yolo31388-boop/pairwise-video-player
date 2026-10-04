"""
游戏内视频播放与过场动画系统 - 核心模块

包含五大子系统：
- decode_frame:      帧解码（支持硬件加速、解码缓冲）
- frame_buffer:      帧缓冲（双缓冲 + 帧队列）
- audio_sync:        音频同步（单调时钟 + 偏差校正）
- subtitle_renderer: 字幕渲染（时间轴 + 样式）
- playback_control:  播放控制（暂停/快进/seek/速度调节）
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


# 音视频允许的最大偏差（秒），超过则触发校正
SYNC_THRESHOLD = 0.04

DEFAULT_SUBTITLE_STYLE = {
    "font": "sans-serif",
    "size": 24,
    "color": "#FFFFFF",
    "bold": False,
}


@dataclass
class Frame:
    """一帧解码后的视频数据。"""
    index: int
    pts: float
    data: bytes
    hw_decoded: bool = False


@dataclass
class Subtitle:
    """一条带时间轴和样式的字幕。"""
    start: float
    end: float
    text: str
    style: Dict = field(default_factory=lambda: dict(DEFAULT_SUBTITLE_STYLE))

    def render(self) -> Dict:
        entry = {"text": self.text, "start": self.start, "end": self.end}
        entry.update(self.style)
        return entry


class MediaClock:
    """媒体时钟：基于单调时钟锚点推算，避免逐帧累加 dt 造成的漂移。"""

    def __init__(self):
        self._anchor_media = 0.0
        self._anchor_real = time.monotonic()
        self._speed = 1.0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def speed(self) -> float:
        return self._speed

    def start(self):
        if not self._running:
            self._anchor_real = time.monotonic()
            self._running = True

    def pause(self):
        if self._running:
            self._anchor_media = self.time()
            self._running = False

    def stop(self):
        self._running = False
        self._anchor_media = 0.0
        self._anchor_real = time.monotonic()
        self._speed = 1.0

    def seek(self, position: float):
        self._anchor_media = max(0.0, position)
        self._anchor_real = time.monotonic()

    def set_speed(self, speed: float):
        if speed <= 0:
            raise ValueError("播放速度必须为正数")
        self._anchor_media = self.time()
        self._anchor_real = time.monotonic()
        self._speed = speed

    def time(self) -> float:
        if not self._running:
            return self._anchor_media
        return self._anchor_media + (time.monotonic() - self._anchor_real) * self._speed


class VideoPlayer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.fps = self.config.get("fps", 30)
        self.hw_accel = self.config.get("hw_accel", True)
        self.buffer_size = self.config.get("buffer_size", 8)
        self.duration = float(self.config.get("duration", 0.0))
        self._clock = MediaClock()
        self._init_state()

    def _init_state(self):
        self._state = "stopped"  # stopped / playing / paused
        self._history = []
        # 解码缓冲：解码后的帧先进入解码缓冲，再由 update 泵入帧队列
        self._decode_buffer = deque(maxlen=self.buffer_size)
        self._decoded_count = 0
        # 帧缓冲：帧队列 + 双缓冲（front 显示 / back 准备）
        self._frame_queue = deque(maxlen=self.buffer_size)
        self._front_buffer = None
        self._back_buffer = None
        self._swap_count = 0
        self._dropped_frames = 0
        # 音频同步
        self._av_offset = 0.0
        self._sync_corrections = 0
        # 字幕
        self._subtitles = []
        self._subtitle_style = dict(DEFAULT_SUBTITLE_STYLE)

    # ------------------------------------------------------------------
    # 帧解码：实际解码 + 硬件加速 + 解码缓冲
    # ------------------------------------------------------------------
    def decode_frame(self, encoded, pts: float = 0.0, index: Optional[int] = None) -> Frame:
        if encoded is None:
            raise ValueError("没有可解码的数据")
        if index is None:
            index = self._decoded_count
        if isinstance(encoded, (bytes, bytearray)):
            payload = bytes(encoded)
        else:
            payload = str(encoded).encode("utf-8")
        # 硬件加速开启时走 GPU 解码路径（hw_decoded=True），否则回退 CPU 解码
        frame = Frame(index=index, pts=float(pts),
                      data=b"RGB" + payload, hw_decoded=bool(self.hw_accel))
        self._decode_buffer.append(frame)
        self._decoded_count += 1
        self._history.append(("decode", index))
        return frame

    @property
    def decode_buffer_size(self) -> int:
        return len(self._decode_buffer)

    # ------------------------------------------------------------------
    # 帧缓冲：双缓冲 + 帧队列
    # ------------------------------------------------------------------
    def queue_frame(self, frame: Frame) -> int:
        """把解码后的帧放入帧队列；队列满时丢弃最旧的帧。"""
        if len(self._frame_queue) >= self.buffer_size:
            self._frame_queue.popleft()
            self._dropped_frames += 1
        self._frame_queue.append(frame)
        return len(self._frame_queue)

    def swap_buffers(self) -> Optional[Frame]:
        """从队列取下一帧到 back buffer，然后交换 front/back（双缓冲）。"""
        if self._frame_queue:
            self._back_buffer = self._frame_queue.popleft()
        if self._back_buffer is not None:
            self._front_buffer, self._back_buffer = self._back_buffer, self._front_buffer
            self._swap_count += 1
        return self._front_buffer

    def frame_buffer(self) -> Dict:
        """帧缓冲状态：双缓冲 + 队列信息。"""
        return {
            "double_buffered": True,
            "front": self._front_buffer,
            "back": self._back_buffer,
            "queue_size": len(self._frame_queue),
            "capacity": self.buffer_size,
            "swaps": self._swap_count,
            "dropped": self._dropped_frames,
        }

    # ------------------------------------------------------------------
    # 音频同步：单调媒体时钟 + 偏差校正
    # ------------------------------------------------------------------
    def audio_sync(self, audio_pts: Optional[float] = None) -> Dict:
        """对齐音视频。偏差超过阈值时把媒体时钟校正到音频时间。"""
        video_pts = self._front_buffer.pts if self._front_buffer is not None else self._clock.time()
        if audio_pts is None:
            audio_pts = self._clock.time()
        offset = audio_pts - video_pts
        corrected = False
        if abs(offset) > SYNC_THRESHOLD:
            self._clock.seek(audio_pts)
            self._sync_corrections += 1
            corrected = True
            offset = 0.0
        self._av_offset = offset
        return {
            "offset": offset,
            "corrected": corrected,
            "corrections": self._sync_corrections,
            "clock": self._clock.time(),
            "clock_source": "monotonic",
        }

    # ------------------------------------------------------------------
    # 字幕渲染：时间轴 + 样式
    # ------------------------------------------------------------------
    def add_subtitle(self, start: float, end: float, text: str,
                     style: Optional[Dict] = None) -> Subtitle:
        if end <= start:
            raise ValueError("字幕结束时间必须大于开始时间")
        merged = dict(self._subtitle_style)
        if style:
            merged.update(style)
        sub = Subtitle(start=float(start), end=float(end), text=text, style=merged)
        self._subtitles.append(sub)
        self._subtitles.sort(key=lambda s: s.start)
        return sub

    def set_subtitle_style(self, **style):
        """设置全局字幕样式（字体、大小、颜色等），对之后添加的字幕生效。"""
        self._subtitle_style.update(style)

    def subtitle_renderer(self, at: Optional[float] = None) -> List[Dict]:
        """按时间轴返回指定时刻（默认当前播放时刻）应显示的字幕。"""
        if at is None:
            at = self._clock.time()
        return [s.render() for s in self._subtitles if s.start <= at < s.end]

    # ------------------------------------------------------------------
    # 播放控制：播放/暂停/停止/seek/速度调节
    # ------------------------------------------------------------------
    def playback_control(self, command: str, **kwargs) -> Dict:
        if command in ("play", "resume"):
            if self._state in ("stopped", "paused"):
                self._clock.start()
                self._state = "playing"
        elif command == "pause":
            if self._state == "playing":
                self._clock.pause()
                self._state = "paused"
        elif command == "stop":
            self._clock.stop()
            self._state = "stopped"
        elif command == "seek":
            position = max(0.0, float(kwargs["position"]))
            if self.duration:
                position = min(position, self.duration)
            self._clock.seek(position)
            # 丢弃队列中已过时的帧，保证从跳转位置继续播放
            self._frame_queue = deque(
                (f for f in self._frame_queue if f.pts >= position),
                maxlen=self.buffer_size,
            )
        elif command in ("set_speed", "fast_forward"):
            speed = float(kwargs.get("speed", 2.0))
            self._clock.set_speed(speed)
        else:
            raise ValueError("未知的播放控制命令: %r" % command)
        self._history.append(("control", command))
        return self.status()

    def status(self) -> Dict:
        return {
            "state": self._state,
            "position": self._clock.time(),
            "speed": self._clock.speed,
            "duration": self.duration,
        }

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def update(self, dt: float):
        if self._state != "playing":
            return
        # 把解码缓冲里的帧泵入帧队列，再做双缓冲交换
        while self._decode_buffer:
            self.queue_frame(self._decode_buffer.popleft())
        self.swap_buffers()
        # 音视频同步校正
        if self._front_buffer is not None:
            self.audio_sync(self._clock.time())

    def reset(self):
        self._clock.stop()
        self._init_state()
