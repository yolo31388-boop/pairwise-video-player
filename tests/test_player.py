"""
游戏内视频播放与过场动画系统 - 验收测试
运行方式：python -m pytest tests/test_player.py -q
共 14 个测试用例
"""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.player import (
    VideoPlayer,
    FrameDecoder,
    FrameBuffer,
    Frame,
    AudioSync,
    SubtitleRenderer,
    SubtitleStyle,
    PlaybackController,
)


def make_frame(i):
    """构造一帧 RLE 编码数据：(i, 4) 展开为 4 字节像素。"""
    return bytes([i % 256, 4])


class TestVideoPlayer:
    def setup_method(self):
        self.system = VideoPlayer()

    def test_case_01(self):
        """帧解码实际生效：解码后产生真实画面数据，不再只有声音没有画面。"""
        frame = self.system.decode_frame(b"\x07\x03", index=0, timestamp=0.0)
        assert frame is not None
        assert frame.data == b"\x07\x07\x07"
        assert len(frame.data) > 0
        assert self.system.decoder.frames_decoded == 1

    def test_case_02(self):
        """帧解码有硬件加速：硬解耗时与 CPU 占用显著低于软解。"""
        encoded = bytes([1, 200])
        hw = FrameDecoder(hardware_acceleration=True)
        sw = FrameDecoder(hardware_acceleration=False)
        hw_frame = hw.decode_frame(encoded)
        sw_frame = sw.decode_frame(encoded)
        assert hw_frame.hw_accelerated is True
        assert hw_frame.decode_time < sw_frame.decode_time
        assert hw.cpu_usage < sw.cpu_usage
        assert hw.cpu_usage < 1.0

    def test_case_03(self):
        """帧解码有缓冲：解码后的帧进入解码缓冲，播放不卡顿。"""
        decoder = FrameDecoder(buffer_size=4)
        for i in range(6):
            decoder.decode_frame(make_frame(i), index=i, timestamp=i / 30.0)
        buffered = decoder.buffered
        assert len(buffered) == 4
        assert buffered[-1].index == 5
        assert decoder.frames_decoded == 6

    def test_case_04(self):
        """帧缓冲有双缓冲：front/back 交换呈现，画面不撕裂。"""
        buf = FrameBuffer(queue_size=4)
        f0 = Frame(index=0, timestamp=0.0, data=b"a")
        f1 = Frame(index=1, timestamp=1 / 30.0, data=b"b")
        buf.push(f0)
        buf.push(f1)
        assert buf.swap() is True
        assert buf.front is f0
        first_front = buf.front
        assert buf.swap() is True
        assert buf.front is f1
        assert buf.back is first_front

    def test_case_05(self):
        """帧缓冲有队列：解码与播放通过 FIFO 队列解耦同步。"""
        buf = FrameBuffer(queue_size=3)
        frames = [Frame(index=i, timestamp=i / 30.0, data=bytes([i]))
                  for i in range(5)]
        for f in frames[:3]:
            buf.push(f)
        assert len(buf) == 3
        buf.push(frames[3])
        buf.push(frames[4])
        assert len(buf) == 3
        order = []
        while buf.swap():
            order.append(buf.front.index)
        assert order == [2, 3, 4]

    def test_case_06(self):
        """音频同步有校正：音画偏移被收敛，不再越播越不同步。"""
        sync = AudioSync(threshold=0.04, correction_rate=0.5)
        video_ts, audio_ts = 10.0, 15.0
        corrected = sync.audio_sync(video_ts, audio_ts)
        assert sync.corrections == 1
        assert abs(audio_ts - corrected) < abs(audio_ts - video_ts)
        assert sync.audio_sync(10.0, 10.01) == 10.0

    def test_case_07(self):
        """音频同步有时钟：基于单调时钟，不随系统时间漂移。"""
        sync = AudioSync()
        sync.reset_clock(base=2.0)
        t1 = sync.clock()
        time.sleep(0.02)
        t2 = sync.clock()
        assert t1 >= 2.0
        assert t2 > t1
        assert t2 - t1 >= 0.01

    def test_case_08(self):
        """字幕渲染实际生效且有时间轴：台词时间内字幕准时出现。"""
        renderer = SubtitleRenderer()
        renderer.add_subtitle("第二句台词", start=5.0, end=8.0)
        renderer.add_subtitle("第一句台词", start=1.0, end=3.0)
        timeline = renderer.timeline
        assert [s.text for s in timeline] == ["第一句台词", "第二句台词"]
        assert renderer.subtitle_at(0.5) == []
        active = renderer.subtitle_at(2.0)
        assert len(active) == 1
        assert active[0].text == "第一句台词"
        assert renderer.subtitle_at(6.0)[0].text == "第二句台词"
        assert renderer.subtitle_at(9.0) == []

    def test_case_09(self):
        """字幕渲染有样式：字体、大小、颜色可配置并随渲染输出。"""
        style = SubtitleStyle(font="hei", size=32, color=(255, 0, 0), bold=True)
        renderer = SubtitleRenderer()
        renderer.add_subtitle("重点台词", start=0.0, end=2.0, style=style)
        rendered = renderer.render(1.0)
        assert len(rendered) == 1
        assert rendered[0]["font"] == "hei"
        assert rendered[0]["size"] == 32
        assert rendered[0]["color"] == (255, 0, 0)
        assert rendered[0]["bold"] is True

    def test_case_10(self):
        """播放控制有暂停：暂停后进度停止，恢复后继续。"""
        control = PlaybackController(duration=60.0)
        control.play()
        control.update(1.0)
        assert control.position == pytest.approx(1.0)
        control.pause()
        assert control.state == PlaybackController.PAUSED
        control.update(1.0)
        assert control.position == pytest.approx(1.0)
        control.resume()
        control.update(1.0)
        assert control.position == pytest.approx(2.0)

    def test_case_11(self):
        """播放控制有 seeking：可跳转到指定时间并钳制边界。"""
        control = PlaybackController(duration=60.0)
        assert control.seek(30.0) == 30.0
        assert control.position == 30.0
        assert control.seek(999.0) == 60.0
        assert control.seek(-5.0) == 0.0

    def test_case_12(self):
        """播放控制有速度调节：倍速播放按速度推进进度。"""
        control = PlaybackController(duration=60.0)
        control.set_speed(2.0)
        control.play()
        control.update(1.0)
        assert control.position == pytest.approx(2.0)
        control.set_speed(0.5)
        control.update(1.0)
        assert control.position == pytest.approx(2.5)
        with pytest.raises(ValueError):
            control.set_speed(0)

    def test_case_13(self):
        """播放控制有快进：快进倍率叠加，非法倍率报错。"""
        control = PlaybackController(duration=60.0)
        control.fast_forward(2.0)
        assert control.speed == pytest.approx(2.0)
        control.fast_forward(2.0)
        assert control.speed == pytest.approx(4.0)
        with pytest.raises(ValueError):
            control.fast_forward(0.5)

    def test_case_14(self):
        """完整过场播放：解码出画面、音画同步、字幕按时间轴出现。"""
        player = VideoPlayer()
        frames = [make_frame(i) for i in range(30)]
        player.load(
            frames,
            duration=1.0,
            fps=30.0,
            subtitles=[
                {"text": "开场白", "start": 0.0, "end": 0.5},
                {"text": "结束语", "start": 0.5, "end": 1.0},
            ],
        )
        player.play()
        dt = 1.0 / 30.0
        saw_subtitle = False
        for _ in range(31):
            player.update(dt)
            if player.subtitles.render(player.control.position):
                saw_subtitle = True
        assert player.decoder.frames_decoded > 0
        assert player.current_frame is not None
        assert player.current_frame.data
        assert saw_subtitle
        assert abs(player._state["sync_offset"]) < 0.04
        assert player.control.position == pytest.approx(1.0)
        assert player.control.state == PlaybackController.STOPPED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
