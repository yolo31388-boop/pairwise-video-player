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

from video.player import VideoPlayer, MediaClock, SYNC_THRESHOLD


class TestVideoPlayer:
    def setup_method(self):
        self.system = VideoPlayer()

    # 帧解码：decode_frame 实际产出解码后的帧（有画面数据）
    def test_case_01(self):
        frame = self.system.decode_frame(b"encoded-0", pts=0.0)
        assert frame is not None
        assert frame.data, "解码后应有画面数据"
        assert frame.data.startswith(b"RGB")
        assert frame.pts == 0.0
        assert self.system._decoded_count == 1

    # 帧解码：硬件加速生效，关闭时回退 CPU 解码
    def test_case_02(self):
        hw = VideoPlayer({"hw_accel": True})
        assert hw.decode_frame(b"x").hw_decoded is True
        cpu = VideoPlayer({"hw_accel": False})
        assert cpu.decode_frame(b"x").hw_decoded is False

    # 帧解码：解码后的帧进入解码缓冲，update 时泵入帧队列
    def test_case_03(self):
        for i in range(3):
            self.system.decode_frame(b"f%d" % i, pts=i / 30.0)
        assert self.system.decode_buffer_size == 3
        self.system.playback_control("play")
        self.system.update(1 / 30.0)
        assert self.system.decode_buffer_size == 0
        status = self.system.frame_buffer()
        assert status["queue_size"] + (1 if status["front"] else 0) == 3

    # 帧缓冲：双缓冲，swap 后 front/back 交换且计数增加
    def test_case_04(self):
        f0 = self.system.decode_frame(b"a", pts=0.0)
        f1 = self.system.decode_frame(b"b", pts=1 / 30.0)
        self.system.queue_frame(self.system._decode_buffer.popleft())
        self.system.queue_frame(self.system._decode_buffer.popleft())
        first = self.system.swap_buffers()
        assert first is f0
        second = self.system.swap_buffers()
        assert second is f1
        assert self.system.frame_buffer()["swaps"] == 2
        assert self.system.frame_buffer()["double_buffered"] is True

    # 帧缓冲：帧队列有容量上限，溢出时丢弃最旧的帧
    def test_case_05(self):
        player = VideoPlayer({"buffer_size": 4})
        for i in range(6):
            player.queue_frame(player.decode_frame(b"f%d" % i, pts=i / 30.0))
        status = player.frame_buffer()
        assert status["queue_size"] == 4
        assert status["dropped"] == 2
        assert player._frame_queue[0].index == 2  # 最旧的 0、1 已被丢弃

    # 音频同步：5 秒偏差被校正回阈值以内
    def test_case_06(self):
        self.system.playback_control("play")
        video_pts = self.system._clock.time()
        result = self.system.audio_sync(audio_pts=video_pts + 5.0)
        assert result["corrected"] is True
        assert result["corrections"] == 1
        assert abs(result["offset"]) <= SYNC_THRESHOLD

    # 音频同步：时钟基于单调时钟锚点，暂停时不漂移
    def test_case_07(self):
        clock = MediaClock()
        clock.start()
        clock.seek(10.0)
        clock.pause()
        t1 = clock.time()
        time.sleep(0.02)
        assert clock.time() == t1, "暂停时时钟不应漂移"
        assert self.system.audio_sync()["clock_source"] == "monotonic"

    # 字幕渲染：按时间轴在正确区间显示，台词说完字幕即消失
    def test_case_08(self):
        self.system.add_subtitle(1.0, 3.0, "第一句台词")
        self.system.add_subtitle(3.0, 5.0, "第二句台词")
        assert self.system.subtitle_renderer(at=0.5) == []
        shown = self.system.subtitle_renderer(at=2.0)
        assert [s["text"] for s in shown] == ["第一句台词"]
        shown = self.system.subtitle_renderer(at=4.0)
        assert [s["text"] for s in shown] == ["第二句台词"]
        assert self.system.subtitle_renderer(at=5.0) == []

    # 字幕渲染：样式可配置（字体/大小/颜色），全局样式对后续字幕生效
    def test_case_09(self):
        sub = self.system.add_subtitle(0.0, 1.0, "默认样式")
        assert sub.style["size"] == 24 and sub.style["color"] == "#FFFFFF"
        custom = self.system.add_subtitle(1.0, 2.0, "自定义",
                                          style={"size": 32, "color": "#FF0000"})
        assert custom.style["size"] == 32 and custom.style["color"] == "#FF0000"
        self.system.set_subtitle_style(font="serif", size=28)
        styled = self.system.add_subtitle(2.0, 3.0, "全局样式")
        assert styled.style["font"] == "serif" and styled.style["size"] == 28

    # 播放控制：暂停后位置冻结，恢复后继续前进
    def test_case_10(self):
        self.system.playback_control("play")
        time.sleep(0.02)
        self.system.playback_control("pause")
        pos = self.system.status()["position"]
        assert self.system.status()["state"] == "paused"
        time.sleep(0.02)
        assert self.system.status()["position"] == pos
        self.system.playback_control("resume")
        time.sleep(0.02)
        assert self.system.status()["position"] > pos

    # 播放控制：seek 跳转到指定时间，不超过时长，丢弃过时帧
    def test_case_11(self):
        player = VideoPlayer({"duration": 60.0})
        player.playback_control("play")
        for i in range(3):
            player.queue_frame(player.decode_frame(b"f%d" % i, pts=float(i)))
        state = player.playback_control("seek", position=30.0)
        assert abs(state["position"] - 30.0) < 0.1
        assert all(f.pts >= 30.0 for f in player._frame_queue)
        state = player.playback_control("seek", position=999.0)
        assert state["position"] <= 60.0 + 0.01  # 时钟公差

    # 播放控制：速度调节，2 倍速时位置约两倍前进，非法速度报错
    def test_case_12(self):
        self.system.playback_control("play")
        self.system.playback_control("set_speed", speed=2.0)
        assert self.system.status()["speed"] == 2.0
        start = self.system.status()["position"]
        time.sleep(0.05)
        elapsed = self.system.status()["position"] - start
        assert elapsed > 0.08, "2 倍速下 50ms 应前进约 0.1s"
        with pytest.raises(ValueError):
            self.system.playback_control("set_speed", speed=0)

    # 集成：完整播放流程，帧按序显示且音视频偏差在阈值内
    def test_case_13(self):
        player = VideoPlayer({"fps": 30, "duration": 1.0})
        for i in range(5):
            player.decode_frame(b"frame-%d" % i, pts=i / 30.0)
        player.playback_control("play")
        shown = []
        for _ in range(5):
            player.update(1 / 30.0)
            front = player.frame_buffer()["front"]
            if front is not None:
                shown.append(front.index)
        assert shown == sorted(shown), "帧应按解码顺序显示"
        assert len(shown) >= 1
        sync = player.audio_sync()
        assert abs(sync["offset"]) <= SYNC_THRESHOLD

    # reset：恢复到初始状态
    def test_case_14(self):
        self.system.decode_frame(b"x", pts=0.0)
        self.system.add_subtitle(0.0, 1.0, "台词")
        self.system.playback_control("play")
        self.system.reset()
        assert self.system.status()["state"] == "stopped"
        assert self.system.status()["position"] == 0.0
        assert self.system.status()["speed"] == 1.0
        assert self.system.decode_buffer_size == 0
        assert self.system.frame_buffer()["queue_size"] == 0
        assert self.system.subtitle_renderer(at=0.5) == []
        assert self.system._history == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
