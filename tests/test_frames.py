"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path

import frames


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


def test_fps_mode_args_picks_modern_flag_on_ffmpeg_5_plus(monkeypatch):
    """ffmpeg >= 5 understands -fps_mode; 9.0 removed -vsync entirely."""
    for banner in ("ffmpeg version 9.0.1 Copyright (c)", "ffmpeg version n5.1.2"):
        frames._FPS_MODE_ARGS = None
        monkeypatch.setattr(
            frames.subprocess, "run",
            lambda *a, **k: __import__("types").SimpleNamespace(stdout=banner, returncode=0),
        )
        assert frames.fps_mode_args() == ["-fps_mode", "vfr"]
    frames._FPS_MODE_ARGS = None


def test_fps_mode_args_falls_back_to_vsync_on_ffmpeg_4(monkeypatch):
    """-fps_mode only landed in 5.0, so older builds still need -vsync."""
    frames._FPS_MODE_ARGS = None
    monkeypatch.setattr(
        frames.subprocess, "run",
        lambda *a, **k: __import__("types").SimpleNamespace(stdout="ffmpeg version 4.4.1", returncode=0),
    )
    assert frames.fps_mode_args() == ["-vsync", "vfr"]
    frames._FPS_MODE_ARGS = None


def test_fps_mode_args_defaults_to_modern_when_version_unreadable(monkeypatch):
    """An unparseable or missing ffmpeg gets the modern flag, not the removed one."""
    frames._FPS_MODE_ARGS = None
    monkeypatch.setattr(
        frames.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("no ffmpeg")),
    )
    assert frames.fps_mode_args() == ["-fps_mode", "vfr"]
    frames._FPS_MODE_ARGS = None
