"""Regression checks using the bundled HAR assets and synthetic recordings."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import argparse
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from core.config import ROOT, load_config
from core.mainthread import HARThread, PeopleMonitoringThread, VideoStreamingThread
from core.types import Joint, SJoints, SkeletonFrame, KinectFrame, KinectFrameReader
from processes.har import HAR
from processes.paint import paint, VideoMode
from pykinect2_emu.pykinect2_emu import KinectEmulator
from tools.nn_model import Recognizer
from tools.skeleton_reader import SkeletonReader

CONFIG = str(ROOT / 'config.example.json')


@pytest.fixture(scope='module')
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope='module')
def recognizer():
    return Recognizer()


@pytest.fixture
def har(monkeypatch, recognizer):
    monkeypatch.setattr(HAR, '_HAR__set_nn_model', lambda self: setattr(self, 'recognizer', recognizer))
    instance = HAR(6)
    instance.database_connect()
    instance.start_monitoring()
    return instance


def pose():
    raw = SkeletonReader(ROOT / 'data/nn-model/162database.txt', 'TST', '3D').read_skeletons(True)[0]
    skeleton = SkeletonFrame([Joint(*row) for row in raw], [SJoints(40 + i * 2, 30 + i * 2) for i in range(17)])
    skeleton.tracking_id = 1
    return skeleton


def make_recording(tmp_path, count=40):
    (tmp_path / 'Body').mkdir()
    (tmp_path / 'Color').mkdir()
    skeleton = pose()
    rows = '\n'.join(' '.join(map(str, [*j.tolist(), *s])) for j, s in
                     zip(skeleton.player_coordinates, skeleton.player_screen_coordinates))
    (tmp_path / 'Body/FileSkeleton.txt').write_text(''.join(f'Frame {i}\n{rows}\n' for i in range(count)))
    writer = cv2.VideoWriter(str(tmp_path / 'Color/person_stream.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25, (320, 240))
    assert writer.isOpened(), 'Test environment needs an MP4 encoder.'
    for _ in range(count):
        writer.write(np.full((240, 320, 3), 100, np.uint8))
    writer.release()
    return tmp_path


def test_config_joint_order():
    config = load_config(CONFIG)
    assert len(config['SKELETON_USABLE_JOINTS_NUMBER']) == 17
    assert max(max(edge) for edge in config['SKELETON_LINES']) == 16


def test_saved_model_inference(recognizer):
    result = recognizer.predict_actions(np.zeros((1, 32, 32), np.float32))
    assert result.shape == (1, 9)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-5)
    with pytest.raises(ValueError):
        recognizer.predict_actions(np.full((1, 32, 32), np.nan))


def test_activity_maps_for_six_bodies(har):
    for _ in range(32):
        har.append_frame([pose() for _ in range(6)])
    result = har.get_person_class()
    assert len(result) == 6
    assert all(label in har.classes for label in result)


def test_missing_and_reassigned_bodies_clear_windows(har):
    for _ in range(32):
        har.append_frame([pose(), None, None, None, None, None])
    har.append_frame([None] * 6)
    assert not har.skeletons['skeleton_000']
    har.append_frame([pose(), None, None, None, None, None])
    changed = pose()
    changed.tracking_id = 2
    har.append_frame([changed, None, None, None, None, None])
    assert len(har.skeletons['skeleton_000']) == 1
    assert har.get_person_class() == ['no_action'] * 6


@pytest.mark.parametrize('worker_type', [HARThread, PeopleMonitoringThread, VideoStreamingThread])
def test_idle_workers_stop_without_a_frame(qt_app, worker_type):
    worker = worker_type(CONFIG) if worker_type is VideoStreamingThread else worker_type(CONFIG, 6)
    worker.start()
    worker.requestInterruption()
    assert worker.wait(2000), f'{worker_type.__name__} did not stop'


def test_render_without_background_file():
    frame = KinectFrame(6)
    frame.set_color_frame(np.zeros((240, 320, 3), np.uint8), (320, 240))
    frame.skeleton_frames[0] = pose()
    for mode in [VideoMode.SKELETAL, VideoMode.REAL]:
        result = paint(frame, CONFIG, mode, ['NoID'] * 6, ['no_action'] * 6)
        assert result.shape == (1080, 1920, 3)


def test_replay_loops_and_releases_video(tmp_path):
    directory = make_recording(tmp_path, count=3)
    with KinectEmulator(str(directory)) as source:
        reader = KinectFrameReader(source)
        for _ in range(8):
            assert reader.get_color_frame().shape == (240, 320, 3)
            assert reader.set_bodies()
            assert reader.is_body_tracked(0)
            assert not reader.is_body_tracked(1)
    assert not source._capture.isOpened()


def test_missing_replay_video_fails_instead_of_spinning(tmp_path):
    directory = make_recording(tmp_path, count=1)
    (directory / 'Color/person_stream.mp4').unlink()
    with pytest.raises(FileNotFoundError):
        KinectEmulator(str(directory))


def test_nonfinite_screen_joints_are_skipped():
    source = SimpleNamespace(body_joints_to_color_space=lambda _: [SimpleNamespace(x=float('inf'), y=3)])
    reader = KinectFrameReader(source)
    reader.bodies = [SimpleNamespace(joints=[])]
    assert reader.get_color_space_joints(0, [0]) == [[0, 0]]


def test_gui_replay_reaches_inference_and_closes(qt_app, tmp_path, monkeypatch, recognizer):
    from service import App
    directory = make_recording(tmp_path)
    monkeypatch.setattr(HAR, '_HAR__set_nn_model', lambda self: setattr(self, 'recognizer', recognizer))
    widget = App(argparse.Namespace(config=CONFIG, source='recording', recording=str(directory), identify=False))
    results, errors = [], []
    widget.human_action_recognition_thread.send_action_info.connect(results.append)
    # Do not open a modal dialog when an integration test encounters a failure.
    for worker in widget.workers():
        worker.failed.disconnect(widget.on_worker_error)
        worker.failed.connect(errors.append)
    try:
        widget.start()
        deadline = time.monotonic() + 20
        while not results and not errors and time.monotonic() < deadline:
            qt_app.processEvents()
            time.sleep(0.01)
        assert not errors
        assert results, 'No HAR result arrived from the replay pipeline'
        assert len(results[0]) == 6
        assert results[0][1:] == ['no_action'] * 5
    finally:
        widget.shutdown()
        widget.close()
    assert all(not worker.isRunning() for worker in widget.workers())
