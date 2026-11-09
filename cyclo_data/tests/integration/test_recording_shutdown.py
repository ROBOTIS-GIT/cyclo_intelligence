"""Opt-in real ROS tests. Run only in an isolated container with /workspace writable.

CYCLO_RECORDING_SHUTDOWN_INTEGRATION=1 ROS_DOMAIN_ID=233 pytest -s this_file
No hardware, commands, model runtime, network or GPU is needed.
"""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('CYCLO_RECORDING_SHUTDOWN_INTEGRATION') != '1',
    reason='requires explicitly isolated ROS recording environment',
)


@pytest.mark.parametrize('shutdown_signal', [signal.SIGINT, signal.SIGTERM])
@pytest.mark.parametrize('recorder_exits_first', [False, True])
def test_active_shutdown_stops_bag_and_retains_partial_without_overwrite(
        tmp_path, shutdown_signal, recorder_exits_first):
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import rclpy
    from ament_index_python.packages import get_package_prefix
    from cyclo_data.reader.bag_reader import BagReader
    from interfaces.srv import RecordingCommand
    from rosbag_recorder.srv import SendCommand
    from sensor_msgs.msg import CompressedImage, JointState

    assert os.environ['ROS_DOMAIN_ID'] == '233'
    processes = []
    done = threading.Event()
    node = None
    producer = None
    report = {'signal': shutdown_signal.name, 'recorder_exits_first': recorder_exits_first}
    session = f'shutdown_{tmp_path.parent.name}_{tmp_path.name}_{shutdown_signal.name}'
    folder = Path('/workspace/rosbag2') / f'Task_{session}_inference_MCAP'
    assert not folder.exists()

    def launch(name, argv, router=False):
        log = (tmp_path / f'{name}.log').open('w')
        env = dict(os.environ)
        if router:
            env['ZENOH_CONFIG_OVERRIDE'] = (
                'connect/endpoints=[];listen/endpoints=["tcp/127.0.0.1:7447"];'
                'scouting/multicast/enabled=false')
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        processes.append((name, process, log))
        return process

    def stop(process, sig=signal.SIGINT):
        if process.poll() is None:
            os.killpg(process.pid, sig)
            process.wait(timeout=15)
        return process.returncode

    def wait(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if predicate():
                return True
        return False

    def command(code, expected=True):
        req = RecordingCommand.Request()
        req.command = code
        req.robot_type = 'omy_f3m'
        req.task_info.task_type = 'inference'
        req.task_info.task_num = session
        req.task_info.task_instruction = ['Shutdown preservation test']
        req.topics = ['/joint_states']
        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=40)
        result = future.result()
        assert result is not None
        assert result.success == expected, result.message
        return result

    def hashes(path):
        return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in path.rglob('*') if p.is_file()}

    try:
        launch('router', [str(Path(get_package_prefix('rmw_zenoh_cpp')) /
                              'lib/rmw_zenoh_cpp/rmw_zenohd')], router=True)
        bag = launch('bag', [str(Path(get_package_prefix('rosbag_recorder')) /
                                'lib/rosbag_recorder/service_bag_recorder')])
        rclpy.init()
        node = rclpy.create_node('shutdown_integration_test')
        bag_client = node.create_client(SendCommand, '/rosbag_recorder/send_command')
        assert bag_client.wait_for_service(timeout_sec=20)
        camera = node.create_publisher(CompressedImage, '/camera/cam_wrist/color/image_rect_raw/compressed', 10)
        joints = node.create_publisher(JointState, '/joint_states', 10)
        data = launch('data', ['python3', '-m', 'cyclo_data.cyclo_data_node'])
        client = node.create_client(RecordingCommand, '/data/recording')
        assert client.wait_for_service(timeout_sec=30)
        command(RecordingCommand.Request.REFRESH_TOPICS)
        assert wait(lambda: camera.get_subscription_count() >= 1 and joints.get_subscription_count() >= 1)
        published = []

        def publish():
            while not done.is_set():
                stamp = node.get_clock().now().to_msg()
                image = CompressedImage()
                image.header.stamp = stamp
                image.format = 'jpeg'
                ok, jpeg = cv2.imencode('.jpg', np.full((120, 160, 3), len(published) % 255, np.uint8))
                assert ok
                image.data = jpeg.tobytes()
                camera.publish(image)
                joint = JointState()
                joint.header.stamp = stamp
                joint.name = ['joint1']
                joint.position = [float(len(published))]
                joints.publish(joint)
                published.append(time.time())
                done.wait(0.05)

        producer = threading.Thread(target=publish, daemon=True)
        producer.start()
        command(RecordingCommand.Request.START)
        start = len(published)
        assert wait(lambda: len(published) >= start + 30)
        command(RecordingCommand.Request.STOP)
        saved = folder / '0'
        assert wait(lambda: (saved / 'episode_info.json').exists() and
                    json.loads((saved / 'episode_info.json').read_text()).get('transcoding_status') == 'done')
        saved_hashes = hashes(saved)

        command(RecordingCommand.Request.START)
        start = len(published)
        assert wait(lambda: len(published) >= start + 30)
        if recorder_exits_first:
            stop(bag, shutdown_signal)
        started_shutdown = time.monotonic()
        report['data_exit'] = stop(data, shutdown_signal)
        exited_at = time.time()
        report['shutdown_seconds'] = time.monotonic() - started_shutdown
        # These idle-worker scenarios must finish before launch escalates SIGINT.
        assert report['shutdown_seconds'] < 5.0, report
        shutdown_log = (tmp_path / 'data.log').read_text()
        report['stop_failure_reported'] = 'Best-effort rosbag stop on shutdown failed' in shutdown_log
        start = len(published)
        assert wait(lambda: len(published) >= start + 25)
        report['native_recorder_still_alive'] = bag.poll() is None
        segment = folder / '1/segments/0'
        report['metadata_flushed_before_recorder_exit'] = (segment / 'metadata.yaml').exists()
        stop(bag)
        reader = BagReader(segment)
        assert reader.open()
        messages = list(reader.read_messages(['/joint_states']))
        reader.close()
        report['joint_messages'] = len(messages)
        report['messages_after_data_exit'] = sum(timestamp > exited_at for _, _, timestamp in messages)
        raw = next(segment.rglob('*.mjpeg.tmp'))
        report['raw_bytes'] = raw.stat().st_size
        report['timestamp_rows'] = pq.read_table(next(segment.rglob('*_timestamps.parquet'))).num_rows
        capture = cv2.VideoCapture(str(raw), cv2.CAP_FFMPEG)
        frames = 0
        while capture.read()[0]:
            frames += 1
        capture.release()
        report['decoded_partial_frames'] = frames
        report['saved_unchanged'] = hashes(saved) == saved_hashes
        report['partial_not_auto_saved'] = not list((folder / '1').rglob('episode_info.json'))
        report['partial_not_auto_discarded'] = raw.is_file()
        before_restart = hashes(folder)

        bag = launch('bag-restart', [str(Path(get_package_prefix('rosbag_recorder')) /
                                        'lib/rosbag_recorder/service_bag_recorder')])
        assert bag_client.wait_for_service(timeout_sec=20)
        data = launch('data-restart', ['python3', '-m', 'cyclo_data.cyclo_data_node'])
        assert wait(lambda: 'cyclo_data node ready.' in (tmp_path / 'data-restart.log').read_text(), 30)
        result = command(RecordingCommand.Request.START, expected=False)
        report['restart_error'] = result.message
        report['restart_no_overwrite'] = hashes(folder) == before_restart
        assert report['metadata_flushed_before_recorder_exit'], report
        assert report['messages_after_data_exit'] == 0, report
        assert report['data_exit'] == 0, report
        assert report['shutdown_seconds'] < 12, report
        assert report['stop_failure_reported'] == recorder_exits_first, report
        assert report['native_recorder_still_alive'] != recorder_exits_first, report
        assert report['raw_bytes'] > 0 and frames >= 20
        assert frames == report['timestamp_rows']
        assert report['saved_unchanged'] and report['restart_no_overwrite']
        assert report['partial_not_auto_saved'] and report['partial_not_auto_discarded']
    finally:
        done.set()
        if producer is not None:
            producer.join(timeout=5)
        for name, process, log in reversed(processes):
            try:
                stop(process)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                report.setdefault('forced_kills', []).append(name)
            log.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        (tmp_path / 'result.json').write_text(json.dumps(report, indent=2))
        tmp_path.chmod(0o755)
        tmp_path.parent.chmod(0o755)
        print(json.dumps(report), flush=True)
