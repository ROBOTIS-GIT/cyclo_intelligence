from types import SimpleNamespace

from cyclo_data.services import edit_service as edit_service_module
from cyclo_data.services.edit_service import EditService
from orchestrator.internal.communication.communicator import Communicator


class _Logger:
    def __init__(self):
        self.info_messages = []
        self.error_messages = []

    def info(self, message):
        self.info_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)

    def warn(self, _message):
        pass


class _Node:
    def __init__(self):
        self.io_callback_group = object()
        self.logger = _Logger()
        self.services = []

    def create_service(self, service_type, name, callback, callback_group=None):
        handle = SimpleNamespace(
            service_type=service_type,
            name=name,
            callback=callback,
            callback_group=callback_group,
        )
        self.services.append(handle)
        return handle

    def get_logger(self):
        return self.logger


class _Publisher:
    def publish(self, _message):
        pass


def test_edit_service_owns_dataset_info_without_robot_lifecycle():
    node = _Node()

    service = EditService(node, _Publisher())

    assert service._info_server.name == '/dataset/get_info'
    assert [handle.name for handle in node.services] == [
        '/data/edit',
        '/dataset/get_info',
    ]


def test_dataset_info_callback_uses_shared_lock(monkeypatch):
    node = _Node()
    service = EditService(node, _Publisher())
    lock_modes = []

    class _Lock:
        def __init__(self, *, exclusive):
            lock_modes.append(exclusive)

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback):
            return None

    task_info = SimpleNamespace(
        robot_type='ffw_sg2_rev1',
        task_instruction='pick',
        episode_count=3,
        total_duration_s=6.0,
        fps=15,
        success_count=1,
        failure_count=1,
        unlabeled_count=1,
        success_episode_indices=[0],
        failure_episode_indices=[1],
        unlabeled_episode_indices=[2],
    )
    monkeypatch.setattr(edit_service_module, 'DatasetOperationLock', _Lock)
    monkeypatch.setattr(
        service._editor,
        'get_rosbag_task_info',
        lambda path: task_info,
    )
    response = SimpleNamespace()

    result = service._get_info_callback(
        SimpleNamespace(dataset_path='/workspace/rosbag2/Task_test_MCAP'),
        response,
    )

    assert result is response
    assert lock_modes == [False]
    assert response.success is True
    assert response.dataset_info.episode_count == 3
    assert list(response.dataset_info.success_episode_indices) == [0]
    assert list(response.dataset_info.failure_episode_indices) == [1]
    assert list(response.dataset_info.unlabeled_episode_indices) == [2]


def test_communicator_does_not_advertise_dataset_info_service():
    node = _Node()
    communicator = Communicator.__new__(Communicator)
    communicator.node = node
    communicator.get_image_topic_list_callback = lambda request, response: response
    communicator.browse_file_callback = lambda request, response: response
    communicator.list_trees_callback = lambda request, response: response

    communicator.init_services()

    assert '/dataset/get_info' not in [handle.name for handle in node.services]
