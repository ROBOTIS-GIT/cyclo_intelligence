"""Exercise LOAD option forwarding without ROS or hardware dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]


def method(relative, class_name, method_name, namespace):
    path = ROOT / 'orchestrator' / relative
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[method_name]


class CurrentRequest:
    vla_rail_enabled = False


class OldRequest:
    pass


class RailLoadPlumbingTests(unittest.TestCase):
    def client_call(self, request_type=CurrentRequest, **overrides):
        calls = []
        client = SimpleNamespace(
            CMD_LOAD=0, _service_prefix='/groot', _inference_command_client=object(),
            service_inference_command='/groot/inference_command', load_timeout_sec=90,
            load_availability_timeout_sec=30,
            _call_service=lambda *a, **kw: calls.append((a, kw)) or a[1],
        )
        namespace = dict(Optional=Optional, ServiceResponse=object,
                         InferenceCommand=SimpleNamespace(Request=request_type),
                         _uint16_or_zero=int, _positive_float_or_zero=float)
        call = method('internal/communication/container_service_client.py',
                      'ContainerServiceClient', 'inference_command', namespace)
        for key, value in overrides.items():
            setattr(client, key, value)
        return client, call, calls

    def test_service_forwards_enabled_flag_and_defaults_off(self):
        client, call, calls = self.client_call()
        self.assertTrue(call(client, 0, vla_rail_enabled=True).vla_rail_enabled)
        self.assertFalse(call(client, 0).vla_rail_enabled)
        self.assertEqual(len(calls), 2)

    def test_service_rejects_invalid_mode_before_any_request(self):
        for command, extra, prefix in (
            (1, {}, '/groot'),
            (0, {}, '/lerobot'),
        ):
            with self.subTest(command=command, extra=extra, prefix=prefix):
                client, call, calls = self.client_call(_service_prefix=prefix)
                with self.assertRaises(ValueError):
                    call(client, command, vla_rail_enabled=True, **extra)
                self.assertEqual(calls, [])

    def test_old_service_interface_allows_default_but_rejects_enabled(self):
        client, call, calls = self.client_call(OldRequest)
        call(client, 0)
        with self.assertRaisesRegex(RuntimeError, 'Rebuild interfaces'):
            call(client, 0, vla_rail_enabled=True)
        self.assertEqual(len(calls), 1)

    def supervisor_call(self, request_type=CurrentRequest):
        requests = []
        owner = SimpleNamespace(
            backend='groot', policy_path='/model', inference_mode='robot',
            task_instruction='pick', action_request_mode='async', acceleration_mode='pytorch',
            acceleration_engine_path='', control_hz=100, inference_hz=15,
            chunk_align_window_s=.3, vla_rail_enabled=True,
        )
        supervisor = SimpleNamespace(
            stopped=False, pending=None, verified=True, publish_to_robot=False,
            instruction='', node=SimpleNamespace(robot_type='ffw_sg2_rev1'),
            _instruction=str, client=SimpleNamespace(service_is_ready=lambda: True,
                call_async=lambda req: requests.append(req) or SimpleNamespace(done=lambda: True)),
        )
        namespace = dict(InferenceCommand=SimpleNamespace(Request=request_type),
                         LOAD=0, START=1, STOP=4, UNLOAD=5, CAPABILITIES=7)
        submit = method('bt/inference/supervisor.py', 'InferenceSupervisor', 'submit', namespace)
        return supervisor, owner, submit, requests

    def test_supervised_bt_load_forwards_rail_and_checks_old_interfaces(self):
        supervisor, owner, submit, requests = self.supervisor_call()
        submit(supervisor, 0, owner)
        self.assertTrue(requests[-1].vla_rail_enabled)
        supervisor, owner, submit, requests = self.supervisor_call(OldRequest)
        with self.assertRaisesRegex(RuntimeError, 'Rebuild interfaces'):
            submit(supervisor, 0, owner)
        self.assertEqual(requests, [])
        owner.vla_rail_enabled = False
        submit(supervisor, 0, owner)
        self.assertEqual(len(requests), 1)

    def test_signature_differs_when_rail_toggled(self):
        path = ROOT / 'orchestrator/internal/communication/inference_mode.py'
        namespace = {}
        exec(compile(path.read_text(), str(path), 'exec'), namespace)
        signature = namespace['inference_runtime_signature']
        args = ('/model', 'pytorch', '', 'async', 100, 15, .3)
        self.assertNotEqual(signature(*args), signature(*args, vla_rail_enabled=True))

    def test_sdk_service_schema_matches_rosidl(self):
        source = ROOT.parent / 'cyclo_brain/sdk/robot_client/robot_client/messages/__init__.py'
        tree = ast.parse(source.read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'INFERENCE_COMMAND_REQUEST_DEF'
                                  for t in n.targets))
        sdk_definition = ast.literal_eval(assignment.value)
        service = (ROOT.parent / 'interfaces/srv/InferenceCommand.srv').read_text().split('---')[0]
        fields = lambda text: [line.split('#', 1)[0].strip() for line in text.splitlines()
                               if line.split('#', 1)[0].strip() and '=' not in line]
        self.assertEqual(fields(sdk_definition), fields(service))


if __name__ == '__main__':
    unittest.main()
