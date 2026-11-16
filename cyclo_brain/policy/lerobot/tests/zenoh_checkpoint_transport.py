"""Isolated smoke-test transport. Never use with host networking or a robot domain."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


def configure(server):
    os.environ["ROS_DOMAIN_ID"] = "176"
    os.environ["ZENOH_CONFIG_OVERRIDE"] = (
        'transport/shared_memory/enabled=false;scouting/multicast/enabled=false;'
        + ('mode="peer";connect/endpoints=[];listen/endpoints=["tcp/127.0.0.1:7447"]'
           if server else 'mode="client";connect/endpoints=["tcp/127.0.0.1:7447"];listen/endpoints=[]')
    )


class ZenohCheckpointTransport:
    def __init__(self, worker):
        configure(True)
        from zenoh_ros2_sdk import ROS2ServiceServer
        from engine_process.protocol import (ENGINE_COMMAND_REQUEST_DEF, ENGINE_COMMAND_RESPONSE_DEF,
                                             response_to_message_kwargs)

        def callback(request):
            return self.service.response_msg_class(**response_to_message_kwargs(worker.handle(request)))

        self.service = ROS2ServiceServer(
            service_name="/cyclo_test/checkpoint", srv_type="interfaces/srv/EngineCommand",
            request_definition=ENGINE_COMMAND_REQUEST_DEF, response_definition=ENGINE_COMMAND_RESPONSE_DEF,
            callback=callback,
        )
        self._responses = queue.Queue()
        try:
            self._process = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             text=True, bufsize=1)
        except BaseException:
            self.service.close()
            raise

        def read():
            try:
                for line in self._process.stdout:
                    if line.startswith("CYCLO_RESPONSE "):
                        self._responses.put(json.loads(line.removeprefix("CYCLO_RESPONSE ")))
                    else:
                        print(line, end="", flush=True)
            finally:
                self._responses.put({"error": "Zenoh test client exited"})

        self._reader = threading.Thread(target=read)
        self._reader.start()

    def call(self, request, timeout_s):
        from engine_process.protocol import EngineCommandResponse
        self._process.stdin.write(json.dumps({"request": asdict(request), "timeout_s": timeout_s}) + "\n")
        self._process.stdin.flush()
        try:
            payload = self._responses.get(timeout=timeout_s + 5.)
        except queue.Empty as exc:
            raise TimeoutError("Zenoh test client did not return") from exc
        if "error" in payload:
            raise RuntimeError(payload["error"])
        return EngineCommandResponse(**payload["response"])

    def close(self):
        self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._reader.join(timeout=5)
        assert not self._reader.is_alive()
        self._process.stdout.close()
        self.service.close()


def client_main():
    configure(False)
    from zenoh_ros2_sdk import ROS2ServiceClient
    from engine_process.protocol import (ENGINE_COMMAND_REQUEST_DEF, ENGINE_COMMAND_RESPONSE_DEF,
                                         EngineCommandRequest, request_to_message_kwargs, response_from_message)

    client = ROS2ServiceClient(
        service_name="/cyclo_test/checkpoint", srv_type="interfaces/srv/EngineCommand",
        request_definition=ENGINE_COMMAND_REQUEST_DEF, response_definition=ENGINE_COMMAND_RESPONSE_DEF,
    )
    try:
        for line in sys.stdin:
            try:
                call = json.loads(line)
                client.timeout = call["timeout_s"]
                request = EngineCommandRequest(**call["request"])
                response = client.call(**request_to_message_kwargs(request))
                if response is None:
                    raise TimeoutError("Zenoh service deadline exceeded")
                result = {"response": asdict(response_from_message(response))}
            except Exception as exc:
                result = {"error": repr(exc)}
            print("CYCLO_RESPONSE " + json.dumps(result), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    client_main()
