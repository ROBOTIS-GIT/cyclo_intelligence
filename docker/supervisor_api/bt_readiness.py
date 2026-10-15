"""Read-only Task Engine ROS discovery, independent of rosapi and s6 process state."""
import logging
import threading

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
DEFAULT_SERVICES = ('/bt/nodes/catalog', '/bt/load_and_run', '/bt/set_running')
SUPPORTED_SERVICES = (*DEFAULT_SERVICES, '/bt/set_paused')


class ReadinessUnavailable(RuntimeError):
    pass


class BTReadiness:
    """One lazy private ROS context/node shared by concurrent readiness polls.

    service_is_ready reads the local DDS graph; no executor, service invocation,
    or wait_for_service call is needed. Discovery can be false briefly after
    process startup, and the caller must still await its real catalog response.
    """
    def __init__(self, services=DEFAULT_SERVICES):
        self.services = tuple(dict.fromkeys(services))
        if not self.services or any(name not in SUPPORTED_SERVICES for name in self.services):
            raise ValueError('Unsupported Task Engine readiness service')
        self._lock = threading.Lock()
        self._context = self._node = None
        self._clients = {}
        self._closed = False

    def _open(self):
        # No ROS dependency is imported until the first readiness request.
        from rclpy.context import Context
        from rclpy.node import Node
        from interfaces.srv import GetNodeCatalog, LoadAndRunTree
        from std_srvs.srv import SetBool

        self._context = Context()
        self._context.init(args=[])
        self._node = Node('cyclo_bt_readiness', context=self._context)
        types = {'/bt/nodes/catalog': GetNodeCatalog, '/bt/load_and_run': LoadAndRunTree,
                 '/bt/set_running': SetBool, '/bt/set_paused': SetBool}
        self._clients = {name: self._node.create_client(types[name], name) for name in self.services}

    def unavailable(self, reason):
        return {'ready': False, 'services': {name: False for name in self.services},
                'missing_services': list(self.services), 'reason': reason}

    def status(self):
        with self._lock:
            if self._closed:
                raise ReadinessUnavailable('Task Engine readiness probe is shutting down')
            try:
                if self._node is None: self._open()
                if not self._context.ok():
                    raise RuntimeError('ROS context is unavailable')
                services = {name: bool(client.service_is_ready()) for name, client in self._clients.items()}
            except Exception as exc:
                # Discard partial/disconnected clients; the next poll can retry.
                self._disconnect()
                raise ReadinessUnavailable(
                    f'Task Engine ROS readiness check failed ({type(exc).__name__}): {exc}') from exc
            missing = [name for name, ready in services.items() if not ready]
            return {'ready': not missing, 'services': services, 'missing_services': missing,
                    'reason': f'Waiting for Task Engine ROS services: {", ".join(missing)}' if missing else None}

    def _disconnect(self):
        node, context = self._node, self._context
        self._node = self._context = None
        self._clients = {}
        if node is not None:
            try: node.destroy_node()
            except Exception: logger.exception('Could not destroy Task Engine readiness node')
        if context is not None:
            try: context.try_shutdown()
            except Exception: logger.exception('Could not close Task Engine readiness context')

    def close(self):
        with self._lock:
            self._closed = True
            self._disconnect()


def create_router(probe):
    router = APIRouter(prefix='/bt', tags=['behavior-trees'])

    @router.get('/readiness')
    def readiness():
        try:
            return probe.status()
        except ReadinessUnavailable as exc:
            return JSONResponse(status_code=503, content=probe.unavailable(str(exc)))

    return router
