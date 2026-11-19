"""Bounded in-process recording receipts; queries never execute work."""

from collections import OrderedDict
from copy import deepcopy
import threading
from uuid import UUID


class CommandReceipts:
    def __init__(self, capacity=256):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._archives = {}

    @staticmethod
    def validate(request_id):
        if str(UUID(request_id)) != request_id:
            raise ValueError('request_id must be a canonical UUID')

    def lookup(self, request_id, response, request=None):
        self.validate(request_id)
        with self._lock:
            response.request_id = request_id
            response.archive_state = self._archives.get(request_id, '')
            entry = self._entries.get(request_id)
            if entry is None:
                response.success = False
                response.command_state = 'unknown'
                response.message = 'Recording receipt unknown; do not replay the command'
            elif request is not None and entry[0] != request:
                response.success = False
                response.command_state = 'rejected'
                response.message = 'Recording request_id reused with different arguments'
            elif entry[1] is None:
                response.success = False
                response.command_state = 'running'
                response.message = 'Recording command is still running'
            else:
                response.success, response.message = entry[1]
                response.command_state = 'completed'
            return response

    def reserve(self, request):
        self.validate(request.request_id)
        with self._lock:
            if request.request_id in self._entries:
                return False
            # Pending entries are never evicted. Clients query, never re-submit,
            # an uncertain command after a receipt expires or the server restarts.
            while len(self._entries) >= self._capacity:
                completed = next((key for key, (_, result) in self._entries.items()
                                  if result is not None and self._archives.get(key) != 'running'), None)
                if completed is None:
                    raise RuntimeError('Recording receipt capacity exhausted')
                del self._entries[completed]
                self._archives.pop(completed, None)
            self._entries[request.request_id] = (deepcopy(request), None)
            return True

    def complete(self, request_id, response):
        with self._lock:
            request, _ = self._entries[request_id]
            self._entries[request_id] = (request, (bool(response.success), str(response.message)))

    def archive(self, request_id, state):
        with self._lock:
            if request_id in self._entries:
                self._archives[request_id] = state
