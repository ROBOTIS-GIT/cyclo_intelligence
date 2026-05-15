#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Dongyun Kim

import threading

from rclpy.node import Node


class TimerManager:

    def __init__(self, node: Node):
        self._node = node
        self._timer = {}
        self._timer_frequency = {}
        self._timer_callback = {}
        # orchestrator runs under MultiThreadedExecutor so start/stop/
        # stop_all/set_timer can be invoked concurrently from different
        # service callbacks. The lock serialises mutations of the three
        # state dicts (TOCTOU on start, dict-mutation-during-iteration
        # on stop_all).
        self._lock = threading.Lock()

    def start(self, timer_name):
        with self._lock:
            if self._timer.get(timer_name) is None:
                self._timer[timer_name] = self._node.create_timer(
                    1.0/self._timer_frequency[timer_name],
                    self._timer_callback[timer_name])

    def stop(self, timer_name):
        # Tolerate unknown timer names. The orchestrator's FINISH/SKIP/
        # RERECORD paths call stop(operation_mode) defensively, but in
        # the post-§5.5 layout 'inference' has no orchestrator-side
        # timer (the policy container owns its own 100 Hz loop). Without
        # this guard, stop('inference') used to KeyError and abort the
        # FINISH handler before publishing READY.
        with self._lock:
            timer = self._timer.get(timer_name)
            if timer is not None:
                timer.destroy()
                self._timer[timer_name] = None

    def stop_all(self):
        # Snapshot the keys under the lock so concurrent set_timer /
        # stop calls can't mutate the dict mid-iteration.
        with self._lock:
            names = list(self._timer.keys())
        for timer_name in names:
            self.stop(timer_name)

    def set_timer(self, timer_name, timer_frequency, callback_function):
        with self._lock:
            self._timer[timer_name] = None
            self._timer_frequency[timer_name] = timer_frequency
            self._timer_callback[timer_name] = callback_function
