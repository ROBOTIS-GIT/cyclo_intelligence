"""Cyclo-owned event-status confirmation over the existing rmw_zenoh cache.

No controller heartbeat or service is added. A unique source/status endpoint pair
must share a discovered (session, node). Discovery does not carry ROS GIDs: the
pair's distinct GIDs are bound by session-scoped cache replies, with exactly one
reply per topic. Live sequence ordering and generation tokens protect those binds.

Confirmation bounds missed-event detection; it is not atomic controller-side
arbitration. Cache failure or discovery loss revokes output, including on close.
"""

import threading


class CacheConfirmation:
    """Serialize metadata decisions, never hold a lock across a network query."""

    def __init__(self, guard):
        self.guard = guard
        self.lock = threading.RLock()
        self.endpoints = None
        self.identity = None
        self.generation = 0
        self.bindings = {}
        self.latest = {}
        self.source_watermark = None
        self.closed = False

    def set_publishers(self, endpoints):
        with self.lock:
            if self.closed or endpoints == self.endpoints:
                return
            self.endpoints = {key: frozenset(value) for key, value in endpoints.items()}
            self.generation += 1
            self.bindings.clear()
            self.latest.clear()
            self.source_watermark = None
            source, status = endpoints["source"], endpoints["status"]
            if source or status:
                self.guard.observe_external_control()
            self.identity = None
            if len(source) == len(status) == 1:
                source, status = next(iter(source)), next(iter(status))
                if source[:2] == status[:2]:
                    self.identity = source[:2]
            self.guard.invalidate_confirmation(
                "publisher_changed" if self.identity else "publisher_unavailable", reset_source=True)

    def live(self, stream, session, gid, sequence, msg):
        with self.lock:
            if self.closed or self.identity is None or session != self.identity[0]:
                return
            bound = self.bindings.get(stream)
            if bound is not None and gid != bound:
                self.guard.invalidate_confirmation("publisher_identity_changed")
                return
            if stream == "source" and self.source_watermark is not None and sequence < self.source_watermark[1]:
                return
            previous = self.latest.get(stream)
            if previous and previous[0] == gid and sequence <= previous[1]:
                return
            self.latest[stream] = (gid, sequence, msg)
            if stream == "status":
                # Includes ERROR arriving while the first cache query is in flight.
                self.guard.update_status(msg)
            elif bound == gid:
                self.guard.update_source(msg)

    def reconcile(self, query):
        with self.lock:
            if self.closed or self.identity is None:
                return False
            generation, identity = self.generation, self.identity
            token = self.guard.confirmation_token()
            latest = self.latest.get("status")
        try:
            source = query("source", identity)
            status = query("status", identity)
        except Exception:
            with self.lock:
                if not self.closed and generation == self.generation:
                    self.guard.invalidate_confirmation("status_query_failed")
            return False
        with self.lock:
            if (self.closed or generation != self.generation or identity != self.identity
                    or latest is not self.latest.get("status")
                    or token != self.guard.confirmation_token()):
                return False
            for stream, sample in (("source", source), ("status", status)):
                gid, seq, _msg = sample
                previous = self.latest.get(stream)
                if (self.bindings.get(stream, gid) != gid
                        or (previous and (previous[0] != gid or (stream == "status" and seq < previous[1])))):
                    self.guard.invalidate_confirmation("status_cache_replay")
                    return False
            self.bindings.update(source=source[0], status=status[0])
            # Cache raises the sequence floor, never the source freshness clock.
            # An unchanged value ahead of live traffic does not churn the lease.
            live_source = self.latest.get("source")
            if (self.source_watermark is None or source[1] > self.source_watermark[1]):
                self.source_watermark = source
                if live_source is None or source[1] > live_source[1]:
                    self.guard.observe_cached_source(source[2])
            self.latest["status"] = status
            return self.guard.confirm_status(token, status[2])

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.generation += 1
                self.guard.invalidate_confirmation("handoff_closed")


class HandoffTransport:
    """Small subscriber adapter; all blocking work belongs to its bounded worker."""

    def __init__(self, guard, status_definition, **common):
        import zenoh
        from zenoh_ros2_sdk import ROS2Subscriber
        from zenoh_ros2_sdk.attachment import Attachment

        self.coordinator = CacheConfirmation(guard)
        self._stop = threading.Event()
        self._subscribers = {}
        self._watch = None
        self._thread = None
        self._tokens = {"source": set(), "status": set()}
        self._discovery_lock = threading.Lock()
        self._query_timeout = guard.confirmation_interval

        def decode(sub, sample):
            metadata = Attachment.from_bytes(sample.attachment.to_bytes())
            if not metadata.gid:
                raise ValueError("Missing publisher GID")
            msg = sub.session_mgr.store.deserialize_cdr(sample.payload.to_bytes(), sub.msg_type)
            return metadata.gid, metadata.sequence_id, msg

        self._decode = decode
        owner = self

        class MetadataSubscriber(ROS2Subscriber):
            def __init__(self, stream, **kwargs):
                self.stream = stream
                super().__init__(callback=lambda msg: None, **kwargs)

            def _process_sample(self, sample):
                try:
                    session = str(sample.source_info.source_id.zid)
                    owner.coordinator.live(self.stream, session, *decode(self, sample))
                except Exception:
                    owner.coordinator.guard.invalidate_confirmation("status_metadata_invalid")

        try:
            for stream, topic, kind, definition in (
                ("source", guard.source_topic, "std_msgs/msg/String", "string data\n"),
                ("status", guard.control_status_topic, "robotis_interfaces/msg/ControlModeStatus", status_definition),
            ):
                self._subscribers[stream] = MetadataSubscriber(
                    stream, topic=topic, msg_type=kind, msg_definition=definition, **common)
            sub = self._subscribers["source"]
            self._session = sub.session_mgr.session

            def discovery(sample):
                parts = str(sample.key_expr).split("/")
                if len(parts) != 13 or parts[5] != "MP":
                    return
                with self._discovery_lock:
                    for stream, subscriber in self._subscribers.items():
                        if (parts[9] == subscriber.topic.replace("/", "%")
                                and parts[10] == subscriber.dds_type_name and parts[11] == subscriber.type_hash):
                            endpoint = tuple(parts[2:5])
                            if sample.kind == zenoh.SampleKind.PUT:
                                self._tokens[stream].add(endpoint)
                            else:
                                self._tokens[stream].discard(endpoint)
                            self.coordinator.set_publishers(self._tokens)

            self._watch = sub.session_mgr.liveliness.declare_subscriber(
                f"@ros2_lv/{sub.domain_id}/**", discovery, history=True)
            self._thread = threading.Thread(target=self._run, args=(guard.confirmation_interval,),
                                            name="handoff-confirmation", daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise

    def _query(self, stream, identity):
        if self._stop.is_set():
            raise RuntimeError("Handoff closed")
        sub = self._subscribers[stream]
        selector = f"{sub.keyexpr}/@adv/pub/{identity[0]}/**?_anyke;_max=1"
        samples = []
        for reply in self._session.get(selector, timeout=self._query_timeout):
            if (reply.ok is None or str(reply.ok.key_expr) != sub.keyexpr
                    or str(reply.replier_id.zid) != identity[0]):
                raise ValueError("Invalid cache reply provenance")
            samples.append(self._decode(sub, reply.ok))
        if len(samples) != 1:
            raise ValueError("Expected exactly one current publisher cache reply")
        return samples[0]

    def _run(self, interval):
        while not self._stop.is_set():
            self.coordinator.reconcile(self._query)
            self._stop.wait(interval)

    def close(self):
        with self.coordinator.lock:
            if self._stop.is_set():
                return
            self.coordinator.close()
            self._stop.set()
        if self._thread is not None:
            self._thread.join(2 * self._query_timeout + 1.)
        if self._watch is not None:
            self._watch.undeclare()
        for subscriber in self._subscribers.values():
            subscriber.close()
