# RLDX policy runtime

This directory mirrors the LeRobot/GR00T policy layout:

- `RLDX-1/` is the pinned upstream/forked RLDX repository submodule.
- `rldx_engine/` is the Cyclo-side runtime adapter that converts RobotClient
  observations/actions to the RLDX policy API used by the common runtime.
- `zmq_transport/client.py` is the Cyclo-side ZMQ client for RLDX PolicyServer
  endpoints.
- `zmq_transport/server/run_policy_server.sh` is the local GPU-side ZMQ
  PolicyServer wrapper.

Cyclo robot modality parameters live in the submodule itself, so the RLDX
server can import them through the normal RLDX API instead of through a Cyclo
side patch at runtime.

Initialize the RLDX submodule before running the local server:

```bash
git submodule update --init cyclo_brain/policy/rldx/RLDX-1
docker/container.sh start-rldx-server
```

`RLDX_REPO_DIR` can still point to an external checkout for development, but
the default path is the submodule above.

RTC is enabled by default for the local policy server with:

```bash
RLDX_SERVER_RTC_INFERENCE_MODE=guided
RLDX_SERVER_RTC_INFERENCE_DELAY=4
RLDX_SEND_ACTION_PREFIX=true
RLDX_RTC_PREFIX_LEN=4
RLDX_RTC_EXEC_HORIZON=12
RLDX_ACTION_ALIGNMENT_MODE=rtc
RLDX_REFILL_MARGIN_S=0.27
REFILL_STRATEGY=auto
```

`guided` works with ordinary flow-matching checkpoints. Use
`RLDX_SERVER_RTC_INFERENCE_MODE=trained` only for checkpoints trained with RTC,
or `none`/`checkpoint` to disable the wrapper override. The Cyclo-side `rtc`
alignment does not do L2 chunk matching or boundary blending. With
`REFILL_STRATEGY=auto`, the control loop treats `RLDX_REFILL_MARGIN_S` as the
fixed RTC prefix window, so `0.27s` matches four 15 Hz source steps. The client
sends the previous raw RLDX chunk's scheduled prefix through
`options["action_prefix"]`, drops the already-scheduled prefix from the returned
chunk, and then resamples for the robot control loop.

When changing files under `RLDX-1/`, commit and push those changes in the
RLDX-1 repository first, then commit the updated submodule pointer in this
repository.
