"""All version-specific upstream integration lives here; no websocket server."""


def load_native(bundle, precision):
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from lingbotvla.data.vla_data.utils import FeatureTransform

    server = LingbotVLAv2Server(
        path_to_pi_model=str(bundle.weights), robot_norm_path=str(bundle.norm_stats),
        chunk_ret=True, use_length=-1, use_compile=False,
        use_bf16=precision == "bfloat16", use_fp32=precision == "float32",
    )
    # reset(robot_name) uses a CWD-relative configs/robot_configs path. Bind the
    # exact exported training assets instead, without altering upstream files.
    transform = FeatureTransform(
        str(bundle.robot_config), server.data_config, server.config, server.processor,
        chunk_size=server.config.chunk_size, norm_stats_path=str(bundle.norm_stats),
    )
    for kind, expected in (("states", [bundle.metadata["state_key"]]),
                           ("actions", [bundle.metadata["action_key"]]),
                           ("images", bundle.metadata["cameras"])):
        if set(transform.org_features[kind]) != set(expected):
            raise ValueError(f"Saved FeatureTransform has unexpected {kind}")
    if server.config.chunk_size != bundle.horizon:
        raise ValueError("Upstream action horizon differs from exported configuration")
    server.vla.feature_transform = transform
    server.action_key = transform.org_features["actions"]
    return server


def reset_native(server):
    # Full chunks do not consume the upstream step cache, but release it on every
    # session boundary and after warmup so no previous instruction is retained.
    server.global_step = 0
    server.last_action_chunk = None
    server.last_normalized_action_chunk = None
