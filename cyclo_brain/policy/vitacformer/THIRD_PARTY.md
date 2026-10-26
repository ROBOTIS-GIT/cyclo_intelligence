# ViTacFormer source attribution

`vitacformer_engine/model.py` adapts the transformer, ResNet backbone and
DETR/CVAE policy structure from
[RoboVerseOrg/ViTacFormer](https://github.com/RoboVerseOrg/ViTacFormer/tree/d94788272b5f5e18a80bf62e3e5d51e7d2581d77),
commit `d94788272b5f5e18a80bf62e3e5d51e7d2581d77`.
The original repository and its `detr` code use Apache-2.0, the same license
as this repository (see the root `LICENSE`). The DETR source carries
Copyright (c) Facebook, Inc. and its affiliates.

The Cyclo adaptation reconstructs the SH5 checkpoint graph without a training
repository checkout. It adds the SH5 input dimensions, normalization, bounded
action decoding, arm warm-start ramp, right tactile persistence conditioning,
artifact validation and Cyclo engine interface. These modifications are
identified in the source and are not an unmodified upstream ViTacFormer release.

The two `messages/robotis_interfaces/msg` definitions are the ROBOTIS SH5 wire
definitions used by the existing deployment. They are mounted into the Zenoh
SDK's message registry so this backend does not require edits to that submodule.
