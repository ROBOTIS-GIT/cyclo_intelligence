# BT action policy switch

## Request
Add one BT action node with Use VLA Action / Use MLP Action and Actor / Critic / RL checkpoint selection. Default checkpoint name is latest, consistent with the user's current workflow.

## Proposed interface
`SwitchActionPolicy` exposes `action_source` (vla/mlp), `training_stage` (actor/critic/rl), and `bundle_name` (default latest). Stage and name are relevant only for MLP. Resolve the three stages under `/workspace/checkpoint/rlt/stage2/{actor,critic,rl_training}/{bundle_name}`. Critic means the Actor contained in the Critic warmup bundle; the Critic network does not emit robot actions.

## Existing constraints
Main command 20 switches the loaded base/RLT route at a buffer boundary and requires a running session. Policy-update commands reject a different bundle path. Existing SendCommand LOAD does not expose RLT bundle settings, including in the supervised BT path. A functional stage selector therefore needs actual bundle loading and lifecycle ownership, not merely a dropdown or route flag.

## Updated decision: preload with GR00T LOAD
The user requested removing model-loading delays from SwitchActionPolicy. Before executing the active tree, collect its MLP stage/bundle selection and attach it to GR00T LOAD nodes. One tree supports one MLP bundle; conflicting selections fail before execution. No additional user port is required.

LOAD verifies the selected manifest, training phase, artifact sizes and hashes, then sends rlt_enabled and rlt_bundle_path with the initial model load. The initial route remains VLA. Successful LOAD pins the bundle identity; replacing latest on disk does not change running weights. A changed manifest during LOAD fails before START.

SwitchActionPolicy only changes the loaded output route with command 20. It never STOPs, UNLOADs, LOADs, or STARTs a model. An unloaded or mismatched selection fails explicitly. A paused session only records its desired route for the next explicit RESUME. Critic selects the Actor contained in its bundle, not the Critic network output.

## Validation
Test preload path and artifact validation, native ROS request serialization, route-only switches, latest pinning, conflicting selections, pause/resume and cancellation. Validate the saved user tree without executing it. Build/deploy UI help and restart only BT after a confirmed stop. No automatic physical robot test.

## Explicit LOAD selection
The saved refilling_bottles_mlp.xml now puts training_stage="actor" and bundle_name="latest" on LoadGroot. Its two SwitchActionPolicy nodes specify only action_source. Explicit LOAD selection is authoritative; legacy trees with stage/bundle on switches remain supported. Canvas serialization preserves these LOAD ports.
