# Shared behavior-tree editor

This feature owns reusable BT graph and parameter editing, independent of any
Canvas document or robot execution session.

- `lib/`: graph transforms, layout/visibility, and node types.
- `hooks/`: graph state, commands, deletion, rendered nodes, and common keyboard
  behavior. Action and Mission supply their own history/document adapters.
- `parameters/`: pure parameter policy and joint normalization, field components,
  and the parameter draft hook.
- Catalog state is shared while retaining the existing reducer/action contract.

Action-specific files, Redux document persistence, and ROS execution live under
`actionCanvas`; Mission-specific files and document adapters live under
`missionCanvas/btEditor`. Do not import either Canvas into this shared feature.
The old navigation editor and parameter-panel paths are compatibility adapters.
