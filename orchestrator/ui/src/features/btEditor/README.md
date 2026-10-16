<!--
Copyright 2026 ROBOTIS CO., LTD.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Seongwoo Kim
-->

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
