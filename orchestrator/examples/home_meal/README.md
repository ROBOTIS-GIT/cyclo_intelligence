# Home meal missions

These missions use the existing `Home` map and the `home_autonomy` Gazebo world.
The editable copies are installed in `/workspace/navigation/missions/Home/`.
This directory preserves the initial mission definitions; subsequent Design edits
belong to the workspace copies.

| Mission | Behavior |
| --- | --- |
| ObservationMode | Optional manual return mission. Operate uses the map observation plan instead. |
| CleanTable | Table → Sink: collect the empty dishes and stack them at the sink. |
| ServeDish | Sink → Table → Fridge → Table: return the dishes, wait at the refrigerator, and serve food. The two Table visits have different local BT actions. |

Each reach uses joint trajectories and waits for joint feedback. `HomeMeal`
changes props only after a successful reach. Its `check_only` mode verifies the
meal state and station before extending the arms. The simulator rejects out-of-order
commands and commands away from the correct station. This is staged manipulation:
no grasp policy, dish attachment to hands, cooking physics, or door opening is used.
The same three plate entities are reused, keeping their IDs stable for annotation.

## Meal cycle

The world starts with three residents seated at the food-filled table. After a
2-second seated opening, they eat for 30 seconds. The food disappears first;
they remain seated for 0.75 seconds before standing. Once all three stand up,
CleanTable becomes applicable. They then disperse along their household circuits,
pause on their return route until CleanTable has stacked the dishes at the sink, then sit and wait until ServeDish serves the next meal. This gate repeats every cycle; the scene itself does not dispatch robot missions.

Each local BT performs its operation once per mission run. It resets on the next
run, so pickup, sink stacking, dish placement, and food serving can all execute
again on every meal cycle. Do not place an infinite Loop around an individual
HomeMeal action: the second immediate call would be invalid in the current
scene phase. Continuous robot operation requires a supervisor such as Operate
to select and start CleanTable and ServeDish again after each completed mission.
The scene controller stages this cycle; it does not choose or start missions.
Qwen receives the Home mission descriptions and start conditions through the
normal catalog. Existing warehouse LoRA labels are not automatically retrained
or remapped to these Home missions. Home image/decision data is needed for that.

## Running

Build `ffw_gazebo_physics`, `ffw_bringup` in ai_worker and `orchestrator` in
cyclo_intelligence, then restart the existing SG2 Gazebo launch:

```bash
ros2 launch ffw_bringup ffw_sg2_follower_ai_gazebo.launch.py
```

Load **Home**, localize, and select CleanTable or ServeDish in Mission Canvas.
The launch starts the `/simulation/home_meal/*` ROS services automatically.
A scene/world restart resets plates, food, and resident clocks. Stopping midway
preserves the current prop state; it does not silently reset the scene.

`scene_config.json` records station poses and the Home map transform. This map
was recorded with the robot's initial position as the SLAM origin:
`map_x = 5.74224035 - world_y`, `map_y = world_x - 2.93956349`.
If the map is rebuilt with a different origin, update the mission poses in Design.
Nav2 plans directly between these task stations; the authored missions no longer include transit waypoints.

Operate stores its observation plan in `maps/Home/autonomy.json`. After a task,
the server controller returns to the configured spot (Fixed) or visits the next
spot (Patrol). Qwen stays loaded during navigation and local BT execution;
inference resumes only after arrival. The legacy ObservationMode mission is
excluded from Qwen's task candidates. Copy both `maps` and `missions` when
installing this example. See [Autonomy controller](../../../docker/supervisor_api/AUTONOMY_KO.md).
