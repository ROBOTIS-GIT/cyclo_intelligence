# House laundry mission

The `house` map receives copies of Home's CleanTable, ServeDish and ObservationMode, plus DoLaundry. Original Home files are preserved. Every copied mission has a new reasoning ID.

DoLaundry has two navigation goals: LaundryBasket (world 9.04, -0.30, yaw 0) and WashingMachine (9.00, -2.05, yaw 0). Each Local BT checks the laundry state, reaches with the existing JointControl pose, applies the prop transition, and retracts. WashingMachine loads the laundry and starts the animated 24-second wash; the mission completes once washing starts.

The patrol plan visits the dining observation spot (CleanTable / ServeDish) and LaundryObservation (DoLaundry). LaundryObservation is in front of the basket at world (8.50, -0.30), facing east. Autonomy returns to the triggering observation spot after a mission, then resumes patrol. Direct Run executes just the two mission waypoints.

Coordinate conversion uses the Home/house SLAM origin: `map_x = 5.74224035 - world_y`, `map_y = world_x - 2.93956349`, `map_yaw = world_yaw + pi/2`. Verify localization before executing; a differently initialized map requires updating poses.

`HomeLaundry` is automatically discovered by the BT registry. The `home_laundry` ffw_bringup node exposes `/simulation/home_laundry/{pickup_basket,load_washer,start_wash}` and their `check_` variants. It calls the Gazebo `/world/home_autonomy/home_laundry` services. Checks are read-only; mutating commands are never retried. These operations simulate prop transfer, not physical grasping.

The SG2 home launch starts the bridge. Rebuild ffw_bringup and orchestrator after adding the node. The example `missions/house` and `maps/house` directories mirror the installed workspace configuration; do not overwrite edited missions without reviewing them.

Selecting the new mission and observation plan does not train Qwen to recognize laundry. Validate or label/train that scenario separately before relying on automatic selection.
