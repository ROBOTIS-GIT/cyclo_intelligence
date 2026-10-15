# House TV missions

TurnOnTV and TurnOffTV share one TVControl waypoint. Their Local BT checks the simulation station, performs a right-arm control gesture, calls HomeTV, and retracts. Commands are idempotent and may repeat indefinitely.

The workspace house observation plan includes TVObservation, after the dining and laundry spots. observation.json records the added spot and plan entry; merge this entry into an existing plan rather than replacing the entire map plan. Autonomy returns to the trigger spot after mission completion.

HomeTV is discovered by the BT registry. The ffw_bringup home_tv ROS bridge and ffw_home_tv Gazebo plugin must be running. The updated SG2 Home launch starts them after a simulator restart. See ai_worker/ffw_bringup/worlds/HOME_TV_SCENE.md for scene setup and service details.

The house map file is unchanged. Poses use the existing Home/house world-to-map transform. Qwen must be evaluated or trained on TV and viewer scenes before automatic mission selection can be relied upon.
