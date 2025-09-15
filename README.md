# px4-safety-governor (ROS 2 Humble)

Simple PX4 Offboard velocity safety governor with two modes: a baseline (no avoidance) and an exponential repulsive-field avoidance. Includes a visualizer.

## Prerequisites
- ROS 2 Humble installed and sourced in your shell
- PX4 SITL or a real PX4 vehicle with the uXRCE-DDS bridge running (so `/fmu/*` topics are available)

## Workspace setup
1) Create a workspace in your home directory

```bash
cd ~
mkdir -p px4-safety-governor-ws/src
cd px4-safety-governor-ws/src
```

2) Clone dependencies into `src`

```bash
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/ppgerard/px4-safety-governor.git
```

3) Build from the workspace root

```bash
cd ..
colcon build
```

4) Source the overlay

```bash
source install/setup.bash
```

## Running
Make sure PX4 (SITL or real) is up and publishing `/fmu/*` topics via uXRCE-DDS.

Arm the vehicle and switch to Offboard mode (e.g., using QGroundControl) for the node to take control. The node continuously publishes Offboard setpoints, but PX4 will only execute them when armed and in Offboard.

- Baseline (no avoidance): flies back and forth between two waypoints.
```bash
ros2 run px4_offboard safety_governor_no_avoidance
```

- Repulsive-field avoidance: adds exponential repulsion away from people.
```bash
ros2 run px4_offboard safety_governor_repulsive_field
```

- Visualizer.
```bash
ros2 run px4_offboard visualizer
```

- Launch files
```bash
ros2 launch px4_offboard safety_governor_no_avoidance.launch.py
# or
ros2 launch px4_offboard safety_governor_repulsive_field.launch.py
# and optional RViz
ros2 launch px4_offboard visualize.launch.py
```

## Notes
- Operates in PX4 local NED; Offboard velocity control is used.
- People are modeled as a static list in code by default (no subscription). Visualizer republishes their relative (body-frame) positions for RViz.
- Repulsive-field node enforces a contextual speed cap (near vs. free) and a global limit; the repulsion magnitude scales with the current cap.
