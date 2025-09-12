# px4-safety-governor (ROS 2 Humble)

Simple safety governor that drives a PX4 vehicle in Offboard velocity mode and slows down near people. Includes a minimal visualizer and two launch files.

## Prerequisites
- ROS 2 Humble installed and sourced in your shell
- PX4 SITL or a real PX4 vehicle with the uXRCE-DDS bridge running (so `/fmu/*` topics are available)

## Workspace setup (copy-paste)
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

- Minimal governor without geometric avoidance
```bash
ros2 run px4_offboard safety_governor_no_avoidance
```

- Full governor (with avoidance logic)
```bash
ros2 run px4_offboard safety_governor
```

- Visualizer only
```bash
ros2 run px4_offboard visualizer
```

- Launch files
```bash
ros2 launch px4_offboard safety_governor_no_avoidance.launch.py
# or
ros2 launch px4_offboard safety_governor.launch.py
# and optional RViz
ros2 launch px4_offboard visualize.launch.py
```

## Note
- This package operates in PX4 local NED. People are published as a `PoseArray` in the body frame.
