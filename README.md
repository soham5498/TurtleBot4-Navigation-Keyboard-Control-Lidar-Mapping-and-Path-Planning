# TurtleBot4 Navigation, Keyboard Control, Lidar Mapping, and Path Planning

## Overview
This project implements navigation, mapping, and path planning functionalities for the TurtleBot4 using ROS2. It includes a static mapping node for occupancy grid creation and a path planning node using the A* algorithm for autonomous navigation.

## Project Structure
```
TurtleBot4-Navigation-Keyboard-Control-Lidar-Mapping-and-Path-Planning/
├── src/
│   ├── mapping_package/
│   │   ├── mapping_package/
│   │   │   ├── mapping_node.py
│   │   │   ├── path_planning_node.py
│   │   ├── resource/
│   │   ├── test/
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── setup.cfg
│   │   ├── setup.py
├── LICENSE
```

## Nodes
### 1. Mapping Node (`mapping_node.py`)
This ROS2 node creates an occupancy grid map using odometry and laser scan data. It processes data from `/odom` and `/scan` topics, dynamically adjusts the map boundaries, and publishes the occupancy grid to `/map`.

#### **Key Features:**
- Subscribes to odometry (`/odom`) and laser scan (`/scan`) topics.
- Uses a static transform from `map` to `odom`.
- Implements an adaptive occupancy grid that expands dynamically.
- Publishes the occupancy grid map to `/map`.
- Converts quaternion-based orientation to Euler angles.
- Implements methods to determine robot stationarity and adjust the map-to-odom transform accordingly.
- Uses Bresenham’s algorithm to mark occupied and free spaces based on LiDAR readings.

#### **Dependencies:**
- `rclpy` (ROS2 Python Client Library)
- `nav_msgs.msg` (For occupancy grid and odometry messages)
- `sensor_msgs.msg` (For laser scan messages)
- `geometry_msgs.msg` (For transformations)
- `tf2_ros` (For transformation handling)
- `numpy` (For matrix operations)
- `scipy.spatial.transform` (For quaternion transformations)

#### **Usage:**
To launch the mapping node:
```bash
ros2 run mapping_package mapping_node.py
```

---

### 2. Path Planning Node (`path_planning_node.py`)
This ROS2 node implements an A* path planning algorithm for autonomous navigation. It processes odometry and map data, determines the optimal path to a goal, and sends navigation commands.

#### **Key Features:**
- Subscribes to odometry (`/odom`), occupancy grid (`/map`), and goal (`/goal`).
- Uses A* search algorithm for path planning.
- Avoids obstacles and dynamically replans paths.
- Publishes velocity commands to `/cmd_vel`.
- Uses transforms (`map → odom` and `odom → base_link`).
- Publishes goal paths as visual markers.
- Uses heuristic optimization to improve path planning efficiency.

#### **Dependencies:**
- `rclpy` (ROS2 Python Client Library)
- `nav_msgs.msg` (For occupancy grid and odometry messages)
- `geometry_msgs.msg` (For transformations and goal poses)
- `visualization_msgs.msg` (For publishing visualization markers)
- `tf2_ros` (For transformation handling)
- `scipy.spatial.transform` (For quaternion transformations)
- `numpy` (For mathematical computations)

#### **Usage:**
To launch the path planning node:
```bash
ros2 run mapping_package path_planning_node.py
```

## References
The implementation is based on the following references:
- [ROS2 Tutorials](https://docs.ros.org/en/humble/Tutorials.html)
- [ROS2 tf2 Static Broadcaster](https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html)
- [Occupancy Grid Mapping in ROS](https://answers.ros.org/question/337215/getting-deeper-into-map-occupancy-grid/)
- [Lidar Mapping for Autonomous Robots](https://automaticaddison.com/set-up-lidar-for-a-simulated-mobile-robot-in-ros-2/)
- [A* Path Planning](https://github.com/fazildgr8/ros_autonomous_slam/blob/master/nodes/a_star_main.py)

## Authors
- Soham Joita
- lakshmi chandrasekharan


