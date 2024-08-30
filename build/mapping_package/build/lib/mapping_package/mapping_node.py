
'''
Refrences:

https://docs.ros.org/en/humble/Tutorials.html,
https://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Static-Broadcaster-Py.html,
https://answers.ros.org/question/337215/getting-deeper-into-map-occupancy-grid/,
https://answers.ros.org/question/286221/create-2d-occupancy-grid-map-by-laser-data/,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html,
https://automaticaddison.com/set-up-lidar-for-a-simulated-mobile-robot-in-ros-2/,
https://autowarefoundation.github.io/autoware.universe/main/perception/probabilistic_occupancy_grid_map/laserscan-based-occupancy-grid-map/,
https://github.com/ros2/geometry2/blob/rolling/tf2_geometry_msgs/src/tf2_geometry_msgs/tf2_geometry_msgs.py,
https://github.com/salihmarangoz/robot_laser_grid_mapping/blob/main/scripts/grid_mapping.py
'''

"""
StaticMapNode Class and Main Function

This module defines the StaticMapNode class, which is a ROS2 node responsible for creating a static occupancy grid map
from odometry and laser scan data, and broadcasting necessary transforms. The main function initializes and runs this node.

Classes:
    StaticMapNode(Node): ROS2 node for creating and managing an occupancy grid map based on odometry and laser scan data.

Functions:
    main(args=None): Initializes and runs the StaticMapNode.

StaticMapNode Methods:
    __init__(self): Initializes the StaticMapNode, setting up subscriptions, publishers, and initial transforms.
    initialize_static_transform(self): Sets up the initial static transform between "map" and "odom" frames.
    odometry_callback(self, msg): Callback function for odometry messages, updating pose and broadcasting transforms.
    scan_callback(self, msg: LaserScan): Callback function for laser scan messages, updating the occupancy grid.
    quat_to_euler(self, quat): Converts a quaternion to Euler angles.
    is_stationary(self): Checks if the robot is stationary based on odometry data.
    adjust_map_to_odom_transform(self): Adjusts the static transform between "map" and "odom" based on the latest odometry.
    update_grid(self, x0, y0, x1, y1): Updates the occupancy grid based on the laser scan endpoints.
    expand_grid_left(self): Expands the occupancy grid to the left.
    expand_grid_right(self): Expands the occupancy grid to the right.
    expand_grid_up(self): Expands the occupancy grid upward.
    expand_grid_down(self): Expands the occupancy grid downward.
    publish_map(self): Publishes the current occupancy grid as a ROS message.
    broadcast_transform(self): Broadcasts the dynamic transform from "odom" to "base_link".
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
import tf2_ros
from scipy.spatial.transform import Rotation as R
import numpy as np
from collections import deque

class StaticMapNode(Node):
    """
        Initializes the StaticMapNode, setting up subscriptions, publishers, and initial transforms.

        This method initializes various parameters like the robot's pose, occupancy grid resolution, 
        and origin. It also sets up ROS2 subscriptions to odometry and laser scan topics, and a publisher 
        for the occupancy grid map. It also sets up broadcasters for static and dynamic transforms.
    """
    def __init__(self): 
        super().__init__('static_map_node')
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta
        self.resolution = 0.1  # 10 cm per cell
        self.origin = [0.0, 0.0]
        self.occupancy_grid = np.full((200, 200), -1)  # Initialize with unknown values
        self.width = self.occupancy_grid.shape[1]
        self.height = self.occupancy_grid.shape[0]

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odometry_callback, qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.odom_buffer = deque(maxlen=10)
        self.stationary_threshold = 0.1
        self.stationary_duration = 5.0

        self.position_msg = None  # Store the latest odometry message

        self.get_logger().info('StaticMapNode initialized') 

        # Set up the static transform at initialization
        self.initialize_static_transform()


    """
        Sets up the initial static transform between "map" and "odom" frames.

        This method creates a static transform from the "map" frame to the "odom" frame with no translation 
        or rotation, and broadcasts it using the static transform broadcaster.
    """
    
    def initialize_static_transform(self):
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "map"
        static_transform.child_frame_id = "odom"
        static_transform.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        static_transform.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.tf_static_broadcaster.sendTransform(static_transform)
        self.get_logger().info('Initial static transform from map to odom broadcasted')

    """
        Callback function for odometry messages, updating pose and broadcasting transforms.

        Args:
            msg (Odometry): The odometry message received from the subscription.
        
        This method updates the robot's pose based on the odometry data, broadcasts a transform from 
        "map" to "odom", and checks if the robot is stationary. If stationary, it adjusts the static transform.
    """
    def odometry_callback(self, msg):
        self.position_msg = msg
        self.odom_buffer.append(msg)
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self.pose[0] = position.x
        self.pose[1] = position.y
        self.pose[2] = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_euler('xyz')[2]

        # Create and send a transform from map to odom
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = "map"
        transform_stamped.child_frame_id = "odom"
        transform_stamped.transform.translation = Vector3(x=position.x, y=position.y, z=0.0)
        transform_stamped.transform.rotation = orientation

        self.tf_static_broadcaster.sendTransform(transform_stamped)
        self.get_logger().info('Transform from map to odom broadcasted in odometry callback')

        if self.is_stationary():
            self.adjust_map_to_odom_transform()

        self.broadcast_transform()  # Ensure dynamic transform is broadcasted

    """
        Callback function for laser scan messages, updating the occupancy grid.

        Args:
            msg (LaserScan): The laser scan message received from the subscription.

        This method updates the occupancy grid based on the laser scan data and the robot's current position.
        It processes each laser scan point, transforms it into the map frame, and updates the grid cells.
    """
    def scan_callback(self, msg: LaserScan):
        if self.position_msg is None:
            return
        x0 = self.position_msg.pose.pose.position.x
        y0 = self.position_msg.pose.pose.position.y
        th = self.quat_to_euler(self.position_msg.pose.pose.orientation)[2]

        for idx in range(len(msg.ranges)):
            ang = np.pi / 2 + msg.angle_min + (msg.angle_max - msg.angle_min) * idx / (len(msg.ranges) - 1)
            rng = msg.ranges[idx]
            intensity = msg.intensities[idx] if msg.intensities else None

            if not np.isfinite(rng) or (intensity is not None and not np.isfinite(intensity)):
                continue

            x1 = x0 + rng * np.cos(ang + th)
            y1 = y0 + rng * np.sin(ang + th)
            self.update_grid(x0, y0, x1, y1)

        self.publish_map()  # Update and publish the map after processing each scan

    """
        Converts a quaternion to Euler angles.

        Args:
            quat (Quaternion): The quaternion to be converted.

        Returns:
            numpy.ndarray: An array of Euler angles (roll, pitch, yaw).
    """
    def quat_to_euler(self, quat):
        return R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')


    """
        Checks if the robot is stationary based on odometry data.

        Returns:
            bool: True if the robot is stationary, False otherwise.

        This method calculates the total distance traveled based on the odometry buffer. If the total distance is less 
        than the stationary threshold and the duration exceeds the stationary duration, the robot is considered stationary.
    """
    def is_stationary(self):
        if len(self.odom_buffer) < self.odom_buffer.maxlen:
            return False  # Not enough data yet

        distances = [
            np.linalg.norm(np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]))
            for msg in self.odom_buffer
        ]
        total_distance = np.abs(distances[-1] - distances[0])
        
        if total_distance < self.stationary_threshold:
            # Check if the robot has been stationary for the required duration
            time_diff = (
                self.odom_buffer[-1].header.stamp.sec - self.odom_buffer[0].header.stamp.sec
            )
            return time_diff >= self.stationary_duration
        else:
            return False

    """
        Adjusts the static transform between "map" and "odom" based on the latest odometry.

        This method updates the static transform from the "map" frame to the "odom" frame using the latest odometry data 
        and broadcasts it.
    """
    def adjust_map_to_odom_transform(self):
        latest_odom = self.odom_buffer[-1]
        position = latest_odom.pose.pose.position
        orientation = latest_odom.pose.pose.orientation
        
        static_transform = TransformStamped()
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "map"
        static_transform.child_frame_id = "odom"
        static_transform.transform.translation = Vector3(x=position.x, y=position.y, z=0.0)
        static_transform.transform.rotation = orientation

        self.tf_static_broadcaster.sendTransform(static_transform)
        self.get_logger().info('Static transform from map to odom broadcasted')


    """
        Updates the occupancy grid based on the laser scan endpoints.

        Args:
            x0 (float): Starting x-coordinate of the laser scan in the map frame.
            y0 (float): Starting y-coordinate of the laser scan in the map frame.
            x1 (float): Ending x-coordinate of the laser scan in the map frame.
            y1 (float): Ending y-coordinate of the laser scan in the map frame.

        This method updates the occupancy grid cells along the path from (x0, y0) to (x1, y1) using Bresenham's line algorithm.
        It marks free space along the path and occupied space at the endpoint.
    """
    def update_grid(self, x0, y0, x1, y1):
        min_x = min(x0, x1)
        min_y = min(y0, y1)
        max_x = max(x0, x1)
        max_y = max(y0, y1)

        while min_x < self.origin[0]:
            self.expand_grid_left()
        while max_x >= self.origin[0] + self.width * self.resolution:
            self.expand_grid_right()
        while min_y < self.origin[1]:
            self.expand_grid_down()
        while max_y >= self.origin[1] + self.height * self.resolution:
            self.expand_grid_up()

        x0_idx = int((x0 - self.origin[0]) / self.resolution)
        y0_idx = int((y0 - self.origin[1]) / self.resolution)
        x1_idx = int((x1 - self.origin[0]) / self.resolution)
        y1_idx = int((y1 - self.origin[1]) / self.resolution)

        dx = abs(x1_idx - x0_idx)
        dy = abs(y1_idx - y0_idx)
        sx = 1 if x0_idx < x1_idx else -1
        sy = 1 if y0_idx < y1_idx else -1
        err = dx - dy

        while True:
            if 0 <= x0_idx < self.width and 0 <= y0_idx < self.height:
                self.occupancy_grid[y0_idx, x0_idx] = 0  # Mark free space along the ray
            if x0_idx == x1_idx and y0_idx == y1_idx:
                if 0 <= x0_idx < self.width and 0 <= y0_idx < self.height:
                    self.occupancy_grid[y0_idx, x0_idx] = 100  # Mark occupied cell at the end
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0_idx += sx
            if e2 < dx:
                err += dx
                y0_idx += sy

    """
        Expands the occupancy grid to the left.

        This method adds new columns to the left side of the occupancy grid, initializes them with unknown values (-1),
        and adjusts the grid origin accordingly.
    """
    def expand_grid_left(self):
        new_cols = np.full((self.height, 50), -1)
        self.occupancy_grid = np.hstack((new_cols, self.occupancy_grid))
        self.width += 50
        self.origin[0] -= 50 * self.resolution
        self.get_logger().info('Expanded grid to the left')

    """
        Expands the occupancy grid to the right.

        This method adds new columns to the right side of the occupancy grid, initializing them with unknown values (-1).
    """
    def expand_grid_right(self):
        new_cols = np.full((self.height, 50), -1)
        self.occupancy_grid = np.hstack((self.occupancy_grid, new_cols))
        self.width += 50
        self.get_logger().info('Expanded grid to the right')

    """
        Expands the occupancy grid upward.

        This method adds new rows to the top of the occupancy grid, initializing them with unknown values (-1).
    """
    def expand_grid_up(self):
        new_rows = np.full((50, self.width), -1)
        self.occupancy_grid = np.vstack((self.occupancy_grid, new_rows))
        self.height += 50
        self.get_logger().info('Expanded grid upward')

    """
        Expands the occupancy grid downward.

        This method adds new rows to the bottom of the occupancy grid, initializes them with unknown values (-1),
        and adjusts the grid origin accordingly.
    """
    def expand_grid_down(self):
        new_rows = np.full((50, self.width), -1)
        self.occupancy_grid = np.vstack((new_rows, self.occupancy_grid))
        self.height += 50
        self.origin[1] -= 50 * self.resolution
        self.get_logger().info('Expanded grid downward')

    """
        Publishes the current occupancy grid as a ROS message.

        This method converts the occupancy grid to a ROS OccupancyGrid message and publishes it to the "/map" topic.
    """
    def publish_map(self):
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = 'map'
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.width
        map_msg.info.height = self.height
        map_msg.info.origin.position.x = float(self.origin[0])
        map_msg.info.origin.position.y = float(self.origin[1])
        map_msg.info.origin.position.z = 0.0
        map_msg.data = np.ravel(self.occupancy_grid).astype(np.int8).tolist()
        self.map_pub.publish(map_msg)
        self.get_logger().info('Map published')

    """
        Broadcasts the dynamic transform from "odom" to "base_link".

        This method creates and broadcasts a dynamic transform from the "odom" frame to the "base_link" frame
        based on the current robot pose.
    """
    def broadcast_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation = Vector3(x=self.pose[0], y=self.pose[1], z=0.0)
        q = R.from_euler('xyz', [0, 0, self.pose[2]]).as_quat()
        t.transform.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.tf_broadcaster.sendTransform(t)
        self.get_logger().info('Transform from odom to base_link broadcasted')

"""
    Initializes and runs the StaticMapNode.

    Args:
        args (list, optional): Command line arguments passed to the node. Defaults to None.

    This function initializes the ROS2 system, creates an instance of the StaticMapNode, and spins the node
    to keep it active. After the node is terminated, it cleans up the resources.
"""
def main(args=None):
    rclpy.init(args=args)
    node = StaticMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
