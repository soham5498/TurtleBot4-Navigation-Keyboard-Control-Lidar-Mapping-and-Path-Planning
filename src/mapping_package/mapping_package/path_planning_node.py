# https://github.com/fazildgr8/ros_autonomous_slam/blob/master/nodes/autonomous_move.py
# https://github.com/fazildgr8/ros_autonomous_slam/blob/master/nodes/a_star_main.py

import rclpy
from rclpy.node import Node
import math
from math import cos, sin
import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Twist, TransformStamped, Vector3, Quaternion
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformListener, Buffer, TransformBroadcaster, StaticTransformBroadcaster
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from scipy.spatial.transform import Rotation as R

class NodeAstar:
    def __init__(self, position):
        self.location = position
        self.prev = None
        self.g_cost = 0
        self.h_cost = 0
        self.f_cost = 0

    def update_cost(self, g, h):
        self.g_cost = g
        self.h_cost = h
        self.f_cost = g + h

    def __eq__(self, node):
        return self.location == node.location

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')

        self.declare_parameter('goalx', 0.0)
        self.declare_parameter('goaly', 0.0)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.robot_location = [-8.0, -2.0, 0.0]
        self.robot_rotation = [0.0, 0.0, 0.0]
        self.global_map = None
        self.final_goal_location = [self.get_parameter('goalx').get_parameter_value().double_value,
                                    self.get_parameter('goaly').get_parameter_value().double_value,
                                    0.0]
        self.goal_reached = False
        self.final_path = None

        self.create_subscription(Odometry, '/odom', self.callback_odom, qos_profile)
        self.create_subscription(OccupancyGrid, '/map', self.callback_map, qos_profile)
        self.create_subscription(PoseStamped, '/goal', self.callback_goal, qos_profile)

        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', qos_profile)
        self.marker_publisher = self.create_publisher(Marker, 'path_points', qos_profile)

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # TF broadcasters
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        self.tf_dynamic_broadcaster = TransformBroadcaster(self)
        
        # Publish initial static transform
        self.publish_static_transform()

    def publish_static_transform(self):
        static_transform_stamped = TransformStamped()
        static_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        static_transform_stamped.header.frame_id = 'map'
        static_transform_stamped.child_frame_id = 'odom'
        static_transform_stamped.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        static_transform_stamped.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.tf_static_broadcaster.sendTransform(static_transform_stamped)
        self.get_logger().info('Static transform from map to odom published')

    def movebase_client(self, x, y):
        self.get_logger().info(f'Sending goal to navigate_to_pose: ({x}, {y})')
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.w = 1.0

        self.action_client.wait_for_server()
        self.action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback).add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback):
        self.get_logger().info('Received feedback from navigate_to_pose')

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected by navigate_to_pose')
            return

        self.get_logger().info('Goal accepted by navigate_to_pose')
        goal_handle.get_result_async().add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        if result:
            self.get_logger().info('Goal reached successfully')
        else:
            self.get_logger().error('Failed to reach the goal')

    def callback_odom(self, msg):
        self.robot_location = [float(msg.pose.pose.position.x), float(msg.pose.pose.position.y)]
        orientation = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        (roll, pitch, yaw) = self.quat_to_euler(orientation)
        self.robot_rotation = [float(roll), float(pitch), float(yaw)]
        self.publish_dynamic_transform(msg.pose.pose)

    def callback_map(self, msg):
        data = np.array(msg.data, dtype=np.float64)
        map_width = msg.info.width
        map_height = msg.info.height
        self.global_map = np.reshape(data, (map_height, map_width))

    def callback_goal(self, msg):
        self.final_goal_location = [float(msg.pose.position.x), float(msg.pose.position.y), 0.0]
        self.goal_reached = False
        self.final_path = None  # Reset path to trigger replanning
        self.movebase_client(msg.pose.position.x, msg.pose.position.y)

    def publish_dynamic_transform(self, pose):
        dynamic_transform_stamped = TransformStamped()
        dynamic_transform_stamped.header.stamp = self.get_clock().now().to_msg()
        dynamic_transform_stamped.header.frame_id = 'odom'
        dynamic_transform_stamped.child_frame_id = 'base_link'
        dynamic_transform_stamped.transform.translation = Vector3(x=float(pose.position.x), y=float(pose.position.y), z=float(pose.position.z))
        dynamic_transform_stamped.transform.rotation = Quaternion(
            x=float(pose.orientation.x), y=float(pose.orientation.y), z=float(pose.orientation.z), w=float(pose.orientation.w)
        )
        self.tf_dynamic_broadcaster.sendTransform(dynamic_transform_stamped)
        self.get_logger().info('Dynamic transform from odom to base_link published')

    def Distance_compute(self, pos1, pos2, Type='d'):
        x1, y1 = pos1[:2]
        x2, y2 = pos2[:2]
        d = ((x1 - x2) ** 2) + ((y1 - y2) ** 2)
        if Type == 'd':
            return math.sqrt(d)
        if Type == 'eu':
            return d
        if Type == 'manhattan':
            return abs(x1 - x2) + abs(y1 - y2)

    def rot2d(self, v, t):
        x, y = v[0], v[1]
        xr = x * cos(t) - y * sin(t)
        yr = x * sin(t) + y * cos(t)
        return [xr, yr]

    def convert_path(self, path, trans, t):
        npath = []
        for x in path:
            mat = [x[0], x[1]]
            mat = self.rot2d(mat, t)
            npath.append((float(mat[0]) + float(trans[0]), float(mat[1]) + float(trans[1])))
        return npath

    def go_to_goal(self, goal):
        d = self.Distance_compute(self.robot_location, goal)
        theta = self.robot_rotation[2]
        kl = 1.0
        ka = 4.0
        vx = 0.0
        va = 0.0
        heading = math.atan2(goal[1] - self.robot_location[1], goal[0] - self.robot_location[0])
        err_theta = heading - theta
        if d > 0.01:
            vx = kl * abs(d)
            vx = 1.0
        if abs(err_theta) > 0.01:
            va = ka * (err_theta)

        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.angular.z = float(va)
        self.vel_publisher.publish(cmd)

    def Follow_path(self, path):
        cpath = path
        goal_point = cpath[-1]
        for loc in cpath:
            while self.Distance_compute(self.robot_location, loc) > 0.1:
                self.go_to_goal([loc[0] / 10, loc[1] / 10])
                if loc == goal_point:
                    self.goal_reached = True

    def points_publisher(self, points_list):
        marker_data = Marker()
        marker_data.type = Marker.POINTS
        marker_data.action = Marker.ADD
        marker_data.header.frame_id = 'map'
        marker_data.scale.x = 0.1  # width
        marker_data.scale.y = 0.1  # height
        marker_data.color.a = 1.0
        marker_data.color.r = 1.0
        marker_data.color.g = 0.0
        marker_data.color.b = 0.0

        for p in points_list:
            marker_data.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.0))
        self.marker_publisher.publish(marker_data)

    def A_STAR(self, global_map, start, end, Type='8c', e=1, heuristic='eu'):
        self.get_logger().info('Generating New Path')
        start_node = NodeAstar(start)  # Initiate Start Node
        end_node = NodeAstar(end)  # Initiate End Node

        # Check if Start and End are inside the Map bound
        if (start_node.location[1] > global_map.shape[0] - 1 or start_node.location[1] < 0 or
                start_node.location[0] > global_map.shape[1] - 1 or start_node.location[0] < 0):
            self.get_logger().error('[ERROR] Start Location out of bound')
            return None
        if (end_node.location[1] > global_map.shape[0] - 1 or end_node.location[1] < 0 or
                end_node.location[0] > global_map.shape[1] - 1 or end_node.location[0] < 0):
            self.get_logger().error('[ERROR] End Location out of bound')
            return None

        open_list = [start_node]  # Initiate Open List
        closed_nodes = []  # Initiate Closed List
        iterations = 0

        while open_list:
            current_node = open_list[0]
            index = 0
            iterations += 1

            # Try Algo with 4 Neighbors if iterations exceed 4k
            if iterations > 4000 and Type == '8c':
                self.get_logger().info('Trying with 4 Neighbors')
                return None

            # Check if current Node cost is lowest
            for i, x in enumerate(open_list):
                if x.f_cost < current_node.f_cost:
                    current_node = x
                    index = i
            open_list.pop(index)
            closed_nodes.append(current_node)

            # Check if goal node is reached
            if current_node == end_node:
                path = []
                node = current_node
                while node is not None:
                    path.append(node.location)
                    node = node.prev
                # Return Back tracked path from end to goal
                return path[::-1]

            neighbors = []
            # Index changes for 8 Neighbor method
            i_list = [0, 0, -1, 1, -1, -1, 1, 1]
            j_list = [-1, 1, 0, 0, -1, 1, -1, 1]
            # Index changes for 4 Neighbor method
            if Type == '4c':
                i_list = [0, 0, 1, -1]
                j_list = [1, -1, 0, 0]

            # Check for all Neighbors of Current Node
            for k in range(len(i_list)):
                node_pos = [current_node.location[0] + i_list[k], current_node.location[1] + j_list[k]]
                # Map Bound Check
                if (node_pos[1] > global_map.shape[0] - 1 or node_pos[1] < 0 or
                        node_pos[0] > global_map.shape[1] - 1 or node_pos[0] < 0):
                    continue
                # Occupation in Grid Check
                if global_map[node_pos[1]][node_pos[0]] == 1:
                    continue

                # Diagonal/Corner Cutting Check
                try:
                    if (abs(self.Distance_compute([node_pos[0], node_pos[1]], current_node.location, 'd') - 1.4143) < 0.001 and
                            global_map[current_node.location[1] + 1][current_node.location[0]] == 1 and
                            global_map[current_node.location[1]][current_node.location[0] + 1] == 1):
                        continue
                    if (abs(self.Distance_compute([node_pos[0], node_pos[1]], current_node.location, 'd') - 1.4143) < 0.001 and
                            global_map[current_node.location[1] - 1][current_node.location[0]] == 1 and
                            global_map[current_node.location[1]][current_node.location[0] - 1] == 1):
                        continue
                except IndexError:
                    continue

                neighbor_node = NodeAstar((node_pos[0], node_pos[1]))
                neighbor_node.prev = current_node  # Update Neighbor Node Parent
                neighbors.append(neighbor_node)  # Add to other Neighbors

            for neighbor in neighbors:
                if neighbor in closed_nodes:
                    continue

                # Avoid Locations Cornering with obstacles
                comb_nw = [(-1, 0), (-1, 1), (0, 1)]
                comb_ne = [(1, 1), (1, 0), (0, 1)]
                comb_sw = [(-1, 0), (-1, -1), (0, -1)]
                comb_se = [(1, -1), (1, 0), (0, -1)]
                n_fac = 0
                nw_flag = ne_flag = sw_flag = se_flag = True
                try:
                    for xp, yp in comb_nw:
                        if global_map[neighbor.location[1] + yp][neighbor.location[0] + xp] != 1:
                            nw_flag = False
                    for xp, yp in comb_ne:
                        if global_map[neighbor.location[1] + yp][neighbor.location[0] + xp] != 1:
                            ne_flag = False
                    for xp, yp in comb_se:
                        if global_map[neighbor.location[1] + yp][neighbor.location[0] + xp] != 1:
                            se_flag = False
                    for xp, yp in comb_sw:
                        if global_map[neighbor.location[1] + yp][neighbor.location[0] + xp] != 1:
                            sw_flag = False
                    if nw_flag:
                        n_fac += 3
                    if ne_flag:
                        n_fac += 3
                    if se_flag:
                        n_fac += 3
                    if sw_flag:
                        n_fac += 3
                except IndexError:
                    pass

                # Update Costs of the Neighbor Nodes
                g = neighbor.g_cost + 1 + (n_fac ** 2) * 10
                h = self.Distance_compute(neighbor.location, end_node.location, heuristic)
                neighbor.update_cost(g, h * e)
                for onode in open_list:
                    if neighbor == onode and neighbor.g_cost > onode.g_cost:
                        continue
                open_list.append(neighbor)

    def timer_callback(self):
        if self.global_map is None:
            return

        start = (int(self.robot_location[0] + 9), int(self.robot_location[1] + 10))
        end = (int(self.final_goal_location[0] + 9), int(self.final_goal_location[1] + 10))

        if not self.final_path:
            neighbor_type = '8c'
            heuristic = 'eu'
            heuristic_factor = 2

            self.final_path = self.A_STAR(self.global_map[::-1], start, end, neighbor_type, heuristic_factor, heuristic)
            if not self.final_path:
                neighbor_type = '4c'
                self.final_path = self.A_STAR(self.global_map[::-1], start, end, neighbor_type, heuristic_factor, heuristic)
            self.path_odom_frame = self.convert_path(self.final_path, [-9, -10], 0)
            self.goal_reached = False

        new_goal = [self.get_parameter('goalx').get_parameter_value().double_value,
                    self.get_parameter('goaly').get_parameter_value().double_value, 0.0]

        if self.final_goal_location != new_goal:
            self.final_goal_location = new_goal
            start = (int(self.robot_location[0] + 9), int(self.robot_location[1] + 10))
            end = (int(self.final_goal_location[0] + 9), int(self.final_goal_location[1] + 10))
            self.goal_reached = False
            self.final_path = self.A_STAR(self.global_map[::-1], start, end, neighbor_type, heuristic_factor, heuristic)
            self.path_odom_frame = self.convert_path(self.final_path, [-9, -10], 0)

        self.points_publisher(self.path_odom_frame)

        if not self.goal_reached:
            self.Follow_path(self.path_odom_frame)
        elif self.goal_reached:
            self.get_logger().info('Goal Reached')

    def quat_to_euler(self, quat):
        return R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_euler('xyz')


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

