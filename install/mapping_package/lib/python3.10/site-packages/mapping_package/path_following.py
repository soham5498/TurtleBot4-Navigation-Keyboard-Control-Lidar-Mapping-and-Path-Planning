import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import math

class PathFollowingNode(Node):
    def __init__(self):
        super().__init__('path_following_node')
        self.path_subscription = self.create_subscription(
            Path,
            'planned_path',
            self.path_callback,
            10)
        self.odom_subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.velocity_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.current_path = []
        self.current_position = None
        self.current_orientation = None

    def path_callback(self, msg):
        self.current_path = msg.poses
        self.follow_path()

    def odom_callback(self, msg):
        self.current_position = msg.pose.pose.position
        self.current_orientation = msg.pose.pose.orientation

    def follow_path(self):
        if not self.current_path or not self.current_position:
            return

        goal_position = self.current_path[0].pose.position

        # Simple proportional control
        distance = math.sqrt((goal_position.x - self.current_position.x) ** 2 +
                             (goal_position.y - self.current_position.y) ** 2)

        angle_to_goal = math.atan2(goal_position.y - self.current_position.y,
                                   goal_position.x - self.current_position.x)

        # Robot's current orientation in radians
        _, _, current_yaw = self.euler_from_quaternion(self.current_orientation)

        # Angle difference
        angle_diff = angle_to_goal - current_yaw

        # Create a Twist message and publish it
        twist_msg = Twist()
        twist_msg.linear.x = 0.5 * distance  # Proportional control for linear velocity
        twist_msg.angular.z = 1.0 * angle_diff  # Proportional control for angular velocity

        self.velocity_publisher.publish(twist_msg)

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (geometry_msgs.msg.Quaternion) to Euler angles (roll, pitch, yaw)
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = PathFollowingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

