import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import heapq

class PathPlanningNode(Node):
    def __init__(self):
        super().__init__('path_planning_node')
        self.subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
        self.path_publisher = self.create_publisher(Path, 'planned_path', 10)
        self.declare_parameter('goal', [10.0, 10.0])  # Example goal position
        self.map = None
        self.robot_position = None

    def map_callback(self, msg):
        self.map = msg
        self.plan_path()

    def plan_path(self):
        if self.map is None or self.robot_position is None:
            return

        goal = self.get_parameter('goal').get_parameter_value().double_array_value
        start = [self.robot_position.x, self.robot_position.y]

        path = self.a_star(start, goal)
        if path:
            self.publish_path(path)

    def a_star(self, start, goal):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def get_neighbors(node):
            neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            result = []
            for dx, dy in neighbors:
                x2, y2 = node[0] + dx, node[1] + dy
                if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] == 0:
                    result.append((x2, y2))
            return result

        width = self.map.info.width
        height = self.map.info.height
        grid = np.reshape(np.array(self.map.data), (height, width))

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        open_list = []
        heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_list:
            current = heapq.heappop(open_list)[2]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for neighbor in get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

        return []

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.poses = [self.create_pose_stamped(pt) for pt in path]
        self.path_publisher.publish(path_msg)

    def create_pose_stamped(self, pt):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position = Point(x=pt[0], y=pt[1], z=0.0)
        return pose_stamped

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

