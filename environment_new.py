#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import math
from math import pi
import random
import time
import threading
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

DIAGONAL_LEN = math.sqrt(2) * (5 + 5)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', '..', '..', 'turtlebot3_gazebo', 'share', 'turtlebot3_gazebo', 'models', 'goal_target', 'model.sdf')

LIDAR_SEGMENT_STEP_SIZE = 36
MAX_LIDAR_DIST = 2.0
COLLISION_RANGE = 0.12

class Env(Node):
    def __init__(self, is_training):
        super().__init__('env_node')
        
        # Create a reentrant callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        self.start_time = None
        # Initialize variables
        self.position = Point()
        self.goal_position = Pose()
        self.goal_position.position.x = 2.0
        self.goal_position.position.y = 0.0
        self.yaw = 0.
        self.rel_theta = 0.
        self.diff_angle = 0.
        self.scan = []
        self.scan_data = []
        self.past_distance = 0.
        self.collision_threshold = 0
        self.threshold_arrive = 0.2 if is_training else 0.4
        
        # Create locks for thread-safe data access
        self.scan_lock = threading.Lock()
        self.odom_lock = threading.Lock()
        
        # Initialize publishers and subscribers with callback group
        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.sub_odom = self.create_subscription(
            Odometry, 
            'odom', 
            self.getOdometry, 
            10,
            callback_group=self.callback_group
        )
        self.sub_scan = self.create_subscription(
            LaserScan, 
            'scan', 
            self.getScan, 
            10,
            callback_group=self.callback_group
        )
        
        # Initialize service clients
        self.reset_proxy = self.create_client(Empty, '/reset_simulation')
        self.unpause_proxy = self.create_client(Empty, '/unpause_physics')
        self.pause_proxy = self.create_client(Empty, '/pause_physics')
        self.goal = self.create_client(SpawnEntity, '/spawn_entity')
        self.del_model = self.create_client(DeleteEntity, '/delete_entity')
        
        # Start the ROS 2 spinner in a separate thread
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()

    def _spin(self):
        """Separate thread for running the ROS 2 executor"""
        executor = MultiThreadedExecutor()
        executor.add_node(self)
        executor.spin()
        
    def getScan(self, scan_msg):
        """Thread-safe scan callback"""
        with self.scan_lock:
            # print("Scan received")
            self.scan = scan_msg.ranges

    def getOdometry(self, odom):
        """Thread-safe odometry callback"""
        with self.odom_lock:
            # print("Odom received")
            self.position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
            yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 
                                              1 - 2 * (q_y * q_y + q_z * q_z))))

            if yaw >= 0:
                yaw = yaw
            else:
                yaw = yaw + 360

            rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
            rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

            # Calculate the angle between robot and target
            theta = self._calculate_theta(rel_dis_x, rel_dis_y)
            rel_theta = round(math.degrees(theta), 2)
            
            diff_angle = self._calculate_diff_angle(yaw, rel_theta)

            self.rel_theta = rel_theta
            self.yaw = yaw
            self.diff_angle = diff_angle

    def _calculate_theta(self, rel_dis_x, rel_dis_y):
        """Helper method to calculate theta angle"""
        if rel_dis_x > 0 and rel_dis_y > 0:
            return math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            return 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            return math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            return math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            return 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            return 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            return 0
        else:
            return math.pi

    def _calculate_diff_angle(self, yaw, rel_theta):
        """Helper method to calculate difference angle"""
        diff_angle = (yaw - rel_theta)
        if 0 <= diff_angle <= 180 or -180 <= diff_angle < 0:
            return round(diff_angle, 2)
        elif diff_angle < -180:
            return round(360 + diff_angle, 2)
        else:
            return round(-360 + diff_angle, 2)
        
    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance
    
    def getState(self, scan):
        """Thread-safe state getter"""
        with self.odom_lock:
            yaw = self.yaw
            rel_theta = self.rel_theta
            diff_angle = self.diff_angle
            current_position = Point()
            current_position.x = self.position.x
            current_position.y = self.position.y

        min_range = COLLISION_RANGE
        done = False
        arrive = False

        scan_range = self._process_scan_data(scan)

        if min_range > min(scan_range) > 0:
            self.collision_threshold += 1
            if self.collision_threshold > 2:
                done = True
                self.collision_threshold = 0

        current_distance = math.hypot(self.goal_position.position.x - current_position.x,
                                    self.goal_position.position.y - current_position.y)
        if current_distance <= self.threshold_arrive:
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def _process_scan_data(self, scan):
        """Helper method to process scan data"""
        scan_range = []
        for i in range(len(scan)):
            if scan[i] == float('Inf'):
                scan_range.append(MAX_LIDAR_DIST)
            elif np.isnan(scan[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan[i])
        return scan_range

    def step(self, action, past_action):
        """Execute one step in the environment"""
        # Publish velocity command
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.pub_cmd_vel.publish(vel_cmd)

        # Get current scan data
        with self.scan_lock:
            data = self.scan

        if data:
            short_data = self.Pick(data, LIDAR_SEGMENT_STEP_SIZE)
            state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(short_data)
            
            # Process state
            state = [i / MAX_LIDAR_DIST for i in state]
            for pa in past_action:
                state.append(pa)
            state = state + [rel_dis / DIAGONAL_LEN, yaw / 360, rel_theta / 360, diff_angle / 180]
            
            self.scan_data = short_data
            reward = self.setReward(done, arrive)
                
            return np.asarray(state), reward, done, arrive
        else:
            print("Gone")
            return None, 0, False, False

    def Pick(self, state, len_batch):
        """Process raw lidar data into batches"""
        item = []
        new_state = []
        if len(state) == 0:
            new_state = [MAX_LIDAR_DIST] * 10
            return new_state
            
        for elem in state:
            item.append(elem)
            if len(item) == len_batch:
                new_state.append(np.min(np.asarray(item)))
                item = []
        if len(item) != 0:
            new_state.append(np.min(np.asarray(item)))
        return new_state

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, 
                                    self.goal_position.position.y - self.position.y)
        distance_rate = self.past_distance - current_distance

        # Reward for moving closer to the goal
        scaling_factor = 40
        if distance_rate > 0:
            distance_reward = distance_rate * scaling_factor
        else:
            distance_reward = distance_rate * scaling_factor * 2  # Stronger penalty for moving away

        # Calculate heading angle reward
        goal_angle = math.degrees(math.atan2(self.goal_position.position.y - self.position.y, 
                                            self.goal_position.position.x - self.position.x))
        heading_error = abs((goal_angle - self.yaw + 180) % 360 - 180)
        heading_error_normalized = heading_error / 180
        heading_reward = 1 - heading_error_normalized

        # Combine rewards with dynamic weights
        distance_weight = 0.7
        heading_weight = 1.0 - distance_weight
        reward = distance_weight * distance_reward + heading_weight * heading_reward

        self.past_distance = current_distance

        # Penalty for collision
        if done:
            print("Collision!!")
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        # Reward for reaching the goal
        if arrive:
            reward = 150.
            self.pub_cmd_vel.publish(Twist())
            print("Arrive!!")

        return reward

    def is_in_obstacle(self, x, y, obstacle_regions):
        """Check if a point (x, y) lies within any obstacle region."""
        for (x_min, x_max, y_min, y_max) in obstacle_regions:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return True
        return False

    def reset(self):
        self.start_time = time.time()
        self.del_model.call_async(DeleteEntity.Request(name='goal_target'))
        self.pause_proxy.call_async(Empty.Request())
        time.sleep(2)
        self.reset_proxy.call_async(Empty.Request())
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnEntity.Request()
            target.name = 'goal_target'
            target.xml = goal_urdf
            obstacle_regions = [
                (-0.5, 0.5, -0.5, 0.5),                 # Obstacle at (0, 0)
                (0.75, 1.75, 0.75, 1.75),               # Obstacle at (1.25, 1.25)
                (0.75, 1.75, -1.75, -0.75),             # Obstacle at (1.25, -1.25)
                (-1.75, -0.75, 0.75, 1.75),             # Obstacle at (-1.25, 1.25)
                (-1.75, -0.75, -1.75, -0.75),           # Obstacle at (-1.25, -1.25)
                (-2.5, -1.0, -1.0, 1.0),                # near start
            ]
            # Sampling loop to ensure the goal is not in an obstacle region
            while True:
                # x = np.random.uniform(-2.0, 2.0)
                # y = np.random.uniform(-2.0, 2.0)
                x = 1.8
                y = 0.0
                
                if not self.is_in_obstacle(x, y, obstacle_regions):
                    # Valid goal found
                    self.goal_position.position.x = x
                    self.goal_position.position.y = y
                    break
            target.initial_pose = self.goal_position
            self.goal.call_async(target)
        except Exception as e:
            self.get_logger().error("/gazebo/failed to build the target")
        self.unpause_proxy.call_async(Empty.Request())
        
        data = None
        while data is None:
            try:
                data = self.scan
                short_data = self.Pick(data, LIDAR_SEGMENT_STEP_SIZE)
            except:
                pass

        time.sleep(2.0)
        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(short_data)
        state = [i / MAX_LIDAR_DIST for i in state]
        state.append(0) # past linear velocity
        state.append(0) # past angular velocity
        state = state + [rel_dis / DIAGONAL_LEN, yaw / 360, rel_theta / 360, diff_angle / 180]
        length = len(state)

        return np.asarray(state)
