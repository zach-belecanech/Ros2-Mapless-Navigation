
import math
import random
import time
import numpy as np
import gym
from gym import spaces
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Quaternion, Pose
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
from std_srvs.srv import Empty
from gym.envs.registration import register
from gazebo_msgs.srv import SpawnEntity, GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
import tensorflow as tf
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
import threading

class FinalGymEnv(gym.Env):
    def __init__(self, namespace='my_bot', center_pos=(0,0)):
        rclpy.init(args=None)  # Initialize the ROS 2 Python client library
        self.node = rclpy.create_node('robot_gym_node')
        
        self.spawn_goal_marker((0.0,1.0))

        self.entity_client = self.node.create_client(GetEntityState, '/plug/get_entity_state')
        while not self.entity_client.wait_for_service(timeout_sec=1.0):
            print('Service /plug/get_entity_state not available, waiting again...')

        self.set_entity_client = self.node.create_client(SetEntityState, '/plug/set_entity_state')
        while not self.set_entity_client.wait_for_service(timeout_sec=1.0):
            print('Service /plug/get_entity_state not available, waiting again...')

        self.center = center_pos
        self.namespace = namespace
        self.boxes = []
        
        num_lidar_points = 30
        self.observation_space = spaces.Box(
            low=np.array([0.0]*num_lidar_points + [0.0, 0.0, 0.0]),  # Low bounds for each observation
            high=np.array([1.0]*num_lidar_points + [1.0, 1.0, 1.0]),  # High bounds for each observation
            dtype=np.float32
        )


        self.action_space = spaces.Discrete(3)

        self.prev_goal_distance = 0.0
        self.latest_scan = None
        self.latest_odom = None
        self.goal_distance = 0.0
        self.goal_angle_cos = 0.0
        self.goal_angle_sin = 0.0
        self.angular_vel = 0.0
        self.prev_angular_velocity = 0.0
        self.start_distance = self.goal_distance
        
        self.goal_position = self.generate_goal_position()
        
        # ROS Publishers and Subscribers with namespace
        # Create ROS 2 publishers
        qos = QoSProfile(depth=10)  # Customize QoS as needed for your application
        self.cmd_vel_publisher = self.node.create_publisher(Twist, '/cmd_vel', qos)

        # Create ROS 2 subscribers
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, qos)
        self.odom_flag = False
        self.scan_flag = False
        self.reset()
    
        
    def box_positions(self):
        # Create a service client
        temp = []
        for i in range(10):  # Assuming you have 10 boxes as mentioned
            model_name = f'box{i}_robot1'
            try:
                # Prepare the request
                request = GetEntityState.Request()
                request.name = model_name
                request.reference_frame= 'world'
                
                # Call the service and get the response
                future = self.entity_client.call_async(request)
                rclpy.spin_until_future_complete(self.node, future)
                response = future.result()
                
                if response.success:
                    # Store pose information in the dictionary
                    temp.append(response.state.pose)
                else:
                    self.node.get_logger().warn(f"Failed to get model state for {model_name}")
            except Exception as e:
                self.node.get_logger().error(f"Service call failed: {e}")

        self.boxes = temp

    def spawn_goal_marker(self, position):
        # Ensure the service is available
        client = self.node.create_client(SpawnEntity, '/spawn_entity')
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('/spawn_entity service not available, waiting again...')

        try:
            with open("/home/easz/catkin_ws/src/testing_pkg/urdf/marker.sdf", 'r') as file:
                goal_sdf = file.read()

            pose = Pose()
            pose.position.x = position[0]
            pose.position.y = position[1]
            pose.position.z = 0.01  # slightly above the ground to ensure visibility
            print(pose)
            # Prepare the request
            request = SpawnEntity.Request()
            request.name = "goal_marker"
            request.xml = goal_sdf
            request.robot_namespace = ""
            request.initial_pose = pose
            request.reference_frame = "world"

            # Call the service
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            if future.result() is not None:
                self.node.get_logger().info('Goal marker spawned successfully.')
            else:
                self.node.get_logger().error('Failed to call spawn model service.')
        except Exception as e:
            self.node.get_logger().error('Model spawn service call failed: %s' % str(e))
      
    
    def generate_goal_position(self):
        self.box_positions()
        
        max_attempts = 1000
        for _ in range(max_attempts):
            #print('Generating a goal position...')
            # Generate a random angle and radius within 4.5 meters
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, 2.0)
            goal_x = self.center[0] + radius * math.cos(angle)
            goal_y = self.center[1] + radius * math.sin(angle)
            goal_position = (goal_x, goal_y)

            # Check that the goal position is not within 0.2 meters of any box
            if all(self.distance(goal_position, (box.position.x, box.position.y)) >= 0.2 for box in self.boxes):
                print("Valid goal position found: ", goal_position)
                self.goal_position = goal_position
                self.move_goal_marker(goal_position)
                return goal_position

        return None  # Return None if no valid position is found after max_attempts


    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def scan_callback(self, data):
        # Preprocess laser scan data to match the paper's setup
        scan_ranges = np.array(data.ranges)
        num_bins = 30
        bin_size = len(scan_ranges) // num_bins
        binned_ranges = [min(scan_ranges[i:i + bin_size]) for i in range(0, len(scan_ranges), bin_size)]
        self.latest_scan = np.array(binned_ranges) / 6.5  # Normalize distances
        self.scan_flag = True
        
    
    # checked and matched to scale
    def odom_callback(self, msg):
        # Update robot's position and compute goal distance and angle
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        self.angular_vel = msg.twist.twist.angular.z
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        self.latest_odom = np.array([position.x, position.y, yaw])
        
        goal_x, goal_y = self.goal_position
        
        robot_x, robot_y = position.x, position.y
        
        self.goal_distance = np.sqrt((goal_x - robot_x)**2 + (goal_y - robot_y)**2) / 7 # Normalize distance
        goal_angle = math.atan2(goal_y - robot_y, goal_x - robot_x) - yaw
        self.goal_angle_cos = (np.cos(goal_angle) + 1) / 2 # Normalize angles
        self.goal_angle_sin = (np.sin(goal_angle) + 1) / 2 # Normalize angles
        self.odom_flag = True

    def step(self, action):
        # Assume action is already a numpy array (conversion handled outside this function)
        #tf.print("action", action)
        action_space_velocities = [(0.0, 1.0), (-0.50, 0.0), (0.50, 0.0)]
        cmd = Twist()
        cmd.linear.x = action_space_velocities[action][1]  # Set linear velocity
        cmd.angular.z = action_space_velocities[action][0]  # Set angular velocity based on the action
        
        self.cmd_vel_publisher.publish(cmd)
          # Simulate delay for action execution
        time.sleep(0.1)
        done = False

        # print('Normalized Scan Values: ', self.latest_scan)
        # print('Normlaized distance values: ', self.goal_distance)
        # print('Normalized goal angle value cosine: ', self.goal_angle_cos)
        # print('Normalized goal angle value sine: ', self.goal_angle_sin)
        # if np.isinf(self.latest_scan).any():
        #     print("INF IS IN LIDAR DISTANCE ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        #     print("Values before change: ", self.latest_scan)
        #     # Replace infinite values with 1.0
        #     self.latest_scan[np.isinf(self.latest_scan)] = 1.0
        #     print("Values after change: ", self.latest_scan)
        #     print('Infinite values have been replaced.')

        # if np.isinf(self.goal_distance).any():
        #     print("INF IS IN GOAL DISTANCE ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        # if np.isinf(self.goal_angle_cos).any():
        #     print("INF IS IN GOAL ANGLE COS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

        # if np.isinf(self.goal_angle_sin).any():
        #     print("INF IS IN GOAL ANGLE SIN ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        while self.latest_scan is None or self.latest_odom is None:
            rclpy.spin_once(self.node)


        while True:
            rclpy.spin_once(self.node)
            if self.odom_flag and self.scan_flag:
                break
        self.odom_flag = False
        self.scan_flag = False
        print('Distance: ', self.goal_distance)
        full_state = np.concatenate((self.latest_scan, [self.goal_distance, self.goal_angle_cos, self.goal_angle_sin]))
        reward, done = self.compute_reward_and_done(done)
        return np.array(full_state, dtype=np.float32), reward, done


    def compute_reward_and_done(self, done):
        collision_penalty = -200
        goal_reached_reward = 200
        angular_velocity_threshold = 25.0
          # Penalty applied for each oscillation detection
        minimal_progress_penalty = -1  # Penalty if progress is below a threshold

        adjusted_goal_angle_cos = 2 * self.goal_angle_cos - 1

        if np.any(self.latest_scan < 0.015):
            return collision_penalty, True

        if self.goal_distance < 0.03:
            return goal_reached_reward, True

        # Calculate the distance change and apply a progress reward or penalty
        distance_change = (self.prev_goal_distance - self.goal_distance) / self.prev_goal_distance
        progress_reward = 100 * max(0, distance_change) if adjusted_goal_angle_cos > -0.5 else -1

        # Detect and penalize oscillations based on angular velocity changes

        # Apply angular velocity penalty if above the threshold
        angular_velocity_penalty = -0.5 * max(0, abs(self.angular_vel) - angular_velocity_threshold)

        # Apply a penalty if the progress is minimal over multiple steps
        if distance_change < 0.01:  # Threshold for significant progress
            progress_reward += minimal_progress_penalty

        total_reward = progress_reward + angular_velocity_penalty
        self.prev_goal_distance = self.goal_distance
        self.prev_angular_velocity = self.angular_vel
    

        return total_reward, done

        
        

    def reset(self):
        # Reset step counter
        # print("In Robot env: " + str(self.environments))
        # Reset goal position
        # Reset simulation and robot state
        # self.reset_simulation()  # Resets the simulation
        self.reset_environment(self.namespace) # Resets the world to its initial state
        time.sleep(0.1)
        # Get initial observation
        #print('Got goal')
        self.goal_position = self.generate_goal_position()
       #self.move_goal_marker(self.goal_position)
        initial_observation = self.get_initial_observation()
        #print('got observation')
        
        return initial_observation

    def get_initial_observation(self):
        # Wait for fresh sensor data
        while self.latest_scan is None or self.latest_odom is None:
            time.sleep(0.1)  # Wait for callbacks to receive data

        self.start_distance = self.goal_distance
        self.prev_goal_distance = self.start_distance
        # Combine laser scan data and odometry for the initial observation
        initial_observation = np.concatenate([self.latest_scan, [self.goal_distance, self.goal_angle_cos, self.goal_angle_sin]])

        # Reset placeholders for the next reset cycle
        self.latest_scan = None
        self.latest_odom = None

        return initial_observation
    
    def get_namespace(self):
        return self.namespace
    

    def angle_to_quaternion(self,yaw):
        """Convert a yaw angle (in radians) to a quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return Quaternion(x=0.0, y=0.0, z=sy, w=cy)

    def reset_environment(self,namespace):
        
        center_x, center_y = self.center
        
        # Prepare new random positions for the robot and boxes relative to the center
        new_positions = []

        # Generate new position for the robot
        robot_new_x = random.uniform(center_x - 2, center_x + 2)
        robot_new_y = random.uniform(center_y - 2, center_y + 2)
        new_positions.append((robot_new_x, robot_new_y))

        # Generate new positions for the boxes
        for _ in range(0,10):
            box_new_x = random.uniform(center_x - 2, center_x + 2)
            box_new_y = random.uniform(center_y - 2, center_y + 2)
            new_positions.append((box_new_x, box_new_y))

        # Check for overlaps
        if not self.positions_are_valid(new_positions):
            return self.reset_environment(namespace)  # Recursively retry until a valid configuration is found

        quat = self.angle_to_quaternion(random.uniform(0, 2 * math.pi))
        quat_x = float(quat.x)
        quat_y = float(quat.y)
        quat_z = float(quat.z)
        quat_w = float(quat.w)
        # If no overlaps, proceed to update positions
        robot_state = EntityState()
        robot_state.name = self.namespace
        robot_state.pose.position.x = new_positions[0][0]
        robot_state.pose.position.y = new_positions[0][1]
        robot_state.pose.position.z = 0.06  # Adjust height if necessary
        robot_state.pose.orientation.x = quat_x
        robot_state.pose.orientation.y = quat_y
        robot_state.pose.orientation.z = quat_z
        robot_state.pose.orientation.w = quat_w

        robot_state_final = SetEntityState.Request()
        robot_state_final._state = robot_state

        # Call the service
        future = self.set_entity_client.call_async(robot_state_final)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if not response.success:
            self.get_logger().error(f"Failed to reset robot position for {self.namespace}")
            return False

        # Update positions of boxes
        for idx in range(1, 11):
            box_state = EntityState()
            box_state.name = f'box{idx-1}_robot1'
            box_state.pose.position.x = new_positions[idx][0]
            box_state.pose.position.y = new_positions[idx][1]
            box_state.pose.position.z = 0.0  # Ground level

            box_state_final = SetEntityState.Request()
            box_state_final._state = box_state

            future = self.set_entity_client.call_async(box_state_final)
            rclpy.spin_until_future_complete(self.node, future)
            response = future.result()
            if not response.success:
                self.get_logger().error(f"Failed to reset box position for box{idx-1}_robot1")
                return False

        return True

    def move_goal_marker(self, position):
        # Move the existing marker to a new position
        # If no overlaps, proceed to update positions
        goal_state = EntityState()
        goal_state.name = "goal_marker"
        goal_state.pose.position.x = position[0]
        goal_state.pose.position.y = position[1]
        goal_state.pose.position.z = 0.01  # Adjust height if necessary
        goal_state.pose.orientation.x = 0.0
        goal_state.pose.orientation.y = 0.0
        goal_state.pose.orientation.z = 0.0
        goal_state.pose.orientation.w = 0.0

        goal_state_final = SetEntityState.Request()
        goal_state_final._state = goal_state

        # Call the service
        future = self.set_entity_client.call_async(goal_state_final)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if not response.success:
            self.get_logger().error(f"Failed to reset goal position for {self.namespace}")
            return False 
        else:
            print('Should have moved')

    def positions_are_valid(self, positions):
        """Check if there are any overlapping positions."""
        # You can define a threshold distance below which two objects are considered to be overlapping
        threshold = 0.7  # meters
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    if ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5 < threshold:
                        return False
        return True
    


register(
    id='FinalGym',
    entry_point='final_test_env:FinalGymEnv',
)