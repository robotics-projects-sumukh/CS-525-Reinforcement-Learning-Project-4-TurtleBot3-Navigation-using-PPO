import torch
from torch.distributions import MultivariateNormal
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import json
import os
import numpy as np
from tf_transformations import euler_from_quaternion
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_VELOCITY = 1.0
MIN_VELOCITY = 0.0
MAX_ANGULAR = 1.0
MIN_ANGULAR = -1.0
act_dim = 2

class OdomSubscriber(Node):
    """
    ROS2 Node to subscribe to odometry messages.
    """
    def __init__(self):
        super().__init__('odom_subscriber')
        self.current_position = None
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

    def odom_callback(self, msg):
        # Extract the current position from the odometry message
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
		# Convert quaternion to Euler angles
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = [position.x, position.y, yaw]
        
		

def rollout(policy, env, max_episodes=100):
    """
    Rollout episodes with the given policy and environment.

    Parameters:
        policy - The trained policy
        env - The environment
        max_episodes - Number of episodes to run (default: 100)
    
    Returns:
        analysis_data - List of dictionaries containing episode details
    """
    # Check if rclpy is already initialized
    if not rclpy.ok():
        rclpy.init()
        
    odom_node = OdomSubscriber()
    analysis_data = []

    for ep_num in range(max_episodes):
        obs = env.reset()
        done = False
        arrive = False
        t = 0

        # Episode metrics
        ep_data = {
            "episode": ep_num,
            "path": [],            # Positions visited
            "collision": False,   # Collision occurred
            "goal_reached": False, # Did the robot reach the goal?
            "total_reward": 0,     # Total reward
            "length": 0            # Episode length
        }

        past_action = [0., 0.]
        while not done and not arrive:
            # wait for sometime so it doesnt use too much CPU
            # time.sleep(0.05)

            t += 1

            # Spin the ROS2 node to update position
            rclpy.spin_once(odom_node, timeout_sec=0.1)

            # Collect path from /odom
            if odom_node.current_position is not None:
                ep_data["path"].append(odom_node.current_position)

            # Get action and perform a step
            action, _ = get_action(policy, obs)
            obs, reward, done, arrive = env.step(action, past_action)

            # Update collision to True if done is True
            if done:
                ep_data["collision"] = True

            # Update arrive status
            if arrive:
                ep_data["goal_reached"] = True

            ep_data["total_reward"] += reward
            past_action = action

        ep_data["length"] = t
        analysis_data.append(ep_data)

        print(f"Episode {ep_num+1}/{max_episodes}: Length={t}, Reward={ep_data['total_reward']}, "
              f"Collision={ep_data['collision']}, Goal Reached={ep_data['goal_reached']}")

    # Shutdown rclpy only if it was initialized here
    if rclpy.ok():
        rclpy.shutdown()
        
    return analysis_data

def get_action(policy, obs):
    """
    Queries an action from the actor network.

    Parameters:
        obs - Observation at the current timestep

    Returns:
        action - The sampled action
        log_prob - Log probability of the selected action
    """
    mean = policy(obs).to(device)
    cov_var = torch.full(size=(act_dim,), fill_value=0.01).to(device)
    cov_mat = torch.diag(cov_var).to(device)
    dist = MultivariateNormal(mean, cov_mat)
    
    action = dist.sample()
    action[0] = torch.clip(action[0], MIN_VELOCITY, MAX_VELOCITY)
    action[1] = torch.clip(action[1], MIN_ANGULAR, MAX_ANGULAR)
    log_prob = dist.log_prob(action)

    return action.detach().cpu().numpy(), log_prob.detach()


def save_analysis_data(analysis_data, file_path="test.log"):
    """
    Save analysis data to a file.

    Parameters:
        analysis_data - The data to save
        file_path - Path to the log file
    """
    file_path = os.path.join("src/turtlebot3/turtlebot3_teleop/turtlebot3_teleop/script/models", file_path)

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Create the file and write data
    try:
        with open(file_path, "w") as f:
            for episode in analysis_data:
                f.write(str(episode) + "\n")
        print(f"Analysis data saved to {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"Error saving analysis data: {e}")


def eval_policy(policy, env, output_file="test.log"):
    """
    Evaluates a policy over 100 episodes and saves analysis.

    Parameters:
        policy - The trained policy
        env - The environment
        output_file - Path to the output file to save results (default: 'analysis.log')
    """

    start_time = time.time()

    max_episodes = 1
    analysis_data = rollout(policy, env, max_episodes)

    # Save analysis data to a .log file
    save_analysis_data(analysis_data, output_file)
    
    end_time = time.time()
    
    print(f"Time taken to evaluate policy: {end_time - start_time:.2f} seconds")