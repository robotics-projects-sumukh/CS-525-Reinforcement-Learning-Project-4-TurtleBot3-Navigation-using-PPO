#!/usr/bin/env python3

from __future__ import absolute_import, print_function
import gym
import sys
import torch
import rclpy  # ROS 2 client library
from rclpy.node import Node
from ppo import PPO
from net_actor import NetActor
from net_critic import NetCritic
from eval_policy import eval_policy
import numpy as np
from environment_new import Env
import os, glob

state_dim = 16
action_dim = 2
action_linear_max = 1.0  # m/s
action_angular_max = 1.0  # rad/s
is_training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class MAIN(Node):
    def __init__(self):
        super().__init__('main')
        self.env = Env(is_training)

        print('State Dimensions:', state_dim)
        print('Action Dimensions:', action_dim)
        print('Action Max:', f'{action_linear_max} m/s and {action_angular_max} rad/s')

        self.test()

    def test(self):
        actor_model = f'src/turtlebot3/turtlebot3_teleop/turtlebot3_teleop/script/models/ppo_actor.pth'
        print(f"Testing {actor_model}", flush=True)

        if not actor_model:
            print("No model file specified. Exiting.", flush=True)
            sys.exit(0)

        policy = NetActor(state_dim, action_dim).to(device)
        policy.load_state_dict(torch.load(actor_model))

        eval_policy(policy=policy, env=self.env, output_file="test.log")
        exit(0)


def makepath(desired_path, isfile=False):
    if isfile:
        os.makedirs(os.path.dirname(desired_path), exist_ok=True)
    else:
        os.makedirs(desired_path, exist_ok=True)
    return desired_path

def main():
    rclpy.init()
    node = MAIN()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
