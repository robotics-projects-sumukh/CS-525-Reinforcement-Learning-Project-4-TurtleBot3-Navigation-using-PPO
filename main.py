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

        # PPO hyperparameters
        self.hyperparameters = {
            'timesteps_per_batch': 4000,
            'max_timesteps_per_episode': 400,
            'gamma': 0.99,
            'n_updates_per_iteration': 50,
            'lr': 3e-4,
            'clip': 0.2,
            'render': True,
            'render_every_i': 10,
            'log_dir': 'log',
            'exp_id': "v01_wall"
        }
        self.hyperparameters['log_dir'] = f'summary'

        # Train or test
        self.train()

    def train(self):
        print("Start Training ...", flush=True)
        agent = PPO(
            policy_class=NetActor,
            value_func=NetCritic,
            env=self.env,
            state_dim=state_dim,
            action_dim=action_dim,
            **self.hyperparameters
        )
        past_action = np.array([0., 0.])

        print("Training from scratch.", flush=True)

        # Train the PPO model
        agent.learn(total_timesteps=6900000000, past_action=past_action)


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
