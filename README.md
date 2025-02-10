# File Descriptions for PPO Algorithm

### `walls.launch.py`
- Launches Gazebo with the training environment.
- Spawns the TurtleBot3 robot in the simulated environment.

### `Environment_new.py`
- Acts as the environment file for the RL setup.
- Performs the following tasks:
  - Executes steps in the environment.
  - Generates rewards for each state-action pair.
  - Outputs the next state.
  - Spawns the goal at a random position after each episode.
- Facilitates interaction between ROS2, Gazebo, and the PPO algorithm.

### `main.py`
- Main file for training TurtleBot3 using the PPO algorithm.
- Calls `ppo.py`, where the PPO algorithm is implemented.

### `test.py`
- Loads the best-trained model for testing.
- Calls `eval_policy.py` to:
  - Test the model over 100 episodes.
  - Log episode statuses and paths followed.

### `net_actor.py` and `net_critic.py`
- Contain the PyTorch implementations of the actor and critic networks.
- Used by `ppo.py` during training.

### `inspect_log.py`
- Reads the log file created after running `test.py`.
- Computes and displays the accuracy of the tested model.
