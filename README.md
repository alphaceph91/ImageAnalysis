# Image and Texture Analysis Repository

## Overview
This repository contains a collection of Python scripts for **image processing, texture analysis, and feature extraction**. The scripts apply various computer vision and machine learning techniques to analyze textures, compare images, and extract meaningful patterns. 

![Image](https://github.com/alphaceph91/asteroidMLGame/blob/main/image.png)

![Image](https://github.com/alphaceph91/asteroidMLGame/blob/main/training_performance.png)

## Features

### Custom Reinforcement Learning Environment
- The spaceship is controlled by a DQN agent that learns from the environment.
- The environment features a detection circle around the spaceship. When an asteroid enters this circle and is approaching, the spaceship fires a heat-seeking missile.
- Asteroids have varied sizes, speeds, and horizontal movements. Asteroids even interact with each other via simple collision physics.
- The environment maintains various metrics, including score, number of asteroids shot, missiles fired, and shooting accuracy.

### Deep Q-Network (DQN) Agent:
- The agent learns a policy using experience replay and periodically updates a target network.
- Training metrics (total reward, epsilon) are logged in a CSV file, and performance plots are generated.
- Model checkpoints are saved in the models folder, including both the final model and the best-performing model.

### Deep Q-Network (DQN) Agent:
- A separate evaluation script ([testGame.py](https://github.com/alphaceph91/asteroidMLGame/blob/main/testGame.py)) loads the trained model checkpoint and runs test episodes with a greedy policy (epsilon set to 0).
- During evaluation, performance metrics such as shooting accuracy (asteroids shot / missiles fired) are computed and printed.


## How to Run

### Training
- Run the DQN training script: python [DQN_agent.py](https://github.com/alphaceph91/asteroidMLGame/blob/main/DQN_agent.py) to train your agent
- Training metrics will be logged in as a CSV file in log/training_metrics.csv
- A performance plot is saved as log/training_performance.png
- Model checkpoints are saved in the models/ folder

### Evaluation
- Once training is completed, trained agent can be evaluated: python [testGame.py](https://github.com/alphaceph91/asteroidMLGame/blob/main/testGame.py)
- This script loads the trained model (e.g., models/best_dqn_policy.pth), sets epsilon to 0, and runs test episodes to display performance metrics and visually demonstrate the agent's behavior
- You can integrate the trained agent into the main game loop (e.g., in main.py) to have the spaceship controlled autonomously in a live game setting


## Metrics and Performance
- Total Reward: The cumulative reward the agent receives per episode
- Epsilon: The exploration rate used during training (decays over time)
- Shooting Accuracy: Computed as (Asteroids Shot) / (Missiles Fired) to measure how accurately the spaceship targets asteroids


## Future Enhancements
- Advanced Collision Physics: The asteroids interactions will be refined in the future for increased realism
- Additional RL Algorithms: More RL algorthims will be implemented to experiment with Double DQN, Dueling DQN etc
- Improved State Representations: Additional features such as asteroid trajectories, aggregated threat levels will be implemented to enhance performance


## Pretrained Model
- A pretrained model trained on 1000 episodes could be downloaded [here](https://drive.google.com/drive/folders/15FeuCQdentD1Eay-NZthusjxmKjzQw2q?usp=drive_link)
- The model converged at 919th episode with a reward of 4095 and an epsilon of 0.01
