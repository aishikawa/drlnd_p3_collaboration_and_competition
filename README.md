This is a solution for project 3 of Udacity deep reinforcement learning. This repository based on [Udacity's deep-reinforcement-learning repository](https://github.com/udacity/deep-reinforcement-learning).

## Project Details
The purpose of this project is solving the environment based on [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) 

### Environment
In this environment, two agents control rackets to bounce a ball over a net. The task is episodic.

- Ovservation space: 24 variables. 3 stacks of 8 variables correspoinding to position and velocity of ball and racket.
- Action space: 2 numbers correspoinding to movement toward net or away from net, and jumping.
- Reward
    - +0.1: if an agent hits the ball over the net
    - -0.01: if an agent lets a ball hit the ground or hits the ball out of bounds

### Solving the Environment
In order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

## Getting Started

#### Create Python environment
Clone [this repository](https://github.com/udacity/deep-reinforcement-learning), and follow the [instructions]() to set up Python environment.

#### Download the Unity Environment
Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the `tennis/` folder.

#### Instructions
Activate the python environment and run main.py
```
python main.py
```