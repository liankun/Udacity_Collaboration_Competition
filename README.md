[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Instructions

The Project is done in windows 64bit environment. 
- Tennis.ipynb : The main code to train and evalue the agent (one agent version)
- MAgent.py : the code of ReplayPool and Agent to implement multi-agents
- DDPGAgent.py: DDPG Agent
- model.py : the structor of the actor and critic
- SumTree.py: the structure to implement priority buffer
- checkpoint_actor0_gpu.pth : the trained actor weight for the first agent
- checkpoint_critic0_gpu.pth: the trained critic weight for the first agent
- checkpoint_actor1_gpu.pth : the trained actor weight for the second agent
- checkpoint_critic1_gpu.pth: the trained critic weight for the second agent
- Report.ipynb : report of the project
- Tennis_Windows_x86_64 : The unity environment for the project

Environment setting (windows):
- create virtual environment: conda create --name project_collaboration_competition python=3.6.9
- activate environment: conda activate project_collaboration_competition
- install neccessary packets:
  1. install pytorch: conda install pytorch==1.0.1 torchvision==0.2.2 cudatoolkit=10.0 -c pytorch 
  2. install unityagents: pip install unityagents
  3. install matplotlib: conda install matplotlib
- clone the repository: git clone https://github.com/liankun/Udacity_Collaboration_Competition.git
- cd Udacity_Collaboration_Competition
- create IPython kernel: python -m ipykernel install --user --name project_navigation --display-name "project_collaboration_competition" <br/>

Now you can run the project by using the project_collaboration_competition environment in the jupyter notebook.
