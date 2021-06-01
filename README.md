[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Train a Unity Environment (Tennis) using Deep Deterministic Policy Gradient

### Introduction

For this project, two raquets are trained to collaborate in order to keep the ball in the court.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position (x, y) and velocity (v_x, v_y) of the racket and the ball. Each agent receives its own, local observations. This means that each raquet moves along the x axis, from 0 (the net) to the court boundary ~= -12.  The position of the ball along the x axis is negative when the ball is located "above" the raquet and positive when it is on the other side of the court. 

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Clone the `Udacity_RL_P3_Collaboration-Competition` GitHub repository, place the file in the folder and decompress it. 

3. Create a virtual environment and install the required libraries. For OSX users, you can use the MakeFile included in the repo. The option `make all` will create a new venv called `Udacity_RL_P3` and install the relevant dependencies to execute the notebook.

4. Activate the virtual environment using `source ./Udacity_RL_P3/bin/activate`

5. Type `jupyter lab` and select `Udacity_RL_P3` kernel.

## Train and execute the model

Within the virtual environment you can train and evaluate the model using `python main.py`. By default, the script will load the environment and evaluate a pre-trained model. If you  want to retrain the model, set `TRAIN = True` in `main.py` and then run the script.

You can also use the notebook `Tennis.ipynb` to train and evaluate the model. Set the flag train to `TRUE` to re-train the model. Further details can be found [here](Report.md)
