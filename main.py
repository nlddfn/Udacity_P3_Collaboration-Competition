import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from train_eval import evaluate_agent, train_agent
from unityagents import UnityEnvironment

TRAIN = False

# env_path = "/data/Tennis_Linux_NoVis/Tennis.x86_64"
local_path = os.path.dirname(os.path.abspath(__file__))
local_env_path = local_path + "/Tennis.app"
checkpoint_pth = local_path + "/checkpoint_{}_e3200_r077.pth"
# checkpoint_pth = local_path + "/checkpoint_{}.pth"

# Load Env
env = UnityEnvironment(file_name=local_env_path)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print("Number of agents:", num_agents)

# size of each action
action_size = brain.vector_action_space_size
print("Size of each action:", action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print(
    "There are {} agents. Each observes a state with length: {}".format(
        states.shape[0], state_size
    )
)
print("The state for the first agent looks like:", states[0])


start_time = datetime.now()

# VARS
config = {
    "buffer_size": int(1e6),  # replay buffer size
    "batch_size": 1024,  # minibatch size
    "replay_initial": 1024,  # initial memory before updating the network
    "gamma": 0.99,  # discount factor
    "lr_actor": 1e-4,  # learning rate
    "lr_critic": 1e-3,  # learning rate of the critic
    "update_every": 2,  # how often to update the network
    "tau": 1e-3,  # soft update
    "weight_decay": 0,  # l2 weight decay
    "net_body": (256, 128, 64),  # hidden layers
    "per_alpha": 0.6,
    "per_beta": 0.4,
    "per_beta_increment": 0.0001,
    "per_epsilon": 0.0001,
}

agent = Agent(
    state_size=state_size,
    action_size=action_size,
    random_seed=0,
    num_agents=num_agents,
    **config
)

if TRAIN:
    n_episodes = 2600
    print(f"Train Agent for {n_episodes} episodes")
    scores = train_agent(agent, env, n_episodes=n_episodes, min_score=.5, num_agents=num_agents)
    time_elapsed = datetime.now() - start_time
    print("\nTime elapsed (hh:mm:ss.ms) {}".format(time_elapsed))

    # Plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()

# Evaluate agent (using solved agent)
print("Evaluate (solved) Agent")
evaluate_agent(
    agent, env, num_agents=num_agents, checkpoint_pth=checkpoint_pth, num_episodes=10, min_score=.5
)

# Close env
env.close()
