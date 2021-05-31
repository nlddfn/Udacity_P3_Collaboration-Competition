from collections import deque

import numpy as np
import torch


def train_agent(
    agent,
    env,
    n_episodes=26000,
    max_t=1000,
    min_score=.5,
    num_agents=2,
    print_every=100,
):
    """DDPG-Learning.
    Params
    ======
        agent: An instance of Agent class
        env: Unity environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    brain_name = env.brain_names[0]
    score_list = []
    scores_window = deque(maxlen=100)
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        agent.reset()
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones, t)
            scores += rewards
            states = next_states
            if np.any(dones):
                break
        max_score = np.max(scores)
        scores_window.append(max_score)
        score_list.append(max_score)

        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )

        if i_episode % print_every == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

        if np.mean(scores_window) >= min_score:
            print(
                f"\nEnvironment solved in {i_episode:d} episodes!"
                f"\tAverage Score: {np.mean(scores_window):.2f}"
            )
            torch.save(
                agent.actor_local.state_dict(), "checkpoint_env_solved_actor.pth"
            )
            torch.save(
                agent.critic_local.state_dict(), "checkpoint_env_solved_critic.pth"
            )
            break
    return score_list


def execute_policy(env, agent, num_agents, step):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    while True:
        actions = agent.act(states)  # select an action (for each agent)
        env_info = env.step(actions)[brain_name]  # send the action to the environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        rewards = env_info.rewards  # get reward (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break
    print(f"Total score (averaged over agents) for episode {step}: {np.mean(scores)}")
    return np.mean(scores)


def evaluate_agent(
    agent, env, num_agents, checkpoint_pth="{}", num_episodes=100, min_score=30
):
    score_lst = []
    agent.actor_local.load_state_dict(
        torch.load(
            checkpoint_pth.format("actor"),
            map_location=torch.device('cpu')
        ),
    )
    agent.critic_local.load_state_dict(
        torch.load(
            checkpoint_pth.format("critic"),
            map_location=torch.device('cpu')
        ),
    )
    for i in range(num_episodes):
        score_lst.append(execute_policy(env, agent, num_agents, i))

    assert (
        np.mean(score_lst) >= min_score
    ), f"Environment not solved: Expected score >= {min_score}, found {np.mean(score_lst)}"
