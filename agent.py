import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
from utils import OrnsteinUhlenbeckProcess, PEReplayBuffer, ReplayBuffer


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 1024  # minibatch size
REPLAY_INITIAL = BATCH_SIZE  # initial memory before updatting the network
GAMMA = 0.99  # discount factor
LR_ACTOR = 1e-4  # learning rate
LR_CRITIC = 1e-3  # learning rate of the critic
UPDATE_EVERY = 4  # how often to update the network
TAU = 1e-3  # soft update
WEIGHT_DECAY = 0  # L2 weight decay
NET_BODY = (256, 128)  # hidden layers
PRIORITIZED = False
PER_ALPHA = 0
PER_BETA = 0
PER_BETA_INCREMENT = 1e-4
PER_EPSILON = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        state_size,
        action_size,
        random_seed,
        num_agents,
        net_body=NET_BODY,
        gamma=GAMMA,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        replay_initial=REPLAY_INITIAL,
        update_every=UPDATE_EVERY,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
        weight_decay=WEIGHT_DECAY,
        tau=TAU,
        prioritized=PRIORITIZED,
        per_alpha=PER_ALPHA,
        per_beta=PER_BETA,
        per_beta_increment=PER_BETA_INCREMENT,
        per_epsilon=PER_EPSILON,
    ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_initial = replay_initial
        self.update_every = update_every
        self.tau = tau
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prioritized = prioritized

        fc1_units, fc2_units, fc3_units = net_body
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, fc1_units, fc2_units, fc3_units
        ).to(self.device)
        self.actor_target = Actor(
            state_size, action_size, fc1_units, fc2_units, fc3_units
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size, action_size, fc1_units, fc2_units, fc3_units
        ).to(self.device)
        self.critic_target = Critic(
            state_size, action_size, fc1_units, fc2_units, fc3_units
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process
        self.noise = OrnsteinUhlenbeckProcess((num_agents, action_size), random_seed)

        # Replay memory
        if self.prioritized:
            self.memory = PEReplayBuffer(
                buffer_size=buffer_size,
                batch_size=batch_size,
                device=self.device,
                alpha=per_alpha,
                beta=per_beta,
                beta_increment=per_beta_increment,
                epsilon=per_epsilon
            )
        else:
            self.memory = ReplayBuffer(
                buffer_size=buffer_size,
                batch_size=batch_size,
                device=self.device,
                seed=0,
            )

    def step(self, states, actions, rewards, next_states, dones, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(
            states, actions, rewards, next_states, dones
        ):
            if self.prioritized:
                # self.append_sample(state, action, reward, next_state, done)
                self.memory.add(1, (state, action, reward, next_state, done))
            else:
                self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps after reaching the minimal sample size.
        if (self.memory.tree.n_entries > self.replay_initial) and (step % self.update_every == 0):
            experiences = self.memory.sample()
            self.update(experiences, self.gamma)

    def append_sample(self, state, action, reward, next_state, done):
        """Save sample (error,<s,a,r,s'>) to the replay memory"""

        # Set network to eval mode
        self.actor_target.eval()
        self.critic_target.eval()
        self.critic_local.eval()

        state = torch.from_numpy(state).float().to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)

        with torch.no_grad():
            # Get predicted Q values (for next state) from target model
            next_action = self.actor_target.forward(next_state)
            Q_targets_next = self.critic_target.forward(next_state, next_action)
            Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))
            # Get expected Q values from critic model
            Q_expected = self.critic_local.forward(state, action)

        error = (Q_targets - Q_expected).pow(2).data.cpu().numpy()

        # Set network to train mode
        self.actor_target.train()
        self.critic_target.train()
        self.critic_local.train()

        self.memory.add(error, (state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def update(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        (states, actions, rewards, next_states, dones), idxs, is_weight = experiences

        # Update Critic
        # Get expected Q values from critic model
        Q_expected = self.critic_local.forward(states, actions)
        # Get predicted Q values (for next states) from target model
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions)
        Q_target = rewards + (self.gamma * next_Q * (1 - dones))

        if self.prioritized:
            # Update priorities in ReplayBuffer
            loss = (Q_expected - Q_target).pow(2).data.cpu().numpy()
            self.memory.update(idxs, loss.reshape(is_weight.shape) * is_weight)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Update Actor
        # Get expected actions from actor model
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target(self.critic_local, self.critic_target, self.tau)
        self.update_target(self.actor_local, self.actor_target, self.tau)

    def update_target(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
