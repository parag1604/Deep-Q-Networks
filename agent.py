"""
Step 3 of 4 - the DQN agent.

  - act():   epsilon-greedy with respect to Q_w.
  - step():  store the transition; every `update_every` steps, learn from a
             random mini-batch.
  - learn(): regress Q_w(s, a) onto the TD target
                 y = r + gamma (1 - terminated) max_a' Q_target(s', a'),
             computed with a separate TARGET network that slowly follows
             Q_w (soft update, w_target <- tau w + (1 - tau) w_target).
             Without it, every update would also move the target it is
             chasing, which easily diverges.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, state_size, action_size, device, gamma, tau,
                 update_every, lr, buffer_size, batch_size, hid_dim=64):
        self.action_size = action_size
        self.q_net = QNetwork(state_size, hid_dim, action_size).to(device)
        self.q_target = QNetwork(state_size, hid_dim, action_size).to(device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = Adam(self.q_net.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size, device)
        self.t_step = 0

        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def step(self, state, action, reward, next_state, terminated):
        self.memory.add(state, action, reward, next_state, terminated)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            self.learn(self.memory.sample())

    def act(self, state, eps=0.):
        '''Epsilon-greedy action selection.'''
        if np.random.random() < eps:
            return np.random.randint(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.q_net(state)
        return int(action_values.argmax(dim=1).item())

    def learn(self, experiences):
        states, actions, rewards, next_states, terminated = experiences
        with torch.no_grad():
            q_targets_next = self.q_target(next_states).max(1, keepdim=True)[0]
            # a terminal state has no future: do not bootstrap
            q_targets = rewards + self.gamma * q_targets_next * (1 - terminated)
        q_expected = self.q_net(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_net, self.q_target, self.tau)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)
