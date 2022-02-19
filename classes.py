from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    """Class to implement Experience Replay Buffer for off-policy methods

    Attributes:
        buffer: A double ended queue, with a user-defined capacity, that stores information about previous moves
    """
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    """Class to implement Soft Actor Critic's Value Network, which gives the expected return if you start at a state

    Attributes:
        linear1: A Linear input layer to transform the input data
        linear2: A Linear hidden layer to transform the data from the previous layer
        linear3: A Linear output layer to prepare the final prediction
    """
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    """Class to implement Soft Actor Critic's Q Network, which gives the expected return if you start at a state
    and take an action

        Attributes:
            linear1: A Linear input layer to transform the input data
            linear2: A Linear hidden layer to transform the data from the previous layer
            linear3: A Linear output layer to prepare the final prediction
    """
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super().__init__()

        self.linear1 = nn.Linear(num_inputs+num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    """Class to implement Soft Actor Critic's Policy Network, which returns the mean and std of the distribution of
    optimal actions to take based on the state

        Attributes:
            log_std_min: The smallest allowable std (logs used by convention to keep everything positive)
            log_std_max: The largest allowable std (logs used by convention to keep everything positive)

            linear1: A Linear input layer to transform the data from the previous layer
            linear2: A Linear hidden layer to transform the data from the previous layer

            mean_linear: A Linear output layer to prepare the means for each action's distribution
            log_std_linear: A Linear output layer to prepare the log stds for each action's distribution
    """
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        action, log_std, mean, std, z = self.get_action(state)
        log_prob = Normal(mean, std).log_prob(mean+std*z) - torch.log(1-action.pow(2)+epsilon)
        return action, log_prob, z, mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean+std*z)

        return action, log_std,  mean, std, z


class SoftActorCritic:
    """Class to implement Soft Actor Critic agent

        Attributes:
            value_net: The Value Network
            target_value_net: The target Value Network (required since Value Network is indirectly dependent on itself)

            soft_q_net1: The first Q Network
            soft_q_net2: The second Q Network (used to prevent overestimation of Q-values)
            policy_net: The Policy Network

            value_criterion: The Loss function for the Value Network
            soft_q_criterion1: The Loss function for the first Q Network
            soft_q_criterion2: The Loss function for the second Q Network

            value_optimizer: The Optimizer for the Value Network
            soft_q_optimizer1: The Optimizer for the first Q Network
            soft_q_optimizer2: The Optimizer for the second Q Network
            policy_optimizer: The Optimizer for the Policy Network

            replay_buffer: The Experience Replay Buffer for the agent to sample from
    """
    def __init__(self, action_dim, state_dim, hidden_dim, value_lr=3e-4, soft_q_lr=3e-4, policy_lr=3e-4,
                 replay_buffer_size=1000000):
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.target_value_net = ValueNetwork(state_dim, hidden_dim)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def train_soft_q(self, next_state, reward, done, predicted_q_value1, predicted_q_value2, gamma):
        target_value = self.target_value_net(next_state)
        target_q_value = reward+(1-done)*gamma*target_value
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

    def train_value(self, state, predicted_value, new_action, log_prob):
        predicted_new_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        target_value_func = predicted_new_q_value-log_prob
        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return predicted_new_q_value

    def train_policy(self, log_prob, predicted_new_q_value):
        policy_loss = (log_prob-predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def update(self, batch_size, gamma=0.99, soft_tau=1e-2):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        predicted_q_value1 = self.soft_q_net1(state, action)
        predicted_q_value2 = self.soft_q_net2(state, action)
        predicted_value = self.value_net(state)
        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

        self.train_soft_q(next_state, reward, done, predicted_q_value1, predicted_q_value2, gamma)
        predicted_new_q_value = self.train_value(state, predicted_value, new_action, log_prob)
        self.train_policy(log_prob, predicted_new_q_value)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data*(1.0-soft_tau)+param.data*soft_tau)

