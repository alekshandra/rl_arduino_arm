import torch
import matplotlib.pyplot as plt

from env import Environment
from classes import SoftActorCritic


env = Environment()

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 256

sac = SoftActorCritic(action_dim, state_dim, hidden_dim)

max_frames = 1500
max_steps = 300
frame_idx = 0
rewards = []
batch_size = 128

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    print(frame_idx)

    for step in range(max_steps):
        if frame_idx > 1000:
            action, _, _, _, _ = sac.policy_net.get_action(torch.FloatTensor(state).unsqueeze(0))
            action = action.detach()[0]
            next_state, reward, done, _ = env.step(action.numpy())
        else:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

        # print(action)
        # print(next_state, reward, done)
        sac.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        # print(state, reward)
        episode_reward += reward
        frame_idx += 1

        if len(sac.replay_buffer) > batch_size:
            sac.update(batch_size)

        if frame_idx % 10000 == 0:
            plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
            plt.plot(rewards)
            plt.show()

        if done:
            break

    rewards.append(episode_reward)

# View a run
env = Environment()
state = env.reset()
cum_reward = 0

plt.ion()

for t in range(500):
    print(t)
    plt.clf()
    env.render()
    action, _, _, _, _ = sac.policy_net.get_action(torch.FloatTensor(state).unsqueeze(0))
    action = action.detach()[0]
    state, reward, done, info = env.step(action)
    cum_reward += reward
    if done:
        break

    plt.pause(10.1)
