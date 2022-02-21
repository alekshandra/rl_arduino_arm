import math
import torch
import matplotlib.pyplot as plt
import random
from env import Environment
from classes import SoftActorCritic


env = Environment()

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
hidden_dim = 256

sac = SoftActorCritic(action_dim, state_dim, hidden_dim, replay_buffer_size=25000)

max_frames = 100000
max_steps = 300
frame_idx = 0
rewards = []
batch_size = 128
exploration_start = 0.95
exploration_end = 0.001

while frame_idx < max_frames:
    state = env.reset()
    episode_reward = 0
    print("\n\n\n\n\n----------- Frame: ", frame_idx)

    for step in range(max_steps):
        print("\n\n------ Taking A Bold Step -----")
        # state[0:4] /= 180
        # state[5:9] /= 180
        if frame_idx > 2000:
            if random.random() > exploration_end+exploration_start*math.exp(0.0001*(2000-frame_idx)):
                print("... with intention!")
                action, _, _, _, _ = sac.policy_net.get_action(torch.FloatTensor(state).unsqueeze(0))
                action = action.detach()[0]
                next_state, reward, done, _ = env.step(action.numpy())
            else:
                print("... randomly...")
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
        else:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

        # print("Main Script Action: ", action)
        # print("Main Script Next State, Reward, Done: ", next_state, reward, done)
        sac.replay_buffer.push(state, action, reward, next_state, done)

        state = next_state.copy()
        # print("Main Script State, Reward: ", state, reward)
        episode_reward += reward
        frame_idx += 1

        if len(sac.replay_buffer) > batch_size:
            sac.update(batch_size)

        if frame_idx % 10000 == 0:
            fig, axs = plt.subplots(1, 2)
            fig.suptitle('frame %s. reward: %s' % (frame_idx, rewards[-1]))
            axs[0].plot(rewards)
            axs[1].plot(rewards[-1000:])
            # plt.plot(rewards)
            plt.show()

        if done:
            break

    rewards.append(episode_reward)

# View a run
env = Environment()
state = env.reset()
cum_reward = 0

plt.ion()
print("\n\n\n\n-------------- Time To See How It Gets On! ------------")
for t in range(500):
    print(t)
    plt.clf()
    env.render()
    action, _, _, _, _ = sac.policy_net.get_action(torch.FloatTensor(state).unsqueeze(0))
    action = action.detach()[0]
    state, reward, done, info = env.step(action.numpy())
    cum_reward += reward
    if done:
        break

    plt.pause(0.1)
