import random

import matplotlib.pyplot as plt
import numpy as np

from DQN import DeepQNetwork
from env import Env
from memory import Memory

N_VM = 7
EPISODES = 400
MINI_BATCH = 128
MEMORY_SIZE = 10000


def run_env(episodes, mini_batch):
    step = 0
    for episode in range(episodes):
        rwd = 0
        obs = env.reset()
        # print(episode)
        while True:
            step += 1

            action = dqn.choose_action(obs)

            # print(action)
            obs_, reward, done = env.step(action)
            # print(done)
            rwd += reward

            memories.remember(obs, action, reward, obs_, done)

            size = memories.pointer
            batch = random.sample(range(size), size) if size < mini_batch else random.sample(
                range(size), mini_batch)
            if step > 200 and step % 5 == 0:
                dqn.learn(*memories.sample(batch))

            obs = obs_

            if done:
                if episode == episodes - 1:
                    print(env.task_exec)
                    print(max(env.vm_time))
                    print(np.sum(env.vm_cost))
                # TODO(hang): env.vm_cost
                if episode % 10 == 0:
                    print(
                        'episode:' + str(episode) + ' steps:' + str(step) + ' reward:' + str(
                            rwd) + ' eps_greedy:' + str(
                            dqn.epsilon))
                rewards.append(rwd)
                break


if __name__ == '__main__':
    rewards = []

    env = Env(N_VM)

    memories = Memory(MEMORY_SIZE)

    dqn = DeepQNetwork(env.n_actions, env.n_features,
                       learning_rate=0.001,
                       replace_target_iter=200,
                       e_greedy_increment=3e-5
                       )

    run_env(EPISODES, MINI_BATCH)

    dqn.plot_cost()

    plt.plot(np.arange(len(rewards)), rewards)
    plt.plot(np.arange(len(rewards)), [138 for i in range(len(rewards))])
    plt.ylabel('reward')
    plt.xlabel('episode')
    plt.show()
