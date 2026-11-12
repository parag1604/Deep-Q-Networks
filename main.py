"""
Step 4 of 4 - the training loop.

Run episodes with an epsilon-greedy agent whose epsilon decays
multiplicatively per episode (eps <- max(eps_end, eps_decay * eps)), feed
every transition to the agent, and stop when the average return of the
last 100 episodes beats the environment's reward threshold (195 for
CartPole-v0, 200 for LunarLander-v3).

Two details that matter:
  - gymnasium separates `terminated` (a real terminal state: no
    bootstrapping) from `truncated` (a time limit: the state is not
    terminal). Only `terminated` is passed to the agent.
  - `max_t` caps the episode length in addition to the environment's own
    time limit.

Run:  python main.py                          (CartPole-v0)
      python main.py --env LunarLander-v3     (needs gymnasium[box2d])
      python main.py --episodes 20            (quick run)
"""
import argparse
import os
import random
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent


def get_device():
    '''
    cuda if available, otherwise cpu.

    Apple's mps backend is skipped on purpose: these networks are tiny and
    are called once per environment step, so the per-call overhead of mps
    dominates (an actor-critic epoch took about 23 s on mps vs 0.4 s on cpu).
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed, env):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)


def dqn(env, agent, n_episodes=1000, max_t=1000, eps_start=1.0,
        eps_end=0.01, eps_decay=0.995, target_score=195.0):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated)
            state = next_state
            score += reward
            if terminated or truncated:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if len(scores_window) == 100 and np.mean(scores_window) > target_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                  .format(i_episode - 100, np.mean(scores_window)))
            break
    print()
    return scores


def plot_scores(scores, title, save_dir='plots', threshold=None):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores, alpha=0.6,
             label="Episode return")
    if len(scores) >= 100:
        ma = [np.mean(scores[i - 100:i]) for i in range(100, len(scores) + 1)]
        plt.plot(np.arange(100, len(scores) + 1), ma,
                 label="Moving average (100 episodes)")
    if threshold is not None:
        plt.axhline(threshold, color="k", linestyle="--", linewidth=1,
                    label=f"Solved threshold ({threshold:g})")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid()
    path = os.path.join(save_dir, f'{title}.png')
    plt.savefig(path)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description='Train a DQN agent.')
    parser.add_argument('--env', default='CartPole-v0')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--max-t', type=int, default=1000,
                        help='maximum steps per episode')
    parser.add_argument('--buffer-size', type=int, default=int(1e5))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='soft update rate of the target network')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--update-every', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--render', action='store_true',
                        help='watch the greedy agent for one episode')
    args = parser.parse_args()

    env = gym.make(args.env)
    set_seed(args.seed, env)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    agent = Agent(state_size=env.observation_space.shape[0],
                  action_size=env.action_space.n, device=get_device(),
                  gamma=args.gamma, tau=args.tau,
                  update_every=args.update_every, lr=args.lr,
                  buffer_size=args.buffer_size, batch_size=args.batch_size)

    scores = dqn(env, agent, n_episodes=args.episodes, max_t=args.max_t,
                 target_score=env.spec.reward_threshold)
    print('Saved', plot_scores(scores, f'dqn_{args.env}',
                               threshold=env.spec.reward_threshold))
    os.makedirs('models', exist_ok=True)
    torch.save(agent.q_net.state_dict(), f'models/dqn_{args.env}.pth')

    if args.render:
        env = gym.make(args.env, render_mode='human')
        obs, _ = env.reset()
        reward, done = 0, False
        while not done:
            obs, rew, terminated, truncated, _ = env.step(agent.act(obs))
            reward += rew
            done = terminated or truncated
        print("Reward =", reward)
    env.close()


if __name__ == '__main__':
    main()
