# Deep Q-Networks

A modular Deep Q-Network (DQN) with experience replay and a soft-updated target network, trained on CartPole (and LunarLander).

---

## Theory

Lecture notes for this topic (TD learning, Q-learning and DQN) are coming soon on https://parag1604.github.io/notes. Until then, the background is in [Introduction to Reinforcement Learning](https://parag1604.github.io/notes/Reinforcement-Learning/Introduction-to-Reinforcement-Learning) (value functions and the Bellman optimality equation for $Q^*$).

## Python Environment Setup

Python >= 3.10, then:

```bash
pip install -r requirements.txt
```

`gymnasium[box2d]` is only needed for LunarLander (it needs `swig` to build; `brew install swig` on macOS). CartPole works with plain `gymnasium`.

## Data

No dataset is needed: the agent collects its own experience in [Gymnasium](https://gymnasium.farama.org/) environments.

## Running

Read the files in this order; each one is a single building block:

1. `q_network.py`: the MLP $Q_w(s,\cdot)$ with one output per action.
2. `replay_buffer.py`: stores transitions $(s, a, r, s', \text{terminated})$ and samples random mini-batches.
3. `agent.py`: $\epsilon$-greedy acting, and the regression of $Q_w(s,a)$ onto the TD target $r + \gamma\,(1-\text{terminated})\max_{a'} Q_{\text{target}}(s',a')$, with a soft target-network update.
4. `main.py`: the training loop.

```bash
python main.py                         # CartPole-v0
python main.py --env LunarLander-v3
python main.py --episodes 20           # quick run
```

Training stops when the average return of the last 100 episodes exceeds the environment's threshold (195 for CartPole-v0, 200 for LunarLander); with the default seed, CartPole-v0 was solved after 900 episodes. The return curve is saved to `plots/dqn_<env>.png` and the network to `models/` (both git-ignored). Add `--render` to watch the greedy agent after training.

A single-file version of the same algorithm, as the last step of a progression from dynamic programming through tabular Q-learning, is in the [RL-Basics](https://github.com/parag1604/RL-Basics) repository.
