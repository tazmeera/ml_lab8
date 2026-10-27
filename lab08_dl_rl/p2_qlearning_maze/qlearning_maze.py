"""
Lab Sheet-08 | Project 2: Q-Learning based Maze Solver
MCA III Semester (Session 2026-2027)

A custom Gymnasium environment (MazeEnv) + tabular Q-Learning agent.
  * Reward shaping : -0.04 per step, -1 for hitting a wall, +10 for reaching the goal
  * Outputs        : learning curves, Q-value heatmap, solved-path plot,
                     hyperparameter comparison, saved Q-table (.npy)
  * Verification   : agent path length is compared with the true shortest path (BFS)

Run:  python qlearning_maze.py
"""
import os
from collections import deque

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from matplotlib.colors import ListedColormap

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 7

# 0 = free, 1 = wall, S = (0,0), G = (bottom-right)
MAZE = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
])
START, GOAL = (0, 0), (9, 9)
ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # up, down, left, right
ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


class MazeEnv(gym.Env):
    """Grid maze. Observation = flattened cell index, action in {up, down, left, right}."""

    def __init__(self, maze=MAZE, start=START, goal=GOAL, max_steps=400):
        super().__init__()
        self.maze, self.start, self.goal, self.max_steps = maze, start, goal, max_steps
        self.n_rows, self.n_cols = maze.shape
        self.observation_space = spaces.Discrete(self.n_rows * self.n_cols)
        self.action_space = spaces.Discrete(4)
        self.pos, self.steps = start, 0

    def _obs(self):
        return self.pos[0] * self.n_cols + self.pos[1]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.steps = self.start, 0
        return self._obs(), {}

    def step(self, action):
        self.steps += 1
        dr, dc = ACTIONS[int(action)]
        r, c = self.pos[0] + dr, self.pos[1] + dc
        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or self.maze[r, c] == 1:
            reward = -1.0                      # bumped into wall / boundary, stay put
        else:
            self.pos = (r, c)
            reward = 10.0 if self.pos == self.goal else -0.04
        terminated = self.pos == self.goal
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}


def bfs_shortest_path(maze, start, goal):
    """Ground truth shortest path length (number of moves)."""
    q, seen = deque([(start, 0)]), {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if (0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]
                    and maze[nr, nc] == 0 and (nr, nc) not in seen):
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None


def train_q_learning(env, episodes=3000, alpha=0.1, gamma=0.99,
                     eps_start=1.0, eps_min=0.02, eps_decay=0.999, seed=SEED):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.observation_space.n, env.action_space.n))
    eps = eps_start
    rewards, steps_list, successes = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        total, done = 0.0, False
        while not done:
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(q[s]))
            s2, r, term, trunc, _ = env.step(a)
            # Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
            target = r + (0.0 if term else gamma * np.max(q[s2]))
            q[s, a] += alpha * (target - q[s, a])
            s, total, done = s2, total + r, term or trunc
        eps = max(eps_min, eps * eps_decay)
        rewards.append(total)
        steps_list.append(env.steps)
        successes.append(int(env.pos == env.goal))
    return q, np.array(rewards), np.array(steps_list), np.array(successes)


def greedy_path(env, q, limit=200):
    s, _ = env.reset()
    path = [env.pos]
    for _ in range(limit):
        s, _, term, trunc, _ = env.step(int(np.argmax(q[s])))
        path.append(env.pos)
        if term or trunc:
            break
    return path, env.pos == env.goal


def moving_avg(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


if __name__ == "__main__":
    env = MazeEnv()
    optimal = bfs_shortest_path(MAZE, START, GOAL)
    print(f"True shortest path (BFS): {optimal} moves")

    # ------------------------------------------------------------------ main run
    EPISODES = 3000
    q, rewards, steps, succ = train_q_learning(env, episodes=EPISODES)
    path, solved = greedy_path(env, q)
    agent_len = len(path) - 1
    print(f"Agent solved maze: {solved} | agent path = {agent_len} moves | optimal = {optimal}")
    np.save(f"{OUT}/q_table.npy", q)
    print("Saved Q-table -> q_table.npy")

    # ------------------------------------------------------------------ learning curves
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(rewards, alpha=0.25, label="episode"); ax[0].plot(range(49, EPISODES), moving_avg(rewards), label="50-ep avg")
    ax[0].set(title="Total reward per episode", xlabel="Episode", ylabel="Reward"); ax[0].legend()
    ax[1].plot(steps, alpha=0.25); ax[1].plot(range(49, EPISODES), moving_avg(steps))
    ax[1].set(title="Steps per episode", xlabel="Episode", ylabel="Steps", yscale="log")
    ax[2].plot(range(49, EPISODES), moving_avg(succ))
    ax[2].set(title="Success rate (50-ep avg)", xlabel="Episode", ylim=(0, 1.05))
    plt.tight_layout(); plt.savefig(f"{OUT}/learning_curves.png", dpi=150); plt.close()

    # ------------------------------------------------------------------ maze + path + policy
    cmap = ListedColormap(["#f7f7f7", "#2b2d42"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(MAZE, cmap=cmap)
    pr, pc = zip(*path)
    ax[0].plot(pc, pr, "-o", color="#ef233c", ms=5, lw=2, label=f"agent path ({agent_len} moves)")
    ax[0].scatter(*START[::-1], s=250, c="#2a9d8f", marker="s", zorder=5, label="Start")
    ax[0].scatter(*GOAL[::-1], s=300, c="#f4a261", marker="*", zorder=5, label="Goal")
    ax[0].set_title("Learned path through the maze"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_xticks([]); ax[0].set_yticks([])

    vmax = q.max(axis=1).reshape(MAZE.shape)
    vmasked = np.ma.masked_where(MAZE == 1, vmax)
    im = ax[1].imshow(vmasked, cmap="viridis")
    for r in range(MAZE.shape[0]):
        for c in range(MAZE.shape[1]):
            if MAZE[r, c] == 0 and (r, c) != GOAL:
                ax[1].text(c, r, ARROWS[int(np.argmax(q[r * 10 + c]))], ha="center", va="center", color="white", fontsize=12)
    ax[1].set_title("State values max Q(s,a) with learned policy"); ax[1].set_xticks([]); ax[1].set_yticks([])
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    plt.tight_layout(); plt.savefig(f"{OUT}/maze_solution_policy.png", dpi=150); plt.close()

    # ------------------------------------------------------------------ hyperparameter tuning
    configs = {
        "alpha=0.1, gamma=0.99 (base)": dict(alpha=0.1, gamma=0.99),
        "alpha=0.5, gamma=0.99": dict(alpha=0.5, gamma=0.99),
        "alpha=0.01, gamma=0.99": dict(alpha=0.01, gamma=0.99),
        "alpha=0.1, gamma=0.90": dict(alpha=0.1, gamma=0.90),
        "alpha=0.1, eps_decay=0.9995": dict(alpha=0.1, gamma=0.99, eps_decay=0.9995),
        "alpha=0.1, eps_decay=0.99": dict(alpha=0.1, gamma=0.99, eps_decay=0.99),
    }
    rows, curves = [], {}
    for name, cfg in configs.items():
        runs = []
        for sd in range(5):                               # 5 seeds -> mean +/- std
            e = MazeEnv()
            qq, rw, st, sc = train_q_learning(e, episodes=EPISODES, seed=100 + sd, **cfg)
            p, ok = greedy_path(e, qq)
            first = next((i for i in range(len(sc) - 49) if sc[i:i + 50].mean() >= 0.95), np.nan)
            runs.append((rw, ok, len(p) - 1, first))
        rw_all = np.mean([r[0] for r in runs], axis=0)
        curves[name] = rw_all
        rows.append(dict(
            Config=name,
            Solved=f"{sum(r[1] for r in runs)}/5",
            Avg_Path_Moves=np.mean([r[2] for r in runs if r[1]]) if any(r[1] for r in runs) else np.nan,
            Optimal_Moves=optimal,
            Avg_Final_Reward=np.mean([r[0][-200:].mean() for r in runs]),
            Episodes_To_95pct_Success=np.nanmean([r[3] for r in runs]) if not all(np.isnan(r[3]) for r in runs) else np.nan,
        ))
    tune_df = pd.DataFrame(rows)
    tune_df.to_csv(f"{OUT}/hyperparameter_comparison.csv", index=False)

    plt.figure(figsize=(10, 5))
    for name, c in curves.items():
        plt.plot(range(49, EPISODES), moving_avg(c), label=name)
    plt.title("Effect of hyperparameters on learning (mean of 5 seeds)")
    plt.xlabel("Episode"); plt.ylabel("Reward (50-ep avg)"); plt.ylim(-30, 11); plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/hyperparameter_comparison.png", dpi=150); plt.close()

    print("\n=== Hyperparameter comparison ===")
    print(tune_df.round(2).to_string(index=False))
