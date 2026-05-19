"""
Nash-Q Baseline (Hu & Wellman 2003) – 4 Q-tables, state = positions only.
Matches the paper's baseline that fails on non-Markovian tasks.
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from game_parameters import *
from reward_machine_task_I import Reward_Machine   # generic version, works for all tasks
from environment import PacmanGridWorld
import nashpy as nash

# ========== Config ==========
NUM_ACTIONS = 4
GRID_SIZE = GRID_W * GRID_H   # 36
ALPHA = 0.15
GAMMA = 0.9
EPSILON_START = 0.5
EPSILON_END = 0.01
DECAY_EPISODES_RATIO = 0.8
EPISODES = 16000
MAX_STEPS = 9999
WINDOW_SIZE = 80

def pos_to_idx(pos):
    return pos[0] * GRID_W + pos[1]

# ========== 4 Q‑tables (same shape for all) ==========
shape = (GRID_SIZE, GRID_SIZE, NUM_ACTIONS, NUM_ACTIONS)
q_ee = np.zeros(shape)   # ego's estimate of ego's Q
q_ae = np.zeros(shape)   # ego's estimate of adversary's Q
q_ea = np.zeros(shape)   # adversary's estimate of ego's Q
q_aa = np.zeros(shape)   # adversary's estimate of adversary's Q

def solve_nash(P, Q):
    """Return mixed strategies (pi_e, pi_a) for given 4x4 matrices."""
    if np.all(P == 0) and np.all(Q == 0):
        return np.ones(4)/4, np.ones(4)/4
    game = nash.Game(P, Q)
    try:
        equilibria = game.support_enumeration()
        pi_e, pi_a = next(equilibria)
        pi_e = np.clip(pi_e, 0, 1)
        pi_a = np.clip(pi_a, 0, 1)
        if pi_e.sum() > 0:
            pi_e /= pi_e.sum()
        else:
            pi_e = np.ones(4)/4
        if pi_a.sum() > 0:
            pi_a /= pi_a.sum()
        else:
            pi_a = np.ones(4)/4
        return pi_e, pi_a
    except Exception:
        return np.ones(4)/4, np.ones(4)/4

def train_nash_q():
    env = PacmanGridWorld()
    decay_episodes = int(EPISODES * DECAY_EPISODES_RATIO)
    all_r_ego = []
    all_r_adv = []
    task_name_string = TASK

    # Create export directory if it doesn't exist
    os.makedirs("../EXPORT", exist_ok=True)

    for ep in range(1, EPISODES + 1):
        pos_e, pos_a = env.reset()
        rm_ego = Reward_Machine('ego')
        rm_adv = Reward_Machine('adv')
        rm_ego.reset()
        rm_adv.reset()
        ep_r_e = ep_r_a = 0.0

        # Linear epsilon decay
        if ep <= decay_episodes:
            eps = EPSILON_START - (EPSILON_START - EPSILON_END) * ep / decay_episodes
        else:
            eps = EPSILON_END
        eps = max(eps, 0.0)   # safety

        for _ in range(MAX_STEPS):
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)

            # ----- Action selection (ε‑greedy, separate policies for each agent) -----
            if np.random.rand() < eps:
                # Explore: both agents take random actions
                ae = np.random.randint(4)
                aa = np.random.randint(4)
            else:
                # Exploit: each agent uses its own Nash equilibrium policy
                # Adversary's policy from its Q‑tables
                _, pi_a = solve_nash(q_ea[s_e, s_a], q_aa[s_e, s_a])
                aa = np.argmax(pi_a) if len(pi_a) > 0 else np.random.randint(4)
                # Ego's policy from its Q‑tables
                pi_e, _ = solve_nash(q_ee[s_e, s_a], q_ae[s_e, s_a])
                ae = np.argmax(pi_e) if len(pi_e) > 0 else np.random.randint(4)

            # ----- Environment step -----
            next_pos_e, next_pos_a = env.step(ae, aa)
            labels = env.get_labels()
            _, r_e = rm_ego.step(labels)
            _, r_a = rm_adv.step(labels)
            ep_r_e += r_e
            ep_r_a += r_a

            ns_e, ns_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            done = (rm_ego.state == 'v_end') or (rm_adv.state == 'v_end')

            # ----- Compute Nash values for the next state -----
            if done:
                v_ee = v_ae = v_ea = v_aa = 0.0
            else:
                # Ego's view of next state's Nash equilibrium
                pi_e_next, pi_a_next = solve_nash(q_ee[ns_e, ns_a], q_ae[ns_e, ns_a])
                v_ee = pi_e_next @ q_ee[ns_e, ns_a] @ pi_a_next
                v_ae = pi_e_next @ q_ae[ns_e, ns_a] @ pi_a_next

                # Adversary's view of next state's Nash equilibrium
                pi_e_next2, pi_a_next2 = solve_nash(q_ea[ns_e, ns_a], q_aa[ns_e, ns_a])
                v_ea = pi_e_next2 @ q_ea[ns_e, ns_a] @ pi_a_next2
                v_aa = pi_e_next2 @ q_aa[ns_e, ns_a] @ pi_a_next2

            # ----- Update all 4 Q‑tables -----
            idx = (s_e, s_a, ae, aa)
            q_ee[idx] += ALPHA * (r_e + GAMMA * v_ee - q_ee[idx])
            q_ae[idx] += ALPHA * (r_a + GAMMA * v_ae - q_ae[idx])
            q_ea[idx] += ALPHA * (r_e + GAMMA * v_ea - q_ea[idx])
            q_aa[idx] += ALPHA * (r_a + GAMMA * v_aa - q_aa[idx])

            pos_e, pos_a = next_pos_e, next_pos_a
            if done:
                break

        all_r_ego.append(min(ep_r_e, 1.0))
        all_r_adv.append(min(ep_r_a, 1.0))

        if ep % 100 == 0:
            avg_e = np.mean(all_r_ego[-100:]) if len(all_r_ego) >= 100 else np.mean(all_r_ego)
            print(f"Episode {ep:5d} | ε={eps:.3f} | Ego reward last 100: {avg_e:.3f}")

        if ep % 500 == 0:
            ckpt_path = f'../EXPORT/q_models_Nash_{task_name_string}_ep{ep}.npz'
            np.savez(ckpt_path, q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)

    return all_r_ego, all_r_adv

def plot_rewards(ego_rewards, adv_rewards, task_name="task_I"):
    window = WINDOW_SIZE
    n_windows = len(ego_rewards) // window
    if n_windows == 0:
        print("Not enough episodes to plot windowed averages.")
        return
    ego_arr = np.array(ego_rewards[:n_windows*window]).reshape(n_windows, window)
    adv_arr = np.array(adv_rewards[:n_windows*window]).reshape(n_windows, window)
    x_plot = np.arange(1, n_windows+1) * window - window//2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,6), sharex=True)
    fig.suptitle(f"Nash-Q baseline (4 tables) - {task_name}", fontsize=13)
    ax1.plot(x_plot, ego_arr.mean(axis=1), 'k--', linewidth=1.5, label='Nash-Q')
    ax1.set_ylabel("Ego reward")
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend()
    ax2.plot(x_plot, adv_arr.mean(axis=1), 'k--', linewidth=1.5, label='Nash-Q')
    ax2.set_ylabel("Adversary reward")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend()
    plt.tight_layout()
    os.makedirs("../EXPORT", exist_ok=True)
    plt.savefig(f"../EXPORT/nash_q_{task_name}_windowed.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    print("Training Nash-Q baseline (4 Q-tables, positions only)...")
    rewards_e, rewards_a = train_nash_q()
    plot_rewards(rewards_e, rewards_a, task_name=TASK)