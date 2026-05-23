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
import warnings

# ========== Config ==========
NUM_ACTIONS = 4
GRID_SIZE = GRID_W * GRID_H   # 36
ALPHA = 0.15
GAMMA = 0.9
EPSILON_START = 0.25
EPSILON_END = 0.01
DECAY_EPISODES_RATIO = 0.8
EPISODES = 15000
MAX_STEPS = 5000
WINDOW_SIZE = 80

def pos_to_idx(pos):
    return pos[0] * GRID_W + pos[1]

# ========== 4 Q‑tables (same shape for all) ==========
shape = (GRID_SIZE, GRID_SIZE, NUM_ACTIONS, NUM_ACTIONS)
q_ee = np.zeros(shape)   # ego's estimate of ego's Q
q_ae = np.zeros(shape)   # ego's estimate of adversary's Q
q_ea = np.zeros(shape)   # adversary's estimate of ego's Q
q_aa = np.zeros(shape)   # adversary's estimate of adversary's Q

def solve_stage_game(q_matrix_ego, q_matrix_adv, 
                     agent_actions = ['up', 'down', 'left', 'right'],
                     debug = DEBUG
                     ):
    '''
    This function is about solving the game, by updating the tabular q functions (matrices) 
    for both ego and adv as paper stated. This return STRATEGIES (policies) given the q-tables.
    '''

    NUM_ACTIONS = len(agent_actions)
    
    ## FAST BYPASS FOR UNLEARNED MATRICES (Speeds up early training 10x)
    # If the Q-values haven't meaningfully diverged (max diff is less than 0.05),
    # the state hasn't learned a real reward yet. Bypass the expensive Nash solver.
    if (np.max(q_matrix_ego) - np.min(q_matrix_ego) < 0.05) and \
       (np.max(q_matrix_adv) - np.min(q_matrix_adv) < 0.05):
        return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS
    
    #### For pure stragies 
    best_ego_responses = np.argmax(q_matrix_ego, axis=0) 
    best_adv_responses = np.argmax(q_matrix_adv, axis=1) 
    
    pure_equilibria = []
    
    for adv_col in range(NUM_ACTIONS):
        ego_row = best_ego_responses[adv_col]
        if best_adv_responses[ego_row] == adv_col:
            pure_equilibria.append((ego_row, adv_col))
            
    if pure_equilibria:
        best_score = -np.inf
        best_pi_e, best_pi_a = None, None
        
        for ego_action, adv_action in pure_equilibria:
            score = q_matrix_ego[ego_action, adv_action] + q_matrix_adv[ego_action, adv_action]
            
            if score > best_score:
                best_score = score
                best_pi_e = np.zeros(NUM_ACTIONS)
                best_pi_e[ego_action] = 1.0
                best_pi_a = np.zeros(NUM_ACTIONS)
                best_pi_a[adv_action] = 1.0
                
        return best_pi_e, best_pi_a

    #### For pure stragies 

    ## SHIFT MATRICES TO POSITIVE
    min_e = np.min(q_matrix_ego)
    min_a = np.min(q_matrix_adv)
    
    shift_e = abs(min_e) + 1.0 if min_e < 0 else 0.0
    shift_a = abs(min_a) + 1.0 if min_a < 0 else 0.0

    ## INIT a Nash Game with the 2 q-tables
    noise_e = 0.0
    noise_a = 0.0

    if ADD_NOISE:
        noise_e = np.random.uniform(1e-6, 1e-5, size=q_matrix_ego.shape)
        noise_a = np.random.uniform(1e-6, 1e-5, size=q_matrix_adv.shape)
        
    game = nash.Game(q_matrix_ego + shift_e + noise_e, 
                     q_matrix_adv + shift_a + noise_a)
    
    # Mute nashpy's internal UserWarnings about degenerate games
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        try:
            # extract all equilibria
            equilibria = list(game.support_enumeration())
            
            if not equilibria:
                return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS

            best_pi_e, best_pi_a = None, None
            best_score = -np.inf

            for pi_e, pi_a in equilibria:
            
                pi_e = np.clip(pi_e, 0, 1)
                pi_e = pi_e / pi_e.sum() if pi_e.sum() > 0 else np.ones(NUM_ACTIONS)/NUM_ACTIONS
                    
                pi_a = np.clip(pi_a, 0, 1)
                pi_a = pi_a / pi_a.sum() if pi_a.sum() > 0 else np.ones(NUM_ACTIONS)/NUM_ACTIONS

                # finding global optimum
                score = (pi_e @ q_matrix_ego @ pi_a) + (pi_e @ q_matrix_adv @ pi_a)
                if score > best_score:
                    best_score = score
                    best_pi_e = pi_e
                    best_pi_a = pi_a
            
            return best_pi_e, best_pi_a

        except Exception as e:
            # Guaranteed fallback size 4 when equilibria generator is empty
            return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS

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
                _, pi_a = solve_stage_game(q_ea[s_e, s_a], q_aa[s_e, s_a])
                aa = np.argmax(pi_a) if len(pi_a) > 0 else np.random.randint(4)
                # Ego's policy from its Q‑tables
                pi_e, _ = solve_stage_game(q_ee[s_e, s_a], q_ae[s_e, s_a])
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
                pi_e_next, pi_a_next = solve_stage_game(q_ee[ns_e, ns_a], q_ae[ns_e, ns_a])
                v_ee = pi_e_next @ q_ee[ns_e, ns_a] @ pi_a_next
                v_ae = pi_e_next @ q_ae[ns_e, ns_a] @ pi_a_next

                # Adversary's view of next state's Nash equilibrium
                pi_e_next2, pi_a_next2 = solve_stage_game(q_ea[ns_e, ns_a], q_aa[ns_e, ns_a])
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

        if ep % 15000 == 0:
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