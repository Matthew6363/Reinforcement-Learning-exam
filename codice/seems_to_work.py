from environment import PacmanGridWorld
from reward_machine import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def initialize_q_function(shape, strategy="zeros"):
    if strategy == "zeros":
        return np.zeros(shape)
    elif strategy == "optimistic":
        return np.ones(shape) * 1.0
    elif strategy == "random":
        return np.random.uniform(low=0.0001, high=0.001, size=shape)
    else:
        raise ValueError("Unknown initialization strategy")

def train_qrm_sg(total_episodes=7500):
    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')

    shape = (36, 36, 5, 5, 4, 4)

    q_ee = initialize_q_function(shape, strategy="zeros")
    q_ae = initialize_q_function(shape, strategy="zeros")
    q_aa = initialize_q_function(shape, strategy="zeros")
    q_ea = initialize_q_function(shape, strategy="zeros")

    def pos_to_idx(pos): return pos[0] * 6 + pos[1]

    rm_map    = {'start': 0, 'v_1': 1, 'v_2': 2, 'v_3': 3, 'v_end': 4}
    rm_states = list(rm_map.keys())   # used for counterfactual loop

    gamma = 0.9
    alpha = 0.1

    # Epsilon decays from 0.25 → 0.05 over the first 80% of training,
    # then stays fixed. This lets exploitation gradually take over.
    start_epsilon  = 0.30
    end_epsilon    = 0.05
    decay_episodes = int(total_episodes * 0.8)

    successes       = 0
    all_ego_rewards = []
    all_adv_rewards = []

    print(f"Starting QRM-SG Training for {total_episodes} episodes...")

    for episode in range(1, total_episodes + 1):

        pos_e, pos_a = env.reset()
        state_e = rm_ego.reset()
        state_a = rm_adv.reset()

        ep_reward_e = 0.0
        ep_reward_a = 0.0

        # Linearly decay epsilon
        epsilon = max(end_epsilon,
                      start_epsilon - (start_epsilon - end_epsilon)
                      * episode / decay_episodes)

        for step in range(500):
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_e = rm_map[state_e]
            v_a = rm_map[state_a]

            # ── Action selection ────────────────────────────────────────────
            if np.random.rand() < epsilon:
                action_e = int(np.random.choice(4))
                action_a = int(np.random.choice(4))
            else:
                pi_e_ego, _       = solve_stage_game(q_ee[s_e, s_a, v_e, v_a],
                                                      q_ae[s_e, s_a, v_e, v_a])
                _,        pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a],
                                                      q_aa[s_e, s_a, v_e, v_a])
                action_e = np.argmax(pi_e_ego) # Ego picks the best response to Adv
                action_a = np.argmax(pi_a_adv) # Adversary always picks the best response (no randomness)

            # ── Environment step ────────────────────────────────────────────
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()

            ns_e, ns_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)

            # ── Counterfactual RM updates ───────────────────────────────────
            # Core of QRM-SG: update Q-values for ALL 25 RM state pairs at
            # every step, not only the pair the agents currently occupy.
            # This fills the Q-table ~25× faster and is why the paper's curve
            # converges so quickly.
            for u_e_str in rm_states:
                for u_a_str in rm_states:

                    # Simulate what reward/next-state WOULD have been
                    next_u_e_str, r_e_cf = rm_ego.simulate_step(u_e_str, labels)
                    next_u_a_str, r_a_cf = rm_adv.simulate_step(u_a_str, labels)

                    u_e  = rm_map[u_e_str];      u_a  = rm_map[u_a_str]
                    nu_e = rm_map[next_u_e_str]; nu_a = rm_map[next_u_a_str]

                    is_terminal = (next_u_e_str == 'v_end' or
                                   next_u_a_str == 'v_end' or
                                   'collision' in labels)

                    if is_terminal:
                        v_boot_ee = v_boot_ae = v_boot_ea = v_boot_aa = 0.0
                    else:
                        pi_ne_ego, pi_na_ego = solve_stage_game(
                            q_ee[ns_e, ns_a, nu_e, nu_a],
                            q_ae[ns_e, ns_a, nu_e, nu_a])
                        v_boot_ee = pi_ne_ego @ q_ee[ns_e, ns_a, nu_e, nu_a] @ pi_na_ego
                        v_boot_ae = pi_ne_ego @ q_ae[ns_e, ns_a, nu_e, nu_a] @ pi_na_ego

                        pi_ne_adv, pi_na_adv = solve_stage_game(
                            q_ea[ns_e, ns_a, nu_e, nu_a],
                            q_aa[ns_e, ns_a, nu_e, nu_a])
                        v_boot_ea = pi_ne_adv @ q_ea[ns_e, ns_a, nu_e, nu_a] @ pi_na_adv
                        v_boot_aa = pi_ne_adv @ q_aa[ns_e, ns_a, nu_e, nu_a] @ pi_na_adv

                    idx = (s_e, s_a, u_e, u_a, action_e, action_a)
                    q_ee[idx] = (1-alpha)*q_ee[idx] + alpha*(r_e_cf + gamma*v_boot_ee)
                    q_ae[idx] = (1-alpha)*q_ae[idx] + alpha*(r_a_cf + gamma*v_boot_ae)
                    q_ea[idx] = (1-alpha)*q_ea[idx] + alpha*(r_e_cf + gamma*v_boot_ea)
                    q_aa[idx] = (1-alpha)*q_aa[idx] + alpha*(r_a_cf + gamma*v_boot_aa)

            # ── Advance the real RM state ───────────────────────────────────
            next_state_e, r_e = rm_ego.step(labels)
            next_state_a, r_a = rm_adv.step(labels)

            ep_reward_e += r_e
            ep_reward_a += r_a

            pos_e, pos_a = next_pos_e, next_pos_a
            state_e, state_a = next_state_e, next_state_a

            if rm_ego.state == 'v_end' or rm_adv.state == 'v_end':
                if r_e > 0:
                    successes += 1
                break
            elif 'collision' in labels:
                break

        all_ego_rewards.append(min(ep_reward_e, 1.0))
        all_adv_rewards.append(min(ep_reward_a, 1.0))

        if episode % 100 == 0:
            win_rate = (successes / 100) * 100
            print(f"Episodes {episode-99:05d} to {episode:05d} | "
                  f"Ego Win Rate: {win_rate:.1f}%  (ε={epsilon:.3f})")
            successes = 0

    # ── Save ────────────────────────────────────────────────────────────────
    print("Training complete. Saving...")
    np.savez('q_models.npz', q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)
    np.savez('eval_results.npz',
             ego_rewards=all_ego_rewards,
             adv_rewards=all_adv_rewards)

    # ── Plot (non-overlapping 80-episode windows, paper style) ───────────────
    window    = 80
    n_windows = len(all_ego_rewards) // window
    ego_arr   = np.array(all_ego_rewards[:n_windows*window]).reshape(n_windows, window)
    adv_arr   = np.array(all_adv_rewards[:n_windows*window]).reshape(n_windows, window)
    x_plot    = np.arange(1, n_windows+1) * window - window // 2
    ego_w     = ego_arr.mean(axis=1)
    adv_w     = adv_arr.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("Task I", fontsize=13)

    ax1.plot(x_plot, ego_w, color='black', linestyle='--', linewidth=1.5, label='QRM-SG')
    ax1.set_ylabel("Reward of Ego Agent")
    ax1.set_ylim(-0.05, 1.1)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.legend(loc='lower right', fontsize=9)

    ax2.plot(x_plot, adv_w, color='black', linestyle='--', linewidth=1.5, label='QRM-SG')
    ax2.set_ylabel("Reward of Adversarial Agent")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(-0.05, 1.1)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, total_episodes)

    plt.tight_layout()
    plt.savefig("task_I_windowed.png", dpi=150, bbox_inches='tight')
    print("Plot saved to 'task_I_windowed.png'.")
    plt.show()

    return q_ee, q_ae, q_ea, q_aa

if __name__ == "__main__":
    train_qrm_sg(total_episodes=7500)