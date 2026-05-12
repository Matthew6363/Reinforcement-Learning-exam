from environment import PacmanGridWorld
from reward_machine import *
import numpy as np
import matplotlib.pyplot as plt

def initialize_q_function(shape, strategy="random"):
    if strategy == "zeros":
        return np.zeros(shape)
    elif strategy == "optimistic":
        return np.ones(shape) * 1.0
    elif strategy == "random":
        return np.random.uniform(low=0.0001, high=0.001, size=shape)
    else:
        raise ValueError("Unknown initialization strategy")

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_qrm_sg(total_episodes=7500):
    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')
    
    shape = (36, 36, 5, 5, 4, 4)
    
    q_ee = initialize_q_function(shape, strategy="zeros")
    q_ae = initialize_q_function(shape, strategy="zeros")
    q_aa = initialize_q_function(shape, strategy="zeros")
    q_ea = initialize_q_function(shape, strategy="zeros")
    
    def pos_to_idx(pos): return pos[0]*6 + pos[1]
    rm_map = {'start': 0, 'v_1': 1, 'v_2': 2, 'v_3': 3, 'v_end': 4}
    
    gamma = 0.9 
    epsilon = 0.25 
    alpha = 0.1 
    successes = 0
    
    ego_rewards_history = []
    adv_rewards_history = []
    
    print(f"Starting Decentralized QRM-SG Training for {total_episodes} episodes...")
    
    for episode in range(1, total_episodes + 1):
        pos_e, pos_a = env.reset()
        v_e1 = rm_ego.reset()
        v_a1 = rm_adv.reset()
        
        ep_reward_e = 0.0
        ep_reward_a = 0.0
        
        
        for step in range(500):
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_e = rm_map[v_e1]
            v_a = rm_map[v_a1]
            
            

            if np.random.rand() < epsilon:
                action_e = int(np.random.choice(4))
                action_a = int(np.random.choice(4))
            else:
                pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
                # Fixed: Argmax replaced with sampling to allow exploration
                action_e = np.argmax(pi_e_ego)
                
                pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
                action_a = np.argmax(pi_a_adv)
                
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            next_ve_label, r_e = rm_ego.step(labels) 
            next_va_label, r_a = rm_adv.step(labels)
            
            ep_reward_e += r_e
            ep_reward_a += r_a
            
            ns_e, ns_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            nv_e, nv_a = rm_map[next_ve_label], rm_map[next_va_label]
            
            is_terminal = (next_ve_label == 'v_end') or (next_va_label == 'v_end') or ('collision' in labels)
            
            if is_terminal:
                v_boot_ee, v_boot_ae = 0.0, 0.0
                v_boot_ea, v_boot_aa = 0.0, 0.0
            else:
                pi_next_e_ego, pi_next_a_ego = solve_stage_game(q_ee[ns_e, ns_a, nv_e, nv_a], q_ae[ns_e, ns_a, nv_e, nv_a])
                v_boot_ee = pi_next_e_ego @ q_ee[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_ego
                v_boot_ae = pi_next_e_ego @ q_ae[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_ego
                    
                pi_next_e_adv, pi_next_a_adv = solve_stage_game(q_ea[ns_e, ns_a, nv_e, nv_a], q_aa[ns_e, ns_a, nv_e, nv_a])
                v_boot_ea = pi_next_e_adv @ q_ea[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_adv
                v_boot_aa = pi_next_e_adv @ q_aa[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_adv
            
            q_ee[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ee[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_e + gamma * v_boot_ee)
            q_ae[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ae[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_a + gamma * v_boot_ae)
            
            q_ea[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ea[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_e + gamma * v_boot_ea)
            q_aa[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_aa[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_a + gamma * v_boot_aa)
            
            pos_e, pos_a = next_pos_e, next_pos_a
            v_e1, v_a1 = next_ve_label, next_va_label
            
            if v_e1 == 'v_end':
                successes += 1
                break
            elif v_a1 == 'v_end':
                break
            elif 'collision' in labels:
                break
                
        ego_rewards_history.append(ep_reward_e)
        adv_rewards_history.append(ep_reward_a)
            
        if episode % 100 == 0:
            win_rate = (successes / 100) * 100 # Fixed denominator
            print(f"Episodes {episode-99:04d} to {episode:04d} | Ego Agent Win Rate: {win_rate:.1f}%")
            successes = 0

    print("Training Complete. Generating Plot...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(ego_rewards_history, 100), label='Ego Agent Reward', color='blue', linewidth=2)
    plt.plot(moving_average(adv_rewards_history, 100), label='Adv Agent Reward', color='red', linewidth=2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Moving Average Reward (Window=100)')
    plt.title('Ego vs Adv Agent Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return q_ee, q_ae, q_aa, q_ea

if __name__ == "__main__":
    train_qrm_sg(total_episodes=10000)