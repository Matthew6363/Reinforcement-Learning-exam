from environment import PacmanGridWorld
from reward_machine import *
import numpy as np
import matplotlib.pyplot as plt        # ADD 1/4
import matplotlib.ticker as ticker     # ADD 1/4

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
    
    # 5 states for RM (start, v1, v2, v3, v_end)
    shape = (36, 36, 5, 5, 4, 4)
    
    # Initialize using tiny random noise to stabilize the Nash solver
    q_ee = initialize_q_function(shape, strategy="zeros")
    q_ae = initialize_q_function(shape, strategy="zeros")
    
    q_aa = initialize_q_function(shape, strategy="zeros")
    q_ea = initialize_q_function(shape, strategy="zeros")
    
    def pos_to_idx(pos): return pos[0]*6 + pos[1]
    
    # Ensure keys match your Reward_Machine class state names
    rm_map = {'start': 0, 'v_1': 1, 'v_2': 2, 'v_3': 3, 'v_end': 4}
    
    gamma = 0.9 
 
    alpha = 0.1 
    successes = 0
    # before the episode loop
    start_epsilon = 0.10
    end_epsilon   = 0.005
    decay_episodes = int(total_episodes * 0.8)
        
    # Initialize evaluation variables
    eval_episodes = 10  # Number of evaluation episodes (adjust as needed)
    cumulative_reward_e = 0.0
    cumulative_reward_a = 0.0
    eval_episodes_xaxis = []
    eval_ego_rewards = []
    eval_adv_rewards = []
    
    print(f"Starting Decentralized QRM-SG Training for {total_episodes} episodes...")
    
    all_ego_rewards = []   
    all_adv_rewards = []   
    
    for episode in range(1, total_episodes + 1):
        pos_e, pos_a = env.reset()
        state_e = rm_ego.reset()  
        state_a = rm_adv.reset()  
        ep_reward_e = 0.0          
        ep_reward_a = 0.0          
        epsilon = max(end_epsilon,
              start_epsilon - (start_epsilon - end_epsilon) * episode / decay_episodes)
        
        # Increased steps to 500 to allow for complex Case Study I pathing
        for step in range(1000):
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            #print(state_e, '1')  # Use string state for printing
            v_e = rm_map[state_e]  # Integer index
            v_a = rm_map[state_a]  # Integer index
            #print(v_e, '2')
            # --- 1. DECENTRALIZED ACTION SELECTION ---
            if np.random.rand() < epsilon:
                action_e = int(np.random.choice(4))
                action_a = int(np.random.choice(4))
            else:
                
                pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
                action_e = np.argmax(pi_e_ego)
                #action_e = np.random.choice(4, p=pi_e_ego)
                
                pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
                action_a = np.argmax(pi_a_adv)
                #action_a = np.random.choice(4, p=pi_a_adv)
                
            # --- 2. ENVIRONMENT STEP ---
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            # Step Reward Machines
            next_state_e, r_e = rm_ego.step(labels) 
            next_state_a, r_a = rm_adv.step(labels)  
            ep_reward_e += r_e                         
            ep_reward_a += r_a                         
            #print(next_state_e, '3')
            
            ns_e, ns_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            nv_e = rm_map[next_state_e]  # Integer index
            nv_a = rm_map[next_state_a]  # Integer index
            

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
            # Update physical state
            
            pos_e, pos_a = next_pos_e, next_pos_a
            state_e, state_a = next_state_e, next_state_a  # Update string states
            
            
            # --- 4. TERMINATION CHECK ---
            if rm_ego.state == 'v_end' or rm_adv.state == 'v_end':
                # Only count a success if the Ego agent actually got the point!
                if r_e > 0:
                    successes += 1
                break
            # Catch premature physical collisions before power bases are touched
            elif 'collision' in labels:
                break
            
        if episode % 1000 == 0:
            filename = f'q_models_ep{episode}.npz'
            np.savez(filename, q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)
            print(f"Checkpoint saved: {filename}")
            
        if episode % 100 == 0:
            win_rate = (successes / 100) * 100
            print(f"Episodes {episode-99:04d} to {episode:04d} | Ego Agent Win Rate: {win_rate:.1f}%")
            successes = 0
        all_ego_rewards.append(min(ep_reward_e, 1.0))   # ADD 3/4
        all_adv_rewards.append(min(ep_reward_a, 1.0))   # ADD 3/4
    
    # --- EVALUATION PHASE ---
    print("\nRunning Final Evaluation...")
    cumulative_reward_e = 0.0
    cumulative_reward_a = 0.0
    
    for eval_ep in range(eval_episodes):
        # 1. Reset everything for the new episode
        pos_e, pos_a = env.reset()
        state_e = rm_ego.reset()
        state_a = rm_adv.reset()
        
        ep_reward_e = 0.0
        ep_reward_a = 0.0
        
        # 2. Run the episode (max 500 steps to match training)
        for step in range(500):
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_e = rm_map[state_e]
            v_a = rm_map[state_a]
            
            # 3. Calculate the Nash Equilibrium for the current state
            pi_e_ego, _ = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
            _, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
            
            # 4. Action Selection (Sample from the Nash distribution)
            action_e = int(np.random.choice(4, p=pi_e_ego))
            action_a = int(np.random.choice(4, p=pi_a_adv))
            
            # 5. Step the environment and reward machines
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            next_state_e, r_e = rm_ego.step(labels)
            next_state_a, r_a = rm_adv.step(labels)
            
            ep_reward_e += r_e
            ep_reward_a += r_a
            
            pos_e, pos_a = next_pos_e, next_pos_a
            state_e, state_a = next_state_e, next_state_a
            
            # 6. Termination Check
            if rm_ego.state == 'v_end' or rm_adv.state == 'v_end':
                break
            elif 'collision' in labels:
                break
                
        # Add episode rewards to the total
        cumulative_reward_e += ep_reward_e
        cumulative_reward_a += ep_reward_a
    
    # Calculate averages
    avg_eval_reward_e = cumulative_reward_e / eval_episodes
    avg_eval_reward_a = cumulative_reward_a / eval_episodes
    
    print(f"Final Evaluation | Avg Reward -> Ego: {avg_eval_reward_e:.2f} | Adv: {avg_eval_reward_a:.2f}")
    
    eval_episodes_xaxis.append(total_episodes)
    eval_ego_rewards.append(avg_eval_reward_e)
    eval_adv_rewards.append(avg_eval_reward_a)
    
    print("Training Complete. Saving models and data...")
    
    np.savez('q_models.npz', q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)
    np.savez('eval_results.npz', 
             episodes=eval_episodes_xaxis, 
             ego_rewards=eval_ego_rewards, 
             adv_rewards=eval_adv_rewards)
    
    print("Files saved: 'q_models.npz' and 'eval_results.npz'.")
    
    # ADD 4/4 — non-overlapping 80-episode window plot (matches paper style)
    window = 80
    n_windows = len(all_ego_rewards) // window
    ego_arr = np.array(all_ego_rewards[:n_windows * window]).reshape(n_windows, window)
    adv_arr = np.array(all_adv_rewards[:n_windows * window]).reshape(n_windows, window)
    x_plot       = np.arange(1, n_windows + 1) * window - window // 2
    ego_windowed = ego_arr.mean(axis=1)
    adv_windowed = adv_arr.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("Task I", fontsize=13)
    ax1.plot(x_plot, ego_windowed, color='black', linestyle='--', linewidth=1.5, label='QRM-SG')
    ax1.set_ylabel("Reward of Ego Agent")
    ax1.set_ylim(-0.05, 1.1)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.legend(loc='lower right', fontsize=9)
    ax2.plot(x_plot, adv_windowed, color='black', linestyle='--', linewidth=1.5, label='QRM-SG')
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