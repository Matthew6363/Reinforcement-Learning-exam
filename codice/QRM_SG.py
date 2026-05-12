from environment import PacmanGridWorld
from reward_machine import *
import numpy as np

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
    epsilon = 0.25 
    alpha = 0.1 
    successes = 0
    
    # Initialize evaluation variables
    eval_episodes = 10  # Number of evaluation episodes (adjust as needed)
    cumulative_reward_e = 0.0
    cumulative_reward_a = 0.0
    eval_episodes_xaxis = []
    eval_ego_rewards = []
    eval_adv_rewards = []
    
    print(f"Starting Decentralized QRM-SG Training for {total_episodes} episodes...")
    
    for episode in range(1, total_episodes + 1):
        pos_e, pos_a = env.reset()
        state_e = rm_ego.reset()  # String state
        state_a = rm_adv.reset()  # String state
        
        # Increased steps to 500 to allow for complex Case Study I pathing
        for step in range(500):
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
                # Ego agent decides action based on its own Q-estimates
                pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
                action_e = np.argmax(pi_e_ego)
                
                # Adv agent decides action based on its own separate Q-estimates
                pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
                action_a = np.argmax(pi_a_adv)
                
            # --- 2. ENVIRONMENT STEP ---
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            # Step Reward Machines
            next_state_e, r_e = rm_ego.step(labels)  # String state
            next_state_a, r_a = rm_adv.step(labels)  # String state
            #print(next_state_e, '3')
            
            ns_e, ns_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            nv_e = rm_map[next_state_e]  # Integer index
            nv_a = rm_map[next_state_a]  # Integer index
            
            # Ego updates expectations based on Ego's brain
            pi_next_e_ego, pi_next_a_ego = solve_stage_game(q_ee[ns_e, ns_a, nv_e, nv_a], q_ae[ns_e, ns_a, nv_e, nv_a])
            v_boot_ee = pi_next_e_ego @ q_ee[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_ego
            v_boot_ae = pi_next_e_ego @ q_ae[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_ego
                
            # Adv updates expectations based on Adv's brain
            pi_next_e_adv, pi_next_a_adv = solve_stage_game(q_ea[ns_e, ns_a, nv_e, nv_a], q_aa[ns_e, ns_a, nv_e, nv_a])
            v_boot_ea = pi_next_e_adv @ q_ea[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_adv
            v_boot_aa = pi_next_e_adv @ q_aa[ns_e, ns_a, nv_e, nv_a] @ pi_next_a_adv  
            
            # Bellman Updates for Ego's Brain
            q_ee[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ee[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_e + gamma * v_boot_ee)
            q_ae[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ae[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_a + gamma * v_boot_ae)
            
                                        
            # Bellman Updates for Adv's Brain
            q_ea[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_ea[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_e + gamma * v_boot_ea)
            q_aa[s_e, s_a, v_e, v_a, action_e, action_a] = (1 - alpha) * q_aa[s_e, s_a, v_e, v_a, action_e, action_a] + alpha * (r_a + gamma * v_boot_aa)
            # Update physical state
            
            pos_e, pos_a = next_pos_e, next_pos_a
            state_e, state_a = next_state_e, next_state_a  # Update string states
            
            
            # --- 4. TERMINATION CHECK ---
            if rm_ego.state == 'v_end':
                successes += 1
                break
            elif rm_adv.state == 'v_end':
                break
            
        if episode % 100 == 0:
            win_rate = (successes / 500) * 100
            print(f"Episodes {episode-99:04d} to {episode:04d} | Ego Agent Win Rate: {win_rate:.1f}%")
            successes = 0
    
    # --- EVALUATION PHASE (moved outside training loop) ---
    # Reset cumulative rewards for evaluation
    cumulative_reward_e = 0.0
    cumulative_reward_a = 0.0
    
    for eval_ep in range(eval_episodes):
        # Placeholder: Run an evaluation episode (replace with your actual eval logic)
        # Example: Simulate or run env.step with greedy actions, accumulate r_e and r_a
        # For now, assume some dummy rewards; integrate your evaluation code here
        dummy_r_e = 1.0  # Replace with actual reward from eval episode
        dummy_r_a = 0.5  # Replace with actual reward from eval episode
        cumulative_reward_e += dummy_r_e
        cumulative_reward_a += dummy_r_a
    
    avg_eval_reward_e = cumulative_reward_e / eval_episodes
    avg_eval_reward_a = cumulative_reward_a / eval_episodes
    
    print(f"Evaluation | Avg Reward -> Ego: {avg_eval_reward_e:.2f} | Adv: {avg_eval_reward_a:.2f}")
    
    eval_episodes_xaxis.append(total_episodes)  # Or append episode numbers as needed
    eval_ego_rewards.append(avg_eval_reward_e)
    eval_adv_rewards.append(avg_eval_reward_a)
    
    print("Training Complete. Saving models and data...")
    
    np.savez('q_models.npz', q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)
    np.savez('eval_results.npz', 
             episodes=eval_episodes_xaxis, 
             ego_rewards=eval_ego_rewards, 
             adv_rewards=eval_adv_rewards)
    
    print("Files saved: 'q_models.npz' and 'eval_results.npz'.")
    
    return q_ee, q_ae, q_ea, q_aa

if __name__ == "__main__":
    train_qrm_sg(total_episodes=10000)