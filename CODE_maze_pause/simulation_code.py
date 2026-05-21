## EXEC FILE
## File dedicated to the QRM-SG probem execution and solving

# == MAIN PARAMETERS and libs == #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from game_parameters import * # get all env parameters


######### PROBLEM-SPECIFIC REWARD MACHINE and ENV #######################
#########################################################################
# We need to generalize the structure for any reward machine. 
# Having defined the current one of interest, please import it.

#from reward_machine_task_I import *

from environment import *
from reward_machine_maze import *

task_name_string = TASK # tag which is used in exported files

# Some requirements must be fulfilled: 
# * Reward_Machine [class]
# * solve_stage_game [function]
# View reward_machine_TaskI.py for reference.
# The environment too has to be tuned in the get_labels function.
#########################################################################


# ---------------------------------------- #
# Initialization of the Q-function (table) #
# ---------------------------------------- #
INIT_STRATEGIES = ["zeros", "optimistic", "random"] 

def initialize_q_function(shape, strategy=INIT_STRATEGIES[2]):
    '''
    This function initializes the Q-table, given its shape. It can be initialized
    with all zero (being all zero filled), optimistic (all ones) or randomly.
    It returns the table.
    '''
    if strategy == "zeros":
        return np.zeros(shape)
    elif strategy == "optimistic":
        return np.ones(shape) * 1.0
    elif strategy == "random":
        return np.random.uniform(low=0.001, high=0.01, size=shape)
        # return np.random.uniform(low=0.01, high=0.2, size=shape)
    else:
        raise ValueError("Unknown initialization strategy")


def pos_to_idx(pos): 
    '''
    Project 2D coord into a 1D number
    '''
    return pos[0] * GRID_W + pos[1]

# ------------------------------------ #
# QSG-RM solver with training episodes #
# ------------------------------------ #
def train_qrm_sg(total_episodes=1000,
                 gamma = GAMMA,
                 alpha = ALPHA,
                 start_epsilon = START_EPSILON,
                 end_epsilon = END_EPSILON,
                 decay_rate = DECAY_RATE,
                 q_shape = Q_SHAPE,
                 actions = ACTIONS,
                 step_num = STEP_NUM,
                 q_init_strategy = "zeros"):

    ## Build the environment and the two RMs
    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')
    # Notice that it's possible to define just one RM, but it's by far less clear in the end.


    ## Initialize the q-tables with given strategy and 
    q_ee = initialize_q_function(q_shape, strategy=q_init_strategy)
    q_ae = initialize_q_function(q_shape, strategy=q_init_strategy)
    q_aa = initialize_q_function(q_shape, strategy=q_init_strategy)
    q_ea = initialize_q_function(q_shape, strategy=q_init_strategy)

    rm_states_map = {} 
    ## Define the map between RM states and its indexes.
    
    rm_states_map = {'start': 0, 'v_trap': 1, 'v_lose': 2, 'v_escaped': 3, 'opened':4} 
        
    rm_states = list(rm_states_map.keys())   # used for counterfactual loop


    ## Define the number of episodes needed to tune the epsilon decay rate.
    #  In this way, we do allow epsilon to become lower the far we go with exploration
    decay_episodes = int(total_episodes * decay_rate)
    

    ## CHECK Parameters and HYSTORY
    successes       = 0
    adv_wins        = 0
    collisions_cnt  = 0
    traps_cnt       = 0
    timeouts_cnt    = 0
    all_ego_rewards = []
    all_adv_rewards = []
    history = {
        "episodes": [], "epsilon": [], 
        "ego_wr": [], "adv_wr": [], 
        "coll_r": [], "trap_r": [], "time_r": [], 
        "avg_rew_e": [], "avg_rew_a": []
    }


    print(f"Starting QRM-SG Training for {total_episodes} episodes...")


    for episode in range(1, total_episodes + 1): # For each episode...

        ## ...................... ##
        ## EPISODE INITIALIZATION ##
        ## ...................... ##

        ## Initialize the episode
        pos_e, pos_a = env.reset() # get the Agent Position (in game) reset
        state_e = rm_ego.reset()   # get the ego RM position reset -> start
        state_a = rm_adv.reset()   # get the adv RM position reset -> start

        ## Initialize ego and adv rewards for this episode
        ep_reward_e = 0.0
        ep_reward_a = 0.0

        ## Define the epsilon, considering decay rate: it's episode dependent, so 
        #  the higher the episode, the lower the randomness of actions.
        epsilon = max(end_epsilon,
                      start_epsilon - (start_epsilon - end_epsilon) * episode / decay_episodes)

        # ## EPSILON PLATEAU STRATEGY
        # plateau_episodes = int(total_episodes * 0.25) # Hold max exploration for the first 25% of training
        # active_decay_episodes = decay_episodes - plateau_episodes

        # if episode <= plateau_episodes:
        #     epsilon = start_epsilon
        # else:
        #     decay_progress = (episode - plateau_episodes) / active_decay_episodes
        #     epsilon = max(end_epsilon, start_epsilon - (start_epsilon - end_epsilon) * decay_progress)

        for step in range(step_num): # for each step of the game
            
            ## ............. ##
            ## Episode steps ##
            ## ............. ##

            ## Get the current agent position (in the game and RM)
            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_e = rm_states_map[state_e] # get the node name in the RM (ego)
            v_a = rm_states_map[state_a] # get the node name in the RM (adv)

            ## Action selection
            act_at_random = np.random.rand() < epsilon
            num_actions = len(ACTIONS)
            
            if act_at_random:
                action_e = int(np.random.choice(num_actions)) # the ego action is chosen at random
                action_a = int(np.random.choice(num_actions)) # the adv action is chosen at random
            
            else:
                # From the ego point of view
                pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a],
                                                      q_ae[s_e, s_a, v_e, v_a])
                # From the adv point of view
                pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a],
                                                      q_aa[s_e, s_a, v_e, v_a])
                ## Choose the action
                action_e = np.argmax(pi_e_ego) # Ego picks the best response to Adv
                action_a = np.argmax(pi_a_adv) # Adversary always picks the best response (no randomness)

            #print(f"Ego: {action_e}, Adv {action_a}")
            ## Update the environment based on the done action
            next_pos_e, next_pos_a = env.step(action_e, action_a) # recover where the agents are in the game
                

            ## Get the translation of the step for the RM 
            labels = env.get_labels()
            env_trapped = env.ego_trapped ## after the action is done, 
            #ego_on_wall = env.ego_on_wall
            #adv_on_wall = env.adv_on_wall

            ## Get the next reached state in the game by the two players
            next_state_e, next_state_a = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            # These are needed to get future knowledge


            ## Resume: up to now we have performed the t-step out of the current episode, by selecting
            #  the action, executing it and getting the environment aware of it. We end up with the 2D
            #  updated position for both ego and adv (next_pos_e, next_pos_a) and now we have to
            #  update the core of QRM-SG: the table. To do this, we need to consider the RM.
            #  Notice we've still no performed the movement from the RM and retrieved the current
            #  reward.

            ## ......................... ##
            ## Counterfactual RM updates ##
            ## ......................... ##
            # Core of QRM-SG: update Q-values for ALL RM state pairs at every step, not only the 
            # pair the agents currently occupy. This fills the Q-table ~25× faster and is why 
            # the paper's curve converges so quickly.
            # 
            # Actually, now we need to update the value which is associated to having done the action
            # we've done in the state we where. An action is not good by itself, but based on what's going
            # to happen after it's execution in a state. To overcome this knowledge lack (future is unknown)
            # we can simulate a new step forward. From the new hypothetical state, we need to compute 
            # the most appropriate action to do (nash eq. strategy there) that would be done in that situation.

            
            for u_e_str in rm_states: # for each ego-reward machine state (expressed as strings)
                for u_a_str in rm_states: # for each adv-reward machine state (expressed as strings)
                    
                    ## Simulate what reward/next-state WOULD have been.
                    #  We get both the state it would have been (each ego and adv) and
                    #  the reward it would have had in that case
                    next_u_e_str, r_e_cf = rm_ego.simulate_step(u_e_str, labels, env_trapped)#, ego_on_wall, adv_on_wall)
                    next_u_a_str, r_a_cf = rm_adv.simulate_step(u_a_str, labels, env_trapped)#, ego_on_wall, adv_on_wall)

                    ## Get the number version of these states
                    u_e  = rm_states_map[u_e_str]     
                    u_a  = rm_states_map[u_a_str]
                    nu_e = rm_states_map[next_u_e_str]
                    nu_a = rm_states_map[next_u_a_str]
                    
                    is_terminal = (next_u_e_str == 'v_escaped' or
                                   next_u_e_str == 'v_lose' or 
                                   next_u_a_str == 'v_escaped' or
                                   next_u_a_str == 'v_lose') # removed prematue collisio terminal state

                    ## .................................................... ##
                    ## Define the discontued cumulative reward of this case ##
                    ## .................................................... ##
                    #  Once we've reached that next state, described in comments above, we can face dirrerent 
                    #  cases. If we've reached a final case, the expected reward we'd have by following the optimal 
                    #  policy from there is zero: no improvement is expected if agents are still.
                    #  If instead the state of RM is not final, we can expect to be asking again in the future 
                    #  that question. 
                    # 
                    #  We'll need these one-step future reached states because we'll need their associated values.
                    #  Thanks to this, we do compute the optimal policy (from there), meaning that we know the 
                    #  probabilities associated to actions from there. It's all we need to compute the discounted 
                    # cumulative reward: we can multiply
                    #  * the probability that the ego does its actions from those hypothetical next state
                    #  * the probability that the adv does its actions from those hypothetical next state
                    #  * the q-table of those hypothetical next states (the one we'd reach) 
                    # 

                    # Case: if we're in the base case
                    if is_terminal:
                        v_boot_ee = v_boot_ae = v_boot_ea = v_boot_aa = 0.0
                    
                    # Case: otherwise
                    else:
                        # from the ego point of view
                        pi_ne_ego, pi_na_ego = solve_stage_game(
                                                                 q_ee[next_state_e, next_state_a, nu_e, nu_a],
                                                                 q_ae[next_state_e, next_state_a, nu_e, nu_a]
                                                               )
                        v_boot_ee = pi_ne_ego @ q_ee[next_state_e, next_state_a, nu_e, nu_a] @ pi_na_ego
                        v_boot_ae = pi_ne_ego @ q_ae[next_state_e, next_state_a, nu_e, nu_a] @ pi_na_ego

                        # from the adv point of view
                        pi_ne_adv, pi_na_adv = solve_stage_game(
                                                                 q_ea[next_state_e, next_state_a, nu_e, nu_a],
                                                                 q_aa[next_state_e, next_state_a, nu_e, nu_a]
                                                                )
                        v_boot_ea = pi_ne_adv @ q_ea[next_state_e, next_state_a, nu_e, nu_a] @ pi_na_adv
                        v_boot_aa = pi_ne_adv @ q_aa[next_state_e, next_state_a, nu_e, nu_a] @ pi_na_adv
                    
                    # v_boot_ea -> Disc. cumul. rew. the agent will obtain (from adv point of view) if 
                    #              it assumes they both will play optimally from there on
                    # v_boot_ee -> Disc. cumul. rew. the agent will obtain (from its point of view) if 
                    #              it assumes they both will play optimally from there on
                    # ...

                    idx = (s_e, s_a, u_e, u_a, action_e, action_a)
                    
                    ## Update the q-table in this case, using the computed dcr.
                    q_ee[idx] = (1-alpha)*q_ee[idx] + alpha*(r_e_cf + gamma*v_boot_ee)
                    q_ae[idx] = (1-alpha)*q_ae[idx] + alpha*(r_a_cf + gamma*v_boot_ae)
                    q_ea[idx] = (1-alpha)*q_ea[idx] + alpha*(r_e_cf + gamma*v_boot_ea)
                    q_aa[idx] = (1-alpha)*q_aa[idx] + alpha*(r_a_cf + gamma*v_boot_aa)


            ## ............................................... ##
            ## Going back to our real case: what's our reward? ##
            ## ............................................... ##

            ## Get the real RM next state and current reward
            next_state_e, r_e = rm_ego.step(labels, env_trapped)#, ego_on_wall, adv_on_wall)
            next_state_a, r_a = rm_adv.step(labels, env_trapped)#, ego_on_wall, adv_on_wall)

            ## Update the episode reward scalar
            ep_reward_e += r_e
            ep_reward_a += r_a


            ## .......................................... ##
            ## Setting up for next step: update variables ##
            ## .......................................... ##
                 
            ## Set the current pos of the RM as the one reached
            pos_e, pos_a = next_pos_e, next_pos_a


            ## Set the current pos in the game as the one reached
            state_e, state_a = next_state_e, next_state_a

            env.clear_turn_flags()

            ## Case: one between ego and adv reached the end?
            if 'collision' in labels:
                collisions_cnt += 1
                adv_wins += 1
                break

            # 2. Controlla le vittorie effettive degli agenti
            if rm_ego.state == 'v_escaped':
                successes += 1
                break
                
            if rm_adv.state == 'v_lose':
                if 'trapped' in labels:
                    traps_cnt += 1
                else:
                    adv_wins += 1
                break
                
        
        # TIMEOUT case, add a penalty to last action
        else: 
            timeouts_cnt += 1
            r_e += TIMEOUT_PENALTY  
            all_ego_rewards.append(ep_reward_e + TIMEOUT_PENALTY)
            all_adv_rewards.append(ep_reward_a)
            #continue


        ## Back to the episode,having done all steps, we update the rewards
        all_ego_rewards.append(ep_reward_e)
        all_adv_rewards.append(ep_reward_a)

        ## Print the situation every 100 episodes.
        if episode % 100 == 0:
            ego_wr = (successes / 100) * 100
            adv_wr = (adv_wins / 100) * 100
            coll_r = (collisions_cnt / 100) * 100
            trap_r = (traps_cnt / 100) * 100 
            time_r = (timeouts_cnt / 100) * 100
            
            # Calculate mean reward for the last 100 episodes
            avg_rew_e = np.mean(all_ego_rewards[-100:])
            avg_rew_a = np.mean(all_adv_rewards[-100:])

            print(f"Episodes {episode-99:05d}-{episode:05d} | ε: {epsilon:.3f}")
            print(f"  > Win Rate  | Ego: {ego_wr:4.1f}%  Adv: {adv_wr:4.1f}%  Coll: {coll_r:4.1f}%  Trap: {trap_r:4.1f}%  Timeout: {time_r:4.1f}%")
            print(f"  > Avg Rew   | Ego: {avg_rew_e:4.2f}   Adv: {avg_rew_a:4.2f}")
            print("-" * 50)


            file_path = f"../EXPORT/{TASK}_saved_percentages.npz"
            
            history["episodes"].append(episode)
            history["epsilon"].append(epsilon)
            history["ego_wr"].append(ego_wr)
            history["adv_wr"].append(adv_wr)
            history["coll_r"].append(coll_r)
            history["trap_r"].append(trap_r)
            history["time_r"].append(time_r)
            history["avg_rew_e"].append(avg_rew_e)
            history["avg_rew_a"].append(avg_rew_a)

            np.savez(file_path, **history)
            
            # Reset window counters
            successes = 0
            adv_wins = 0
            collisions_cnt = 0
            timeouts_cnt = 0
            traps_cnt = 0
        
        # Save every N episodes (change to 100 if you want every single step)
        if episode % SAVE_EACH == 0:
            ckpt_path = f'../EXPORT/q_models_{task_name_string}_ep{episode}.npz'
            np.savez(ckpt_path, q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)



    print("\nTraining complete. Saving...")
    np.savez(f'../EXPORT/q_models_{task_name_string}.npz', q_ee=q_ee, q_ae=q_ae, q_ea=q_ea, q_aa=q_aa)
    np.savez(f'../EXPORT/eval_results_{task_name_string}.npz',
             ego_rewards=all_ego_rewards,
             adv_rewards=all_adv_rewards)

    ## Plot evolution
    window    = 80
    n_windows = len(all_ego_rewards) // window
    ego_arr   = np.array(all_ego_rewards[:n_windows*window]).reshape(n_windows, window)
    adv_arr   = np.array(all_adv_rewards[:n_windows*window]).reshape(n_windows, window)
    x_plot    = np.arange(1, n_windows+1) * window - window // 2
    ego_w     = ego_arr.mean(axis=1)
    adv_w     = adv_arr.mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    fig.suptitle("Task III", fontsize=13)

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
    plt.savefig(f"../EXPORT/{task_name_string}_windowed.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved to '../EXPORT/{task_name_string}_windowed.png'.")
    plt.show()

    return q_ee, q_ae, q_ea, q_aa


if __name__ == "__main__":
    train_qrm_sg(total_episodes=EPISODES)