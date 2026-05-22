## LIB FILE
## File dedicated to the functional definitions of Reward Machines


# == MAIN PARAMETERS and libs == #
import nashpy as nash
import numpy as np
import datetime
from game_parameters import *

DEBUG = False


def log_error_to_file(error_msg, q_e=None, q_a=None):
    with open("../LOGs/log_error.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] ERROR: {error_msg}\n")
        if q_e is not None:
            f.write(f"Matrix Ego:\n{q_e}\n")
        if q_a is not None:
            f.write(f"Matrix Adv:\n{q_a}\n")
        f.write("-" * 50 + "\n")

# ====================== #
# Reward Machine (Class) #
# ====================== #

class Reward_Machine():

    def __init__(self, agent_type):
        '''
        Initialize function for the RM object/instance. Requires the 
        information of who's the current agent considered, between the two (ref. paper)
        players.
        
        [ Input ] 
          * agent_type : "ego" or "adv".
        '''

        ## Name check
        if agent_type not in ["ego", "adv"]:
            raise TypeError

        self.agent_type = agent_type # set the inner agent type
        self.state = 'start'         # go set the RM at starting point 
        

    def reset(self):
        '''
        Reset the RM, by making the current state the initial state again
        '''

        self.state = 'start'
        return self.state

    
    # def simulate_step(self, hypothetical_state, labels, env_trapped = False, ego_on_wall = False, adv_on_wall = False):
    #     '''
    #     Function allowing the simulation of the RM rewards got if the game state 
    #     would have been the "hypothetical_state" one instead of the one internally saved.
    #     The function temporarily changes the current with the hypothetical one, and computes
    #     a step just to see the reward we'd obtain. The state is then set again to the original one
        
    #     [Input]
    #     * hypothetical_state : (RM) state we're interested in the reward of
    #     * labels : RM edge label to follow 

    #     [Output]
    #     * next_state : the state we'd reached by there 
    #     * reward : the reward it'd have granted us
        
    #     This Allows the algorithm to test counterfactual states without breaking the real game state. 
    #     '''
        
    #     # 1. Save the actual physical state
    #     current_real_state = self.state
        
    #     # 2. Temporarily overwrite it with the hypothetical state
    #     self.state = hypothetical_state
        
    #     # 3. See what the reward and next state WOULD have been
    #     next_state, reward = self.step(labels, env_trapped, ego_on_wall,adv_on_wall)
        
    #     # 4. Revert the machine back to reality
    #     self.state = current_real_state
        
    #     return next_state, reward
        
    
    # def step(self, labels, env_trapped=False, ego_on_wall = False, adv_on_wall = False):
    #     reward = 0.0 

    #     if ego_on_wall:
    #         reward = WALL_HIT_PENALTY
            
    #     elif adv_on_wall:
    #         reward = WALL_HIT_PENALTY

    #     # and 
    #     if self.state == 'start':
                
    #         if 'collision' in labels:
    #             self.state = 'v_lose'
    #             if self.agent_type == 'adv': reward = 1
    #             if self.agent_type == 'ego': reward = -1 # Fear the adversary
                
    #         elif 'trapped' in labels:
    #             if TRAPS_DO_STOP_FOR_A_TURN:
    #                 self.state = 'v_trap'
    #                 if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
    #             else:
    #                 self.state = 'v_lose' # Instant death
    #                 if self.agent_type == 'adv': reward = 1
    #                 if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
            
    #         elif 'key' in labels:
    #             self.state = "opened"
    #             if self.agent_type == 'ego': reward = KEY_REWARD


    #     if self.state == 'opened':
    #         if 'escaped!' in labels:        
    #             self.state = 'v_escaped'
    #             if self.agent_type == 'ego': reward = WINNING_MEGA_REWARD
    #             #print("WIN")
                
    #         elif 'collision' in labels:
    #             self.state = 'v_lose'
    #             if self.agent_type == 'adv': reward = 1
    #             if self.agent_type == 'ego': reward = -1 # Fear the adversary
                
    #         elif 'trapped' in labels:
    #             if TRAPS_DO_STOP_FOR_A_TURN:
    #                 self.state = 'v_trap'
    #                 if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
    #             else:
    #                 self.state = 'v_lose' # Instant death
    #                 if self.agent_type == 'adv': reward = 1
    #                 if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
            
        
    #     elif self.state == 'v_trap':
    #         if TRAPS_DO_STOP_FOR_A_TURN:
    #             if 'collision' in labels:
    #                 self.state = 'v_lose'
    #                 if self.agent_type == 'adv': reward = 1 
    #                 if self.agent_type == 'ego': reward = WINNING_MEGA_REWARD
    #             elif env_trapped == False: 
    #                 self.state = 'start'
    #         else:
    #             # This block is functionally dead code now (which is good), 
    #             # but left in case of counterfactual simulation leaks.
    #             self.state = 'v_lose'
    #             if self.agent_type == 'adv': reward = 1 
    #             if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD

    #     return self.state, reward
    
    def simulate_step(self, hypothetical_state, labels, env_trapped=False):
        current_real_state = self.state
        self.state = hypothetical_state
        next_state, reward = self.step(labels, env_trapped)
        self.state = current_real_state
        return next_state, reward
        
    def step(self, labels, env_trapped=False):
        reward = 0.0 

        if self.agent_type == "ego":
            reward += EGO_LIVING_PENALTY
        if self.agent_type == "adv":
            reward += ADV_LIVING_PENALTY

        # FIXED: Only penalize the specific agent that hit the wall
        if 'ego_on_wall' in labels and self.agent_type == 'ego':
            reward += WALL_HIT_PENALTY
        if 'adv_on_wall' in labels and self.agent_type == 'adv':
            reward += WALL_HIT_PENALTY

        if self.state == 'start':
            if 'collision' in labels:
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward += ADV_CATCH_REWARD
                if self.agent_type == 'ego': reward += EGO_CATCHED_NEG_REWARD # Fear the adversary
                
            elif 'trapped' in labels:
                if TRAPS_DO_STOP_FOR_A_TURN:
                    self.state = 'v_trap'
                    if self.agent_type == 'ego': reward += TRAP_NEGATIVE_REWARD
                else:
                    self.state = 'v_lose' 
                    if self.agent_type == 'adv': reward += ADV_CATCH_REWARD
                    if self.agent_type == 'ego': reward += TRAP_NEGATIVE_REWARD
            
            elif 'key' in labels:
                self.state = "opened"
                if self.agent_type == 'ego': reward += KEY_REWARD

        elif self.state == 'opened':
            if 'escaped!' in labels:        
                self.state = 'v_escaped'
                if self.agent_type == 'ego': reward += WINNING_MEGA_REWARD
                
            elif 'collision' in labels:
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward += ADV_CATCH_REWARD
                if self.agent_type == 'ego': reward += EGO_CATCHED_NEG_REWARD
                
            elif 'trapped' in labels:
                if TRAPS_DO_STOP_FOR_A_TURN:
                    self.state = 'v_trap'
                    if self.agent_type == 'ego': reward += TRAP_NEGATIVE_REWARD
                else:
                    self.state = 'v_lose' 
                    if self.agent_type == 'adv': reward += ADV_TRAP_REWARD
                    if self.agent_type == 'ego': reward += TRAP_NEGATIVE_REWARD
        
        elif self.state == 'v_trap':
            if TRAPS_DO_STOP_FOR_A_TURN:
                if 'collision' in labels:
                    self.state = 'v_lose'
                    if self.agent_type == 'adv': reward += ADV_CATCH_REWARD
                    if self.agent_type == 'ego': reward += EGO_CATCHED_NEG_REWARD
                elif env_trapped == False: 
                    self.state = 'start'
            else:
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward += ADV_TRAP_REWARD
                if self.agent_type == 'ego': reward += TRAP_NEGATIVE_REWARD

        return self.state, reward            
            

# ============= #
# QRM-SG solver #
# ============= #
  
import warnings # Add this at the top of the file

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
    


    
        