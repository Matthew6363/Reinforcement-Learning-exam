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

    
    def simulate_step(self, hypothetical_state, labels, env_trapped = False, ego_on_wall = False, adv_on_wall = False):
        '''
        Function allowing the simulation of the RM rewards got if the game state 
        would have been the "hypothetical_state" one instead of the one internally saved.
        The function temporarily changes the current with the hypothetical one, and computes
        a step just to see the reward we'd obtain. The state is then set again to the original one
        
        [Input]
        * hypothetical_state : (RM) state we're interested in the reward of
        * labels : RM edge label to follow 

        [Output]
        * next_state : the state we'd reached by there 
        * reward : the reward it'd have granted us
        
        This Allows the algorithm to test counterfactual states without breaking the real game state. 
        '''
        
        # 1. Save the actual physical state
        current_real_state = self.state
        
        # 2. Temporarily overwrite it with the hypothetical state
        self.state = hypothetical_state
        
        # 3. See what the reward and next state WOULD have been
        next_state, reward = self.step(labels, env_trapped)
        
        # 4. Revert the machine back to reality
        self.state = current_real_state
        
        return next_state, reward
        
    
    def step(self, labels, env_trapped=False, ego_on_wall = False, adv_on_wall = False):
        reward = 0.0 

        if ego_on_wall:
            reward = WALL_HIT_PENALTY
            
        elif adv_on_wall:
            reward = WALL_HIT_PENALTY

        # and 
        if self.state == 'start':
                
            if 'collision' in labels:
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward = 1
                if self.agent_type == 'ego': reward = -1 # Fear the adversary
                
            elif 'trapped' in labels:
                if TRAPS_DO_STOP_FOR_A_TURN:
                    self.state = 'v_trap'
                    if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
                else:
                    self.state = 'v_lose' # Instant death
                    if self.agent_type == 'adv': reward = 1
                    if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
            
            elif 'key' in labels:
                self.state = "opened"
                if self.agent_type == 'ego': reward = KEY_REWARD


        if self.state == 'opened':
            if 'escaped!' in labels:        
                self.state = 'v_escaped'
                if self.agent_type == 'ego': reward = WINNING_MEGA_REWARD
                #print("WIN")
                
            elif 'collision' in labels:
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward = 1
                if self.agent_type == 'ego': reward = -1 # Fear the adversary
                
            elif 'trapped' in labels:
                if TRAPS_DO_STOP_FOR_A_TURN:
                    self.state = 'v_trap'
                    if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
                else:
                    self.state = 'v_lose' # Instant death
                    if self.agent_type == 'adv': reward = 1
                    if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD
            
        
        elif self.state == 'v_trap':
            if TRAPS_DO_STOP_FOR_A_TURN:
                if 'collision' in labels:
                    self.state = 'v_lose'
                    if self.agent_type == 'adv': reward = 1 
                    if self.agent_type == 'ego': reward = WINNING_MEGA_REWARD
                elif env_trapped == False: 
                    self.state = 'start'
            else:
                # This block is functionally dead code now (which is good), 
                # but left in case of counterfactual simulation leaks.
                self.state = 'v_lose'
                if self.agent_type == 'adv': reward = 1 
                if self.agent_type == 'ego': reward = TRAP_NEGATIVE_REWARD

        return self.state, reward            
            

# ============= #
# QRM-SG solver #
# ============= #
  
def solve_stage_game(q_matrix_ego, q_matrix_adv, 
                     agent_actions = ['up', 'down', 'left', 'right'],
                     debug = DEBUG
                     ):
    '''
    This function is about solving the game, by updating the tabular q functions (matrices) 
    for both ego and adv as paper stated. This return STRATEGIES (policies) given the q-tables.
    '''

    NUM_ACTIONS = len(agent_actions)
    
    ## EMPTY Q-Tables
    if np.all(q_matrix_ego == 0) and np.all(q_matrix_adv == 0):
        return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS
    
    ## SHIFT MATRICES TO POSITIVE (Nash Invariance)
    # nashpy solvers crash with negative payoffs. Shifting by a constant preserves the equilibrium.
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
    
    try:
        # Use support_enumeration (more mathematically stable for grid worlds)
        if LEMKE_HOWSON:
            random_label = np.random.randint(0, NUM_ACTIONS) # FIXED bounds
            pi_e, pi_a = game.lemke_howson(initial_dropped_label=random_label)

            if len(pi_e) < NUM_ACTIONS:
                new_pi_e = np.zeros(NUM_ACTIONS)
                new_pi_e[:len(pi_e)] = pi_e
                pi_e = new_pi_e
                
            if len(pi_a) < NUM_ACTIONS:
                new_pi_a = np.zeros(NUM_ACTIONS)
                new_pi_a[:len(pi_a)] = pi_a
                pi_a = new_pi_a

        else:
            equilibria = game.support_enumeration() 
            pi_e, pi_a = next(equilibria) 
        
        # Normalize and clean probabilities
        pi_e = np.clip(pi_e, 0, 1)
        if pi_e.sum() > 0: pi_e /= pi_e.sum()
        else: pi_e = np.ones(NUM_ACTIONS)/NUM_ACTIONS
            
        pi_a = np.clip(pi_a, 0, 1)
        if pi_a.sum() > 0: pi_a /= pi_a.sum()
        else: pi_a = np.ones(NUM_ACTIONS)/NUM_ACTIONS
            
        return pi_e, pi_a

    except Exception as e:
        # Guaranteed fallback size 4 when equilibria generator is empty
        return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS
    


    
        