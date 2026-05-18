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

    
    def simulate_step(self, hypothetical_state, labels):
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
        next_state, reward = self.step(labels)
        
        # 4. Revert the machine back to reality
        self.state = current_real_state
        
        return next_state, reward
        
    
    def step(self, labels, use_mini_rewards = True):
        '''
        Function to perform a movement in the RM. We do initialize the reward at zero (it's the baseline
        for any non-relevant game steps) and we do define the RM as game requires. 
        Please refer to paper Figure 4. (a) to RM visualization.

        Notice that it handles agent cases for rewards, so it's generic enough to handle 
        both the "ego" or "adv" point of view. Notice that by paper design, no negative reward is given.
        '''
        
        reward = 0.0 # init the reward to its baseline 
        
        ## State: Start / Balance
        #  ---------------------- 
        # If the agent is in the starting RM state, then two are the possible edges available
        # * It does a game action leading to the game state in which goes to ITS power base;
        #   if so, we have a starting state at start and we need an "l" (high level game step translation
        #   for RM machine, ref. paper) ma 

        if self.state == 'start': # If the agent is in the RM starting state...

            # If the agent is in power_e, then we're in the state v1 
            if 'power_e' in labels:
                self.state = 'v_1'
            
            # otherwise, if in power_a and not in power_e, we're in v2
            elif "collision" in labels:
                self.state = 'v_end'
                if self.agent_type == 'adv': reward = 1.0

            ## Case: all dots collected
            elif 'dot_taken' in labels:
                self.state = 'v_2'
                if use_mini_rewards:
                    if self.agent_type == 'ego': reward = INTERM_REWARD # Breadcrumb for Ego
                
        ## State: v_2
        elif self.state == 'v_2':

            if 'dot_taken' in labels:
                if use_mini_rewards:
                    if self.agent_type == 'ego': reward = INTERM_REWARD # Breadcrumb for Ego
            
            if 'power_e' in labels:
                self.state = 'v_1'
            
            if 'all_visited' in labels:
                self.state = 'v_win'
                if self.agent_type == 'ego': reward = 1.0 # Breadcrumb for Ego
            
            if 'collision' in labels:
                self.state = 'v_end'
                if self.agent_type == 'adv': reward = 1.0 # Breadcrumb for Ego

        ## State: v_1 (currently, Ego > Adv)
        elif self.state == 'v_1':
            
            if 'all_visited' in labels:
                self.state = 'v_win'
                if self.agent_type == 'ego': reward = 1.0 # Breadcrumb for Ego

            if 'dot_taken' in labels:
                if use_mini_rewards:
                    if self.agent_type == 'ego': reward = INTERM_REWARD # Breadcrumb for Ego
            
            if 'collision' in labels:
                self.state = 'v_2'
                if use_mini_rewards:
                    if self.agent_type == 'ego': reward = INTERM_REWARD
            
            
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
    #  Bypass Nash solver if the Q-matrix is completely empty (avoids degeneracy crash). 
    #  These strategies are initialized to be equi-prob state-actions 
    if np.all(q_matrix_ego == 0) and np.all(q_matrix_adv == 0):
        # return equi-distributed random policies. Any action has same probalility given a state.
        return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS
    
    ## INIT a Nash Game with the 2 q-tables
    noise_e = 0.0
    noise_a = 0.0
    if ADD_NOISE:
        noise_e = 1e-9 # np.random.uniform(1e-6, 1e-5, size=q_matrix_ego.shape)
        noise_a = 1e-9 # np.random.uniform(1e-6, 1e-5, size=q_matrix_adv.shape)
        
    game = nash.Game(q_matrix_ego + noise_e, q_matrix_adv + noise_a)
    
    try:
        # Use support_enumeration (more mathematically stable for grid worlds)
        if LEMKE_HOWSON:
            random_label = np.random.randint(0, NUM_ACTIONS-1)
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
            pi_e, pi_a = next(equilibria) # find and take the next one (first found) and assing it to the strategies
        
        # Note: pi_e, pi_a are two STRATEGIES, as vectors containing the probabilities 
        # of doing the actions! Notice that actions are ordered wrt to the Q-table passed to this 
        # function,
        
        # Normalize and clean probabilities
        pi_e = np.clip(pi_e, 0, 1)
        if pi_e.sum() > 0: pi_e /= pi_e.sum()
        else: pi_e = np.ones(NUM_ACTIONS)/NUM_ACTIONS
            
        pi_a = np.clip(pi_a, 0, 1)
        if pi_a.sum() > 0: pi_a /= pi_a.sum()
        else: pi_a = np.ones(NUM_ACTIONS)/NUM_ACTIONS
            
        return pi_e, pi_a

    except Exception as e:
        # 4. Guaranteed fallback size 4
        print("Exception occurred, saved in log.", e)
        log_error_to_file(str(e), q_matrix_ego, q_matrix_adv)

        # and return the random one.
        return np.ones(NUM_ACTIONS)/NUM_ACTIONS, np.ones(NUM_ACTIONS)/NUM_ACTIONS
    



def testing_ego_RM():
    print("==============================================")
    print(" TESTING EGO AGENT REWARD MACHINE")
    print("==============================================")
    
    rm_ego = Reward_Machine('ego')
    
    print("\n--- Test 1: Successful Ego Task Sequence ---")
    # Expected sequence: power_e -> power_a -> collision
    sequence_1 = [{'power_e'}, {'ego_at_base_a'}, {'collision'}]
    
    for i, labels in enumerate(sequence_1):
        state, reward = rm_ego.step(labels)
        print(f"Step {i+1} | Input: {labels} | RM State: {state} | Reward: {reward}")
        
    print("\n--- Test 2: Out of Order Ego Task (Fails) ---")
    rm_ego.reset()
    # Sequence: collision -> power_a -> power_e (wrong order)
    sequence_2 = [{'collision'}, {'power_a'}, {'power_e'}]
    
    for i, labels in enumerate(sequence_2):
        state, reward = rm_ego.step(labels)
        print(f"Step {i+1} | Input: {labels} | RM State: {state} | Reward: {reward}")

    print("\n==============================================")
    print(" TESTING ADVERSARIAL AGENT REWARD MACHINE")
    print("==============================================")
    
    rm_adv = Reward_Machine('adv')
    
    print("\n--- Test 3: Successful Adv Task Sequence ---")
    # Expected sequence: power_a -> collision
    sequence_3 = [{'power_a'}, {'collision'}]
    
    for i, labels in enumerate(sequence_3):
        state, reward = rm_adv.step(labels)
        print(f"Step {i+1} | Input: {labels} | RM State: {state} | Reward: {reward}")
        
    print("\n--- Test 4: Premature Collision (Fails) ---")
    rm_adv.reset()
    # Adv tries to collide before getting power
    sequence_4 = [{'collision'}, {'power_a'}]
    
    for i, labels in enumerate(sequence_4):
        state, reward = rm_adv.step(labels)
        print(f"Step {i+1} | Input: {labels} | RM State: {state} | Reward: {reward}")



if __name__ == "__main__":
    
    if DEBUG == True:
        testing_ego_RM()
    
        