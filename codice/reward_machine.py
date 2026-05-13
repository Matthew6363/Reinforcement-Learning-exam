import nashpy as nash
import numpy as np

class Reward_Machine():
    def __init__(self, agent_type):
        self.agent_type = agent_type
        self.reset()
        
    def reset(self):
        self.state = 'start'
        return self.state
    
    def simulate_step(self, hypothetical_state, labels):
        """
        Allows the algorithm to test counterfactual states 
        without breaking the real game state.
        """
        # 1. Save the actual physical state
        current_real_state = self.state
        
        # 2. Temporarily overwrite it with the hypothetical state
        self.state = hypothetical_state
        
        # 3. See what the reward and next state WOULD have been
        next_state, reward = self.step(labels)
        
        # 4. Revert the machine back to reality
        self.state = current_real_state
        
        return next_state, reward
        
    def step(self, labels):
        reward = 0.0
        
        # State: Start / Balance
        if self.state == 'start':
            if 'power_e' in labels and 'power_a' not in labels:
                self.state = 'v_1'
                #if self.agent_type == 'ego': reward = 0.1 # Breadcrumb for Ego
            elif 'power_a' in labels and 'power_e' not in labels:
                self.state = 'v_2'
                #if self.agent_type == 'adv': reward = 0.1 # Breadcrumb for Adv
                
        # State: Ego is Powerful
        elif self.state == 'v_1':
            if 'power_a' in labels and 'power_e' not in labels:
                self.state = 'v_2' # Adv steals power
            elif 'ego_at_base_a' in labels and 'power_a' not in labels:
                self.state = 'v_3' # Ego destroys Adv base
                #if self.agent_type == 'ego': reward = 0.5 # Big breadcrumb!
            elif 'power_a' in labels and 'power_e' in labels:
                self.state = 'start'
                
        # State: Adv is Powerful
        elif self.state == 'v_2':
            if 'power_e' in labels and 'power_a' not in labels:
                self.state = 'v_1' # Ego steals power
            elif 'power_a' in labels and 'power_e' in labels:
                self.state = 'start'
            elif 'collision' in labels and 'power_e' not in labels:
                self.state = 'v_end'
                if self.agent_type == 'adv': reward = 1.0 # Adv wins
                
        # State: Adv base destroyed, Ego ready to capture
        elif self.state == 'v_3':
            if 'collision' in labels:
                self.state = 'v_end'
                if self.agent_type == 'ego': reward = 1.0 # Ego wins

        return self.state, reward             
            
        
def solve_stage_game(q_matrix_ego, q_matrix_adv):
    
    # 1. Bypass Nash solver if the Q-matrix is completely empty (avoids degeneracy crash)
    if np.all(q_matrix_ego == 0) and np.all(q_matrix_adv == 0):
        return np.ones(4)/4, np.ones(4)/4
        
    game = nash.Game(q_matrix_ego, q_matrix_adv)
    
    try:
        # 2. Use support_enumeration (more mathematically stable for grid worlds)
        equilibria = game.support_enumeration()
        pi_e, pi_a = next(equilibria)
        #print(pi_e, pi_a, 'policies')
        
        # 3. Clean and normalize the probabilities
        pi_e = np.clip(pi_e, 0, 1)
        if pi_e.sum() > 0: pi_e /= pi_e.sum()
        else: pi_e = np.ones(4)/4
            
        pi_a = np.clip(pi_a, 0, 1)
        if pi_a.sum() > 0: pi_a /= pi_a.sum()
        else: pi_a = np.ones(4)/4
            
        return pi_e, pi_a
    
    except:
        # 4. Guaranteed fallback size 4
        return np.ones(4)/4, np.ones(4)/4
    
if __name__ == "__main__":
    
    print("==============================================")
    print(" TESTING EGO AGENT REWARD MACHINE")
    print("==============================================")
    
    rm_ego = Reward_Machine('ego')
    
    print("\n--- Test 1: Successful Ego Task Sequence ---")
    # Expected sequence: power_e -> power_a -> collision
    sequence_1 = [{'power_e'}, {'power_a'}, {'collision'}]
    
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
        