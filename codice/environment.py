import numpy as np
import nashpy as nash

class PacmanGridWorld:
    def __init__(self):
        self.grid_size = (6,6)
        self.slip_rate = 0.005
        #self.actions = ['up', 'down', 'left', 'right']
        self.actions = [0,1,2,3]
        
        self.base_a = (0,5)
        self.base_e = (1,4)
        
        self.str_to_idx = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        
    def reset(self):
        
        self.pos_a = (5,0)
        self.pos_e = (0,3)
        return self.pos_e, self.pos_a
    
    def step(self, action_e, action_a):
        
        if isinstance(action_a, list): action_a = action_a[0]
        if isinstance(action_e, list): action_e = action_e[0]
        
        if isinstance(action_a, str): action_a = self.str_to_idx[action_a]
        if isinstance(action_e, str): action_e = self.str_to_idx[action_e]
        
        #apply slip
        if np.random.rand() > self.slip_rate:
            self.pos_a = self.move(self.pos_a, action_a)
        if np.random.rand() > self.slip_rate:
            self.pos_e = self.move(self.pos_e, action_e)
        return self.pos_e, self.pos_a
    
    def move(self, pos, action):
        
        r, c = pos

        if action not in self.actions:
            raise ValueError("invalid actions")
            
        if action == 0: #up
            if (r > 0 and r <= self.grid_size[0] - 1 ): 
                r = r-1
            
        if action == 1: #down
            if ( r >= 0 and r < self.grid_size[0] - 1):
                    r = r+1            
            
        if action == 2: #left
            if ( c > 0 and c <= self.grid_size[1] - 1):
                    c = c-1
            
        if action == 3: #right : 
            if ( c >= 0 and c < self.grid_size[1] - 1):
                c = c+1
        
        return (r,c)
    

    def get_labels(self):
        labels = set()
        if self.pos_a == self.base_a:
            labels.add('power_a')
        if self.pos_e == self.base_e:
            labels.add('power_e')
            
        if self.pos_e == self.base_a:
            labels.add('ego_at_base_a')
        
        if (abs(self.pos_a[0] - self.pos_e[0]) + abs(self.pos_a[1] - self.pos_e[1])) < 2:
            labels.add('collision')
        return labels
            
            
            
if __name__ == "__main__":
    env = PacmanGridWorld()
    pos_a, pos_e = env.reset()
    
    print(f"Initial State -> A: {pos_a}, E: {pos_e}")
    print(f"Initial Labels -> {env.get_labels()}\n")
    
    print("--- Testing Movement (A moves Left, E moves Right) ---")
    pos_a, pos_e = env.step(['left'], ['right'])
    print(f"State -> A: {pos_a}, E: {pos_e}")
    print(f"Labels -> {env.get_labels()}\n")
    
    print("--- Testing Power Base Detection ---")
    env.pos_a = env.base_a 
    env.pos_e = env.base_e
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    print(f"Labels -> {env.get_labels()}\n")
    
    print("--- Testing Collision Detection ---")
    env.pos_a = (2, 2)
    env.pos_e = (2, 3)
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    print(f"Labels -> {env.get_labels()}\n")

    # --- NEW TEST HERE ---
    print("--- Testing Multiple Labels Simultaneously ---")
    # We place both Ego and Adv directly on top of the Adv's base
    env.pos_a = env.base_a  # (0,5)
    env.pos_e = env.base_a  # (0,5)
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    # This will trigger 'power_a' (Adv at base), 'ego_at_base_a' (Ego at Adv base), AND 'collision' (distance < 2)
    print(f"Labels -> {env.get_labels()}\n")
    
    print("--- Testing Random Walk for 5 Steps ---")
    env.reset()
    action_strings = list(env.str_to_idx.keys())
    
    for i in range(5):
        act_a_str = np.random.choice(action_strings)
        act_e_str = np.random.choice(action_strings)
        
        pos_a, pos_e = env.step(act_a_str, act_e_str) 
        
        print(f"Step {i+1} | Actions: A='{act_a_str}', E='{act_e_str}' | State: A={pos_a}, E={pos_e}")