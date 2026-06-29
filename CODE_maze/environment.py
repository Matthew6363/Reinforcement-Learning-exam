## LIB FILE
## File dedicated to the functional definitions of the problem environment


# == MAIN PARAMETERS and libs == #
import numpy as np
from game_parameters import *
DEBUG = False


class PacmanGridWorld:
    def __init__(self, 
                 grid_size = (GRID_H, GRID_W),
                 actions = ACTIONS,
                 str_actions = STR_ACTIONS,
                 fail_rate = FAIL_RATE,
                 wall_coords = WALL_COORDS,
                 trap_coords = TRAP_COORDS,
                 exit_coords = EXIT_COORDS,
                 key_coord = KEY_COORD,
                 starting_pos_ego = START_ego,
                 starting_pos_adv = START_adv,
                 ):
        '''
#         Initialize function for the Game object/instance. Requires some (globally defined)
#         game informations.
#         '''

        self.grid_size = grid_size
        self.fail_rate = fail_rate
        self.actions = actions
        self.str_to_idx = {e:i for i,e in enumerate(str_actions)}
        
        # initialize positions
        self.start_pos_a = starting_pos_adv
        self.start_pos_e = starting_pos_ego

        if TASK == "task_III":
            # with 50% prob change the base to the other (5,1) from (5,0) of
            # all the other cases
            if np.random.rand() > TASK_III_EPSILON: 
                r, c = self.start_pos_a
                self.start_pos_a = (r, c + 1) # now (5, 1)
        
        if TASK == "maze":
            self.traps = trap_coords
            self.walls = wall_coords
            self.exits = exit_coords
            self.keys = key_coord
            self.backup_key = key_coord
            self.trapped = False

        self.pos_a = self.start_pos_a
        self.pos_e = starting_pos_ego
        self.ego_trapped = False  
        
    def reset(self):
        '''
        Reset the Environment, by making the current state the initial state again
        '''
        self.pos_a = self.start_pos_a
        self.pos_e = self.start_pos_e
        
        # Fix: Clear the stun status on episode reset
        self.ego_trapped = False
        self.ego_on_wall = False
        self.adv_on_wall = False
        self.doors_opened = False
        self.keys = self.backup_key

        return self.pos_e, self.pos_a
        

    # --------------------------------------------------------------------
    def move(self, pos, action, is_ego=False):
        r, c = pos
        hold_pos = pos

        # Normal movement 
        if action not in self.actions:
            raise ValueError("invalid actions")

        if action == 0:  # up
            if r > 0: 
                r -= 1
            else:
                if is_ego:
                    self.ego_on_wall = True 
                else:
                    self.adv_on_wall = True
        elif action == 1:  # down
            if r < self.grid_size[0] - 1: 
                r += 1
            else:
                if is_ego:
                    self.ego_on_wall = True 
                else:
                    self.adv_on_wall = True
        elif action == 2:  # left
            if c > 0: 
                c -= 1
            else:
                if is_ego:
                    self.ego_on_wall = True 
                else:
                    self.adv_on_wall = True
        elif action == 3:  # right
            if c < self.grid_size[1] - 1: 
                c += 1
            else:
                if is_ego:
                    self.ego_on_wall = True 
                else:
                    self.adv_on_wall = True

        # Wall collision 
        if (r, c) in self.walls:
            if is_ego:
                self.ego_on_wall = True 
            else:
                self.adv_on_wall = True
            return hold_pos


        # Trap handling: trigger the stun for NEXT turn
        if is_ego and (r, c) in self.traps:
            self.ego_trapped = True   
            
        return (r, c)

    # --------------------------------------------------------------------
    def step(self, action_e, action_a, debug=False):
        if isinstance(action_a, list): action_a = action_a[0] 
        if isinstance(action_e, list): action_e = action_e[0]
        
        if isinstance(action_a, str): action_a = self.str_to_idx[action_a]
        if isinstance(action_e, str): action_e = self.str_to_idx[action_e]
        
        adv_move_chance = np.random.rand()
        
        # ---- EGO STUN LOGIC ----
        if self.ego_trapped:
            self.ego_trapped = False # Clear the stun for next turn
            ego_move_chance = 0.0    # Force the movement to fail this turn
        else:
            ego_move_chance = np.random.rand()

        # ---- MOVEMENT ----
        if adv_move_chance > self.fail_rate:
            self.pos_a = self.move(self.pos_a, action_a, is_ego=False)
            
        if ego_move_chance > self.fail_rate:
            self.pos_e = self.move(self.pos_e, action_e, is_ego=True)

        return self.pos_e, self.pos_a

    # --------------------------------------------------------------------
    def get_labels(self):
        """
        Only reports what is currently true. No side‑effects (no stun changes).
        """
        labels = set()

        if self.pos_e in self.traps:
            labels.add('trapped')      # ego is currently on a trap cell

        if self.pos_e in self.exits:
            if self.doors_opened:
                labels.add('escaped!')
        
        if self.ego_on_wall:
            self.ego_on_wall = False
            labels.add('ego_on_wall')

        if self.adv_on_wall:
            self.adv_on_wall = False
            labels.add('adv_on_wall')

        # Collision: Manhattan distance < 1 (i.e. same cell)
        if (abs(self.pos_a[0] - self.pos_e[0]) + 
            abs(self.pos_a[1] - self.pos_e[1])) < 2:
            labels.add('collision')

        if self.pos_e == self.keys:
            self.doors_opened = True
            self.keys = None
            labels.add('key')

        return labels

    def clear_turn_flags(self):
        """
        To be called at the end of the sim step
        """
        if self.ego_on_wall:
            self.ego_on_wall = False
        if self.adv_on_wall:
            self.adv_on_wall = False
        if self.pos_e == self.keys:
            self.doors_opened = True
            self.keys = None

def build_test_env():
    env = PacmanGridWorld()
    pos_e, pos_a = env.reset()
    
    print(f"\nInitial State -> A: {pos_a}, E: {pos_e}")
    print(f"Initial Labels -> {env.get_labels()}\n")
    
    print("--- Testing Movement (A moves Left, E moves Right) ---")
    pos_e, pos_a = env.step(['right'], ['left'], debug=True)
    print(f"State -> A: {pos_a}, E: {pos_e}")
    print(f"Labels -> {env.get_labels()}\n\n")
    
    print("--- Testing Power Base Detection ---")
    env.pos_a = env.base_a 
    env.pos_e = env.base_e
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    print(f"Labels -> {env.get_labels()}\n\n")
    
    print("--- Testing Collision Detection ---")
    env.pos_a = (2, 2)
    env.pos_e = (2, 3)
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    print(f"Labels -> {env.get_labels()}\n\n")

    # --- NEW TEST HERE ---
    print("--- Testing Multiple Labels Simultaneously ---")
    # We place both Ego and Adv directly on top of the Adv's base
    env.pos_a = env.base_a  # (0,5)
    env.pos_e = env.base_a  # (0,5)
    print(f"State -> A: {env.pos_a}, E: {env.pos_e}")
    # This will trigger 'power_a' (Adv at base), 'ego_at_base_a' (Ego at Adv base), AND 'collision' (distance < 2)
    print(f"Labels -> {env.get_labels()}\n\n")
    
    print("--- Testing Random Walk for 5 Steps ---")
    env.reset()
    action_strings = list(env.str_to_idx.keys())
    
    for i in range(5):
        act_a_str = np.random.choice(action_strings)
        act_e_str = np.random.choice(action_strings)
        
        pos_e, pos_a = env.step(act_e_str, act_a_str) 
        
        print(f"Step {i+1} | Actions: A='{act_a_str}', E='{act_e_str}' | State: A={pos_a}, E={pos_e}")

            
if __name__ == "__main__":
    if DEBUG:
        build_test_env()
