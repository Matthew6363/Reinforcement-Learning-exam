## LIB FILE
## File dedicated to the functional definitions of the problem environment


# == MAIN PARAMETERS and libs == #
import numpy as np

DEBUG = True
GRID_W = 6
GRID_H = 6
FAIL_RATE = 0.005

STR_ACTIONS = ['up', 'down', 'left', 'right']
ACTIONS = [0,1,2,3]
# ACTIONS_to_ids = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

BASE_ego_coord = (0,5)
BASE_adv_coord = (1,4)
START_ego = (5,0)
START_adv = (0,3)



# =================== #
# Environment (Class) #
# =================== #

class PacmanGridWorld:
    def __init__(self, 
                 grid_size = (GRID_W, GRID_H),
                 actions = ACTIONS,
                 str_actions = STR_ACTIONS,
                 fail_rate = FAIL_RATE,
                 base_ego_coords = BASE_ego_coord,
                 base_adv_coords = BASE_adv_coord,
                 starting_pos_ego = START_ego,
                 starting_pos_adv = START_adv,
                 ):
        '''
        Initialize function for the Game object/instance. Requires some (globally defined)
        game informations.
        '''

        self.grid_size = grid_size
        self.fail_rate = fail_rate
        self.actions = actions
        self.base_a = base_adv_coords
        self.base_e = base_ego_coords

        self.str_to_idx = {e:i for i,e in enumerate(str_actions)}
        
        # initialize positions
        self.start_pos_a = starting_pos_adv
        self.start_pos_e = starting_pos_ego
        
        self.pos_a = starting_pos_adv
        self.pos_e = starting_pos_ego
        

    def reset(self):
        '''
        Reset the Environment, by making the current state the initial state again
        '''
        self.pos_a = self.start_pos_a
        self.pos_e = self.start_pos_e

        return self.pos_e, self.pos_a


    def move(self, pos, action):
        '''
        This function encodes the actual dynamics of the environment, allowing us to recover
        where the agent would get after having peformed the action at a given position. This is the core
        for the actual "step" method defined below.
        '''

        r, c = pos # the position is unfolded into rows and columns

        ## Is the action available?
        if action not in self.actions:
            raise ValueError("invalid actions")
            
        ## Case: UP
        if action == 0: #up
            if (r > 0 and r <= self.grid_size[0] - 1 ): 
                r = r-1
                # Go up if the row is not the first one (border)

        ## Case: DOWN    
        elif action == 1:
            if ( r >= 0 and r < self.grid_size[0] - 1):
                r = r+1            
                # Go down if the row is not the last one (border)
        
        ## Case: LEFT
        elif action == 2:
            if ( c > 0 and c <= self.grid_size[1] - 1):
                c = c-1
                # Go left if the row is not the first one (border)
        
        ## Case: RIGHT
        elif action == 3: 
            if ( c >= 0 and c < self.grid_size[1] - 1):
                c = c+1
                # Go right if the row is not the last one (border)

        # Then return the position of the grid which is found        
        return (r,c)



    def step(self, action_e, action_a, debug = False):
        '''
        This env function allows for action execution. Since actions are performed at the same
        time for both ego agent and adv agent, we pass those two at the same time. Notice that
        actions as to be coherent with the available ones. This is already checked in the move function.

        Notice that the need of str_to_idx usage is related to probability vector indexes. 

        Notice that the Fail rate is here used. This allows for failure, but implicilty allow a
        probability of staying in the same cell.

        We perform the action and we return the state/position of the game reached.
        '''

        # if more than one are given, take the first one
        if isinstance(action_a, list): action_a = action_a[0] 
        if isinstance(action_e, list): action_e = action_e[0]
        
        # if one is given but as a string, (e.g. "up"), we need the correspondent index.abs
        # So, the action is transated to the idx. 
        if isinstance(action_a, str): action_a = self.str_to_idx[action_a]
        if isinstance(action_e, str): action_e = self.str_to_idx[action_e]
        
        if debug:
            print(action_a, action_e)

        # Get two numbers
        ego_move_chance = np.random.rand()
        adv_move_chance = np.random.rand()
        
        if adv_move_chance > self.fail_rate:
            # then action is executed, the agent moves
            self.pos_a = self.move(self.pos_a, action_a)
        
        if ego_move_chance > self.fail_rate:
            # then action is executed, the agent moves
            self.pos_e = self.move(self.pos_e, action_e)
        
        # return the reached position.
        return self.pos_e, self.pos_a
    


    def get_labels(self):
        '''
        The paper defined a function L which was able to map the agent-environment interraction
        into a set of labels, allowing for the communication with the Reward Machine (see. reward_machine.py).
        We define then this function, cheching the position of agents wrt to the known positions of 
        power bases. 

        '''
        labels = set() # this is not ordered, but it's not needed since we'll look at presence in the RM

        # Case: Adv CURRENTLY IN the power base
        if self.pos_a == self.base_a:
            labels.add('power_a')
        
        # Case: Ego CURRENTLY IN the power base
        if self.pos_e == self.base_e:
            labels.add('power_e')
        
        # Case: Ego CURRENTLY IN the Adv power base
        if self.pos_e == self.base_a:
            labels.add('ego_at_base_a')
        
        # Case: Adv CURRENTLY IN Ego position (symmetric), i.e. at less than two in distance
        if (abs(self.pos_a[0] - self.pos_e[0]) + abs(self.pos_a[1] - self.pos_e[1])) < 2:
            labels.add('collision')
        return labels


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