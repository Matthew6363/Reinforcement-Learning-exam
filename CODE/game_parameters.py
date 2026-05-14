### GLOBAL PARAMS
DEBUG = False

### GRID WORD PARAMS
GRID_W = 6
GRID_H = 6

BASE_ego_coord = (1,4) # (0,5)
BASE_adv_coord = (0,5) # (1,4) 
START_ego = (0,3) # (5,0)
START_adv = (5,0) # (0,3)

### ACTIONS
STR_ACTIONS = ['up', 'down', 'left', 'right']
ACTIONS = [0,1,2,3]
# ACTIONS_to_ids = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

### QSG-RM
LEMKE_HOWSON = False
GAMMA = 0.9
ALPHA = 0.1
FAIL_RATE = 0.005
EPISODES = 9000
STEP_NUM = 9999 # the max time allowed. Has to be high, since by paper we have to end on v_end only
# Epsilon decays from 0.25 → 0.05 over the first 80% of training,
# then stays fixed. This lets exploitation gradually take over.
START_EPSILON = 0.3
END_EPSILON =   0.05
DECAY_RATE = 0.8
ADD_NOISE = True # in reward machine, 0 if not wanted
ALLOW_COLLISION_EARLY_BREAK = False

### Q-TABLE
Q_SHAPE = (36, 36, 5, 5, 4, 4)








def visualize_grid():
    import numpy as np
    grid = np.full((GRID_H, GRID_W), ".", dtype=object)
    grid[BASE_ego_coord] = "e"
    grid[BASE_adv_coord] = "a" 
    
    grid[START_ego] = "E"  # Agente Ego
    grid[START_adv] = "A"  # Agente Adv
    
    print("--- RENDER                (6x6) ---")
    print("      0  1  2  3  4  5 (Col)")
    for i, row in enumerate(grid):
        print(f"Row {i}: {'  '.join(row)}")
    print("-----------------------------------")
    print("Legend:")
    print(f" E: Start Ego {START_ego} | e: Base Ego {BASE_ego_coord}")
    print(f" A: Start Adv {START_adv} | a: Base Adv {BASE_adv_coord}")
    print()

if __name__ == "__main__":
    
    visualize_grid()