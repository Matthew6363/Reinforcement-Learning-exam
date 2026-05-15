import numpy as np
### [[ GLOBAL PARAMS ]]
DEBUG = False
TASK = "task_I"
TASK_III_EPSILON = 0.5

### [[ GRID and game WORD PARAMS ]]
## Task I
if TASK == "task_I" or TASK == 'task_II':
    GRID_W = 6
    GRID_H = 6
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0,1,2,3]

    if TASK == "task_I":
        Q_SHAPE = (36, 36, 5, 5, 4, 4) # Q-table shape
    
    else: Q_SHAPE = (36, 36, 6, 6, 4, 4) # Q_table shape for task II and III

    BASE_ego_coord = (1,4) # (0,5)
    BASE_adv_coord = (0,5) # (1,4) 
    START_ego = (0,3) # (5,0)
    START_adv = (5,0) # (0,3)

## Add other cases for additional tasks or user defined problems
if TASK == "task_III":
    GRID_W = 6
    GRID_H = 6
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0,1,2,3]

    Q_SHAPE = (36, 36, 6, 6, 4, 4) # Q-table shape

    BASE_ego_coord = (1,4) # (0,5)
    BASE_adv_coord = (0,5) # (1,4) 
    START_ego = (0,3) # (5,0)
    if np.random.rand() > TASK_III_EPSILON : 
        START_adv =(5,0)
    else: 
        START_adv = (5,1)


### [[ Solver Specific parameters QSG-RM ]]
LEMKE_HOWSON = False
GAMMA = 0.9
ALPHA = 0.1
FAIL_RATE = 0.005
EPISODES = 9000
TASK_III_EPSILON = 0.5
STEP_NUM = 9999 # the max time allowed. Has to be high, since by paper we have to end on v_end only
START_EPSILON = 0.25 # Epsilon decays from 0.25 → 0.05 over the first 80% of training,
                    # then stays fixed. This lets exploitation gradually take over.
END_EPSILON =   0.05
DECAY_RATE = 0.8
ADD_NOISE = True # in reward machine, 0 if not wanted
ALLOW_COLLISION_EARLY_BREAK = False


# [[ PYGAME CONFIG ]]
CELL_SIZE = 100
WIDTH = CELL_SIZE * GRID_W
HEIGHT = CELL_SIZE * GRID_H
MARGIN = 40
WINDOW_W = WIDTH + MARGIN * 2
WINDOW_H = HEIGHT + MARGIN * 2
FPS = 1 # Slowed down slightly so you can watch them clearly

BG_COLOR = (250, 250, 250)
GRID_COLOR = (60, 60, 60)
EGO_COLOR = (50, 150, 255)   
ADV_COLOR = (255, 50, 50)    
AGENT_RATIO = 0.4
BASE_ALPHA = 100    







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