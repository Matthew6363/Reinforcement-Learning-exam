import numpy as np
### [[ GLOBAL PARAMS ]]
DEBUG = False
TASK = "maze"
TASK_III_EPSILON = 0.5
GRID_H = GRID_W = 0

### [[ GRID and game WORD PARAMS ]]
## Task I
if TASK == "task_I" or TASK == 'task_II':
    GRID_W = 6
    GRID_H = 6
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0,1,2,3]

    if TASK == "task_III":
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

    Q_SHAPE = (36, 36, 4, 4, 4, 4) # Q-table shape

    BASE_ego_coord = (1,4) # (0,5)
    BASE_adv_coord = (0,5) # (1,4) 
    START_ego = (0,3) # (5,0)
    START_adv =(5,0)

if TASK == "maze":

    TRAPS_DO_STOP_FOR_A_TURN = False
    TRAP_NEGATIVE_REWARD = - 0.5
    WINNING_MEGA_REWARD = 3
    GRID_W = 20
    GRID_H = 15
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0,1,2,3]

    Q_SHAPE = (GRID_W*GRID_H, GRID_W*GRID_H, 4, 4, 4, 4) # Q-table shape

    START_ego = (8,10) # (5,0)
    START_adv =(4,10)


### [[ Solver Specific parameters QSG-RM ]]
LEMKE_HOWSON = False
GAMMA = 0.9
ALPHA = 0.1
FAIL_RATE = 0.005
EPISODES = 100_000
SAVE_EACH = 5000
TASK_III_EPSILON = 0.5
STEP_NUM = 10000 # the max time allowed. Has to be high, since by paper we have to end on v_end only
START_EPSILON = 0.8 # Epsilon decays from 0.25 → 0.05 over the first 80% of training,
                    # then stays fixed. This lets exploitation gradually take over.
END_EPSILON =   0.05
DECAY_RATE = 0.8
ADD_NOISE = True # in reward machine, 0 if not wanted
ALLOW_COLLISION_EARLY_BREAK = False


# [[ PYGAME CONFIG ]]
CELL_SIZE = 50
WIDTH = CELL_SIZE * GRID_W
HEIGHT = CELL_SIZE * GRID_H
MARGIN = 40
WINDOW_W = WIDTH + MARGIN * 2
WINDOW_H = HEIGHT + MARGIN * 2
FPS = 2 # Slowed down slightly so you can watch them clearly

BG_COLOR = (250, 250, 250)
GRID_COLOR = (60, 60, 60)
EGO_COLOR = (50, 150, 255)   
ADV_COLOR = (255, 50, 50)   
TRAP_COLOR = (255, 100, 100)
WALL_COLOR = (60, 60, 60)
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




WALL_COORDS = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19),
    (1, 0), (1, 4), (1, 7),(1,19),
    (2, 0), (2, 2), (2, 9), (2, 13), (2, 14), (2, 15), (2, 17), (2, 19),
    (3, 0), (3, 2), (3, 3), (3, 5), (3, 8), (3, 12), (3, 15), (3, 17), (3, 19),
    (4, 0), (4, 2), (4, 5), (4, 6), (4, 7), (4, 8), (4, 12), (4, 13), (4, 15), (4, 19),
    (5, 0), (5, 1), (5, 2), (5, 4), (5, 5), (5, 8), (5, 12), (5,16), (5,18), (5,19),
    (6, 2), (6, 5), (6, 8), (6, 12), (6, 14),
    (7, 8), (7, 12), (7, 15), (7, 16), (7, 17),
    (8, 0), (8, 2), (8, 3), (8, 4), (8, 5), (8, 7), (8, 8), (8, 12), (8, 13), (8, 16), (8, 19),
    (9, 0), (9, 2), (9, 5), (9, 8), (9, 9), (9, 15), (9, 16), (9, 17), (9, 19),
    (10, 0), (10, 2), (10, 3), (10, 5), (10, 6), (10, 16), (10, 19),
    (11, 0), (11, 8), (11, 10), (11, 12), (11, 19),
    (12, 0), (12, 2), (12, 3), (12, 5), (12, 6), (12, 10), (12, 12), (12, 13), (12, 16), (12, 19),
    (13, 0), (13, 10), (13, 12), (13, 15), (13, 16), (13, 17), (13, 19),
    (14, 0), (14, 1), (14, 2), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13), (14, 14), (14, 15), (14, 16), (14, 17), (14, 18), (14, 19)
]

TRAP_COORDS = [(4, 1),(11, 4), (11, 16), (4, 16)]
EXIT_COORDS = [(6, 0), (7, 0), (6, 19), (7, 19)]