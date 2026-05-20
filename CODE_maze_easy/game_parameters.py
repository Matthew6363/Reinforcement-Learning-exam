import numpy as np
### [[ GLOBAL PARAMS ]]
DEBUG = False
TASK = "maze_easy"
TASK_III_EPSILON = 0.5
GRID_H = GRID_W = 0

### [[ GRID and game WORD PARAMS ]]

if TASK == "maze" or TASK == "maze_easy":
    TRAPS_DO_STOP_FOR_A_TURN = False
    
    # [[ Rewards and penalties ]]
    TRAP_NEGATIVE_REWARD = -0.5
    TIMEOUT_PENALTY = 0 # -2.5 
    EGO_LIVING_PENALTY = -0.1
    ADV_LIVING_PENALTY = -0.1
    WALL_HIT_PENALTY = -0.5
    WINNING_MEGA_REWARD = 7
    ADV_CATCH_REWARD = 1
    ADV_TRAP_REWARD  = 0.8
    EGO_CATCHED_NEG_REWARD = -1
    KEY_REWARD = 3


    # [[Suggested Rewards]]
    TIMEOUT_PENALTY        = -5.0   
    EGO_LIVING_PENALTY     = -0.05
    ADV_LIVING_PENALTY     = -0.02
    WALL_HIT_PENALTY       = -1.5 
    WINNING_MEGA_REWARD    =  8.0   
    KEY_REWARD             =  3.0   
    EGO_CATCHED_NEG_REWARD = -3.0 
    TRAP_NEGATIVE_REWARD   = -3.0   
    ADV_CATCH_REWARD       =  5.0   
    ADV_TRAP_REWARD        =  1.0  
    
    # Reduced Grid Size
    GRID_W = 10
    GRID_H = 7 # from 10
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0, 1, 2, 3]

    # Q-table shape for 10x10 (100 * 100 joint states)
    Q_SHAPE = (GRID_W*GRID_H, GRID_W*GRID_H, 5, 5, 4, 4) 

    # New Starting Positions
    START_ego = (0, 0) # (8,6) does work
    START_adv = (2, 4)
    

### [[ Solver Specific parameters QSG-RM ]]
LEMKE_HOWSON = False
GAMMA = 0.9
ALPHA = 0.1
FAIL_RATE = 0.005

# Adjusted for 10,000 states (much faster training)
EPISODES = 200 
SAVE_EACH = 500
TASK_III_EPSILON = 0.5
STEP_NUM = 300  
START_EPSILON = 0.6 
END_EPSILON = 0.05
DECAY_RATE = 0.8
ADD_NOISE = True 
ALLOW_COLLISION_EARLY_BREAK = True 

# [[ PYGAME CONFIG ]]
CELL_SIZE = 50
WIDTH = CELL_SIZE * GRID_W
HEIGHT = CELL_SIZE * GRID_H
MARGIN = 40
WINDOW_W = WIDTH + MARGIN * 2
WINDOW_H = HEIGHT + MARGIN * 2
FPS = 2

BG_COLOR = (250, 250, 250)
GRID_COLOR = (60, 60, 60)
KEY_COLOR = (60,60,255)
EGO_COLOR = (50, 150, 255)   
ADV_COLOR = (255, 50, 50)   
TRAP_COLOR = (255, 100, 100)
EXIT_COLOR = (50, 255, 50)
WALL_COLOR = (60, 60, 60)
AGENT_RATIO = 0.4
BASE_ALPHA = 100    

WALL_COORDS = [
    (0, 2), (0, 9),
    (1, 4), (1, 5), (1, 6), (1,7), (1,9),
    (2, 1), (2, 2), (2, 4), (2, 5), (2, 6), (2, 7),(2,9),
    (3, 1), (3, 2),
    (4,5), (4,6), (4,7), (4,8),
    (5, 1), (5, 2), (5, 3), (5,5),
    (6, 7), (6, 9)
]

# KEY (Yellow)
KEY_COORD = (2, 3)

# TRAPS (Purple)
TRAP_COORDS = [(1, 5), (6, 2)] # Adjusted for 7x10 bounds

# STARTING POSITIONS
START_ego = (6, 0)
START_adv = (0, 3) # Bottom right corner

# EXIT (Green)
EXIT_COORDS = [(0, 0)]