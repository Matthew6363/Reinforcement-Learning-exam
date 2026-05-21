import numpy as np
### [[ GLOBAL PARAMS ]]
DEBUG = False
TASK = "maze_pause"
TASK_III_EPSILON = 0.5
GRID_H = GRID_W = 0

### [[ GRID and game WORD PARAMS ]]

if TASK == "maze_pause":
    TRAPS_DO_STOP_FOR_A_TURN = True
    
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
    TRAP_NEGATIVE_REWARD   = -1.0   
    ADV_CATCH_REWARD       =  5.0   
    ADV_TRAP_REWARD        =  1.0  # maybe it can push ego towards
    
    # Reduced Grid Size
    GRID_W = 8
    GRID_H = 9 # from 10
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
EPISODES = 10000 
SAVE_EACH = 500
TASK_III_EPSILON = 0.5
STEP_NUM = 5000  
START_EPSILON = 0.9 
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

# New 10x10 maze layout
WALL_COORDS = [
    # Top and bottom boundaries with exit gaps
    # (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0,5), (0, 6), (0, 7), 
    (8, 0), (8, 1), (8, 2), (8, 3), 
    (8, 4), (8, 6), (8, 7), 
    # Internal structure
    (0, 4),
    (1, 1), (1, 2), 
    (1, 4), (1, 6),
    (2, 1), (2, 6),
    (3, 1), (3, 3), (3, 4), (3, 5), (3, 6), 
    (5, 1), (5, 2), (5, 4), (5, 5), (5, 6), 
    (6, 4), 
    (7, 1), #(8, 3), (8, 4), (8, 5), (8, 6)
]

# Maybe?
#WALL_COORDS = [
#    # Top and bottom boundaries with exit gaps
#    # (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0,5), (0, 6), (0, 7), 
#    (8, 0), (8, 1), (8, 2), (8, 3), 
#    (8, 4), (8, 6), (8, 7), 
#    # Internal structure
#    (0, 4),
#    (1, 1), (1, 2), 
#    (1, 4), (1, 6),
#    (2, 1), (2, 6),
#    (3, 1), (3, 3), (3, 4), (3, 5), (3, 6), 
#    (5, 1), (5, 2), (5, 4), (5, 5), (5, 6), 
#    (6, 4), 
#    (7, 1), #(8, 3), (8, 4), (8, 5), (8, 6)
#]

KEY_COORD =   (4, 4) # (0,5)
TRAP_COORDS = [(1, 3), (6, 5)]
EXIT_COORDS = [(8, 5)]