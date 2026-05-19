import numpy as np
### [[ GLOBAL PARAMS ]]
DEBUG = False
TASK = "maze"
TASK_III_EPSILON = 0.5
GRID_H = GRID_W = 0

### [[ GRID and game WORD PARAMS ]]

if TASK == "maze":
    TRAPS_DO_STOP_FOR_A_TURN = False
    TRAP_NEGATIVE_REWARD = -0.5
    WINNING_MEGA_REWARD = 5
    WALL_HIT_PENALTY = -0.5
    KEY_REWARD = 3
    
    # Reduced Grid Size
    GRID_W = 8
    GRID_H = 10
    STR_ACTIONS = ['up', 'down', 'left', 'right']
    ACTIONS = [0, 1, 2, 3]

    # Q-table shape for 10x10 (100 * 100 joint states)
    Q_SHAPE = (GRID_W*GRID_H, GRID_W*GRID_H, 5, 5, 4, 4) 

    # New Starting Positions
    START_ego = (1, 0) # (8,6) does work
    START_adv = (3, 4)
    

### [[ Solver Specific parameters QSG-RM ]]
LEMKE_HOWSON = False
GAMMA = 0.9
ALPHA = 0.1
FAIL_RATE = 0.005

# Adjusted for 10,000 states (much faster training)
EPISODES = 20000 
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
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 6), (0, 7), 
    (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 6), (9, 7), 
    # Internal structure
    (1, 4),
    (2, 1), (2, 2), (2, 4), (2, 6),
    (3, 1), (3, 6),
    (4, 1), (4, 3), (4, 4), (4, 5), (4, 6), 
    (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), 
    (7, 4), 
    (8, 1), #(8, 3), (8, 4), (8, 5), (8, 6)
]

KEY_COORD = (5, 4)
TRAP_COORDS = [(2, 3), (7, 5)]
EXIT_COORDS = [(9, 5)]