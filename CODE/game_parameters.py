### GLOBAL PARAMS
DEBUG = False

### GRID WORD PARAMS
GRID_W = 6
GRID_H = 6
FAIL_RATE = 0.005
BASE_ego_coord = (0,5)
BASE_adv_coord = (1,4)
START_ego = (5,0)
START_adv = (0,3)

### ACTIONS
STR_ACTIONS = ['up', 'down', 'left', 'right']
ACTIONS = [0,1,2,3]
# ACTIONS_to_ids = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

### ACTIONS
STR_ACTIONS = ['up', 'down', 'left', 'right']
ACTIONS = [0,1,2,3]
# ACTIONS_to_ids = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

### QSG-RM
GAMMA = 0.9
ALPHA = 0.1

EPISODES = 12000
STEP_NUM = 9999 # the max time allowed. Has to be high, since by paper we have to end on v_end only

# Epsilon decays from 0.25 → 0.05 over the first 80% of training,
# then stays fixed. This lets exploitation gradually take over.
START_EPSILON = 0.3
END_EPSILON =   0.05
DECAY_RATE = 0.8

### Q-TABLE
Q_SHAPE = (36, 36, 5, 5, 4, 4)

