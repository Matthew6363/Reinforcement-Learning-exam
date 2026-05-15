import pygame
import sys
import numpy as np
from environment import PacmanGridWorld
from reward_machine_task_I import *
from game_parameters import *
# --- PYGAME CONFIG ---
CELL_SIZE = 100
GRID_SIZE = 6
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE
FPS = 20 # Slowed down slightly so you can watch them clearly

BG_COLOR = (30, 30, 30)
GRID_COLOR = (60, 60, 60)
EGO_COLOR = (50, 150, 255)   
ADV_COLOR = (255, 50, 50)    
BASE_ALPHA = 100             

def draw_grid(screen):
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

def draw_base(screen, pos, color):
    r, c = pos
    surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(surface, (*color, BASE_ALPHA), (0, 0, CELL_SIZE, CELL_SIZE))
    screen.blit(surface, (c * CELL_SIZE, r * CELL_SIZE))
    font = pygame.font.SysFont(None, 48)
    img = font.render('P', True, color)
    screen.blit(img, (c * CELL_SIZE + 35, r * CELL_SIZE + 35))

def draw_ghost(screen, pos, color):
    r, c = pos
    cx = c * CELL_SIZE + CELL_SIZE // 2
    cy = r * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 3)
    pygame.draw.rect(screen, color, (cx - CELL_SIZE//3, cy, (CELL_SIZE//3)*2, CELL_SIZE//3))

def pos_to_idx(pos): 
    return pos[0]*6 + pos[1]

def visualize_trained_agents(model_path='../EXPORT/q_models.npz', num_episodes=1):
    try:
        # --- MODIFIED: Load the specific checkpoint ---
        print(f"Loading weights from {model_path}...")
        models = np.load(model_path)
        q_ee = models['q_ee']
        q_ae = models['q_ae']
        q_ea = models['q_ea']
        q_aa = models['q_aa']
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found.")
        return

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visualizing Fully Trained QRM-SG Agents")
    clock = pygame.time.Clock()

    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')
    rm_map = {'start': 0, 'v_1': 1, 'v_2': 2, 'v_3': 3, 'v_end': 4}

    for episode in range(1, num_episodes + 1):
        pos_e, pos_a = env.reset()
        rm_ego.reset()
        rm_adv.reset()
        
        print(f"--- Playing Test Episode {episode} ---")
        
        # --- INITIAL RENDER: Draw the starting frame and pause ---
        screen.fill(BG_COLOR)
        draw_grid(screen)
        draw_base(screen, env.base_a, ADV_COLOR)
        draw_base(screen, env.base_e, EGO_COLOR)
        draw_ghost(screen, pos_e, EGO_COLOR)
        draw_ghost(screen, pos_a, ADV_COLOR)
        pygame.display.flip()
        
        # Freeze the frame for 1.5 seconds before starting
        pygame.time.wait(500)
        # ---------------------------------------------------------
        
        for step in range(100):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_e, v_a = rm_map[rm_ego.state], rm_map[rm_adv.state]

            # STRICT EXPLOITATION (No epsilon randomness)
            pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
            pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
            
            action_a = np.argmax(pi_a_adv)
            action_e = np.argmax(pi_e_ego)

            # --- 1. ENVIRONMENT STEP ---
            pos_e, pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            # --- 2. REWARD MACHINE STEP (CALL THIS EXACTLY ONCE!) ---
            _, r_e = rm_ego.step(labels)
            _, r_a = rm_adv.step(labels)

            # --- 3. RENDER FRAME ---
            screen.fill(BG_COLOR)
            draw_grid(screen)
            draw_base(screen, env.base_a, ADV_COLOR)
            draw_base(screen, env.base_e, EGO_COLOR)
            draw_ghost(screen, pos_e, EGO_COLOR)
            draw_ghost(screen, pos_a, ADV_COLOR)
            pygame.display.flip()
            clock.tick(FPS)

            # --- 4. ENDGAME CHECKS ---
            # Check if the Reward Machine declared a winner
            if rm_ego.state == 'v_end' or rm_adv.state == 'v_end':
                if r_e > 0:
                    print("Ego Agent Won!")
                elif r_a > 0:
                    print("Adv Agent Won!")
                else:
                    print("Game Over (Draw)")
                    
                pygame.time.wait(500)
                break
                
            # Check if they just physically crashed into each other early
            # elif 'collision' in labels:
            #     print("Premature Collision (Draw).")
            #     pygame.time.wait(1000)
            #     break

    pygame.quit()

if __name__ == "__main__":
    # Change the filename here to watch different stages of learning
    
    
    # 1. Watch the early naive strategy (Before the pitfall)
    for i in range(1,76): 
        visualize_trained_agents('../EXPORT/q_models.npz', num_episodes=5)
    