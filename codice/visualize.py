import pygame
import sys
import numpy as np

# Import your working modules
from environment import PacmanGridWorld
from reward_machine import Reward_Machine, solve_stage_game

# --- PYGAME CONFIGURATION ---
CELL_SIZE = 100
GRID_SIZE = 6
WIDTH = CELL_SIZE * GRID_SIZE
HEIGHT = CELL_SIZE * GRID_SIZE
FPS = 10 # Slightly faster visualization

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

def train_and_visualize(total_episodes=7500, render_every=100):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("QRM-SG Decentralized Learning (4 Matrices)")
    clock = pygame.time.Clock()

    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')
    
    shape = (36, 36, 5, 5, 4, 4)
    
    # Ego Agent's Brain (Estimates Ego's rewards and Adv's rewards)
    q_ee = np.zeros(shape)
    q_ae = np.zeros(shape)
    
    # Adversary Agent's Brain (Estimates Ego's rewards and its own rewards)
    q_ea = np.zeros(shape)
    q_aa = np.zeros(shape)
    
    def pos_to_idx(pos): return pos[0]*6 + pos[1]
    rm_map = {'start': 0, 'v_1': 1, 'v_2': 2, 'v_3': 3, 'v_end': 4}
    
    gamma = 0.9 
    epsilon = 0.25 
    alpha = 0.1 
    successes = 0

    print(f"Starting Training. Rendering every {render_every} episodes...")

    for episode in range(1, total_episodes + 1):
        pos_e, pos_a = env.reset()
        rm_ego.reset()
        rm_adv.reset()
        
        render_this_episode = (episode % render_every == 0) or (episode == 1)
        
        if render_this_episode:
            print(f"--- WATCHING EPISODE {episode} ---")

        for step in range(500):
            if render_this_episode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

            s_1, s_2 = pos_to_idx(pos_e), pos_to_idx(pos_a)
            v_1, v_2 = rm_map[rm_ego.state], rm_map[rm_adv.state]
            
            # --- 1. DECENTRALIZED ACTION SELECTION ---
            if np.random.rand() < epsilon:
                action_e = int(np.random.choice(4))
                action_a = int(np.random.choice(4))
            else:
                # Ego uses its internal matrices to solve the game
                pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_1, s_2, v_1, v_2], q_ae[s_1, s_2, v_1, v_2])
                action_e = np.random.choice(4, p=pi_e_ego)
                
                # Adversary uses its own separate internal matrices to solve the game
                pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_1, s_2, v_1, v_2], q_aa[s_1, s_2, v_1, v_2])
                action_a = np.random.choice(4, p=pi_a_adv)
                
            # --- 2. ENVIRONMENT STEP ---
            next_pos_e, next_pos_a = env.step(action_e, action_a)
            labels = env.get_labels()
            
            next_ve_str, r_e = rm_ego.step(labels) 
            next_va_str, r_a = rm_adv.step(labels)
            
            ns_1, ns_2 = pos_to_idx(next_pos_e), pos_to_idx(next_pos_a)
            nv_1, nv_2 = rm_map[next_ve_str], rm_map[next_va_str]
            
            # --- 3. DECENTRALIZED Q-TABLE UPDATES ---
            is_terminal = (next_ve_str == 'v_end') or (next_va_str == 'v_end') or ('collision' in labels)
            
            if is_terminal:
                v_boot_ee, v_boot_ae = 0.0, 0.0
                v_boot_ea, v_boot_aa = 0.0, 0.0
                
            else:
                # Ego agent calculates future expectations based on its brain
                pi_next_e_ego, pi_next_a_ego = solve_stage_game(q_ee[ns_1, ns_2, nv_1, nv_2], q_ae[ns_1, ns_2, nv_1, nv_2])
                v_boot_ee = pi_next_e_ego @ q_ee[ns_1, ns_2, nv_1, nv_2] @ pi_next_a_ego
                v_boot_ae = pi_next_e_ego @ q_ae[ns_1, ns_2, nv_1, nv_2] @ pi_next_a_ego
                
                # Adversary calculates future expectations based on its brain
                pi_next_e_adv, pi_next_a_adv = solve_stage_game(q_ea[ns_1, ns_2, nv_1, nv_2], q_aa[ns_1, ns_2, nv_1, nv_2])
                v_boot_ea = pi_next_e_adv @ q_ea[ns_1, ns_2, nv_1, nv_2] @ pi_next_a_adv
                v_boot_aa = pi_next_e_adv @ q_aa[ns_1, ns_2, nv_1, nv_2] @ pi_next_a_adv
            
            # Update Ego's Matrices
            q_ee[s_1,s_2,v_1,v_2,action_e,action_a] = (1 - alpha) * q_ee[s_1,s_2,v_1,v_2,action_e,action_a] + \
                                        alpha * (r_e + gamma * v_boot_ee)
            q_ae[s_1,s_2,v_1,v_2,action_e,action_a] = (1 - alpha) * q_ae[s_1,s_2,v_1,v_2,action_e,action_a] + \
                                        alpha * (r_a + gamma * v_boot_ae)
                                        
            # Update Adversary's Matrices
            q_ea[s_1,s_2,v_1,v_2,action_e,action_a] = (1 - alpha) * q_ea[s_1,s_2,v_1,v_2,action_e,action_a] + \
                                        alpha * (r_e + gamma * v_boot_ea)
            q_aa[s_1,s_2,v_1,v_2,action_e,action_a] = (1 - alpha) * q_aa[s_1,s_2,v_1,v_2,action_e,action_a] + \
                                        alpha * (r_a + gamma * v_boot_aa)
            
            pos_e, pos_a = next_pos_e, next_pos_a
            
            # --- RENDER LOGIC ---
            if render_this_episode:
                screen.fill(BG_COLOR)
                draw_grid(screen)
                draw_base(screen, env.base_a, ADV_COLOR)
                draw_base(screen, env.base_e, EGO_COLOR)
                draw_ghost(screen, pos_e, EGO_COLOR)
                draw_ghost(screen, pos_a, ADV_COLOR)
                
                pygame.display.flip()
                clock.tick(FPS) 

            # --- 4. TERMINATION CHECK ---
            if rm_ego.state == 'v_end':
                successes += 1
                if render_this_episode: 
                    print("Ego Agent Won! Pausing briefly...")
                    pygame.time.wait(1000)
                break
            elif rm_adv.state == 'v_end':
                if render_this_episode:
                    print("Adversary Won! Pausing briefly...")
                    pygame.time.wait(1000)
                break
            elif 'collision' in labels:
                if render_this_episode:
                    print("Premature Collision! Pausing briefly...")
                    pygame.time.wait(500)
                break
            
        if episode % 100 == 0:
            win_rate = (successes / 100) * 100
            print(f"Episodes {episode-99:04d} to {episode:04d} | Ego Agent Win Rate: {win_rate:.1f}%")
            successes = 0

    print("Training Complete.")
    pygame.quit()
    return q_ee, q_ae, q_ea, q_aa

if __name__ == "__main__":
    q_ee, q_ae, q_ea, q_aa = train_and_visualize(total_episodes=7500, render_every=100)