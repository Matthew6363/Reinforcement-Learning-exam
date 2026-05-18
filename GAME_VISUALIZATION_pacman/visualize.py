import pygame
import sys
import numpy as np
import os
from PIL import Image

SAVE_GIF = True

code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'CODE_pacman'))
sys.path.append(code_dir)

######### PROBLEM-SPECIFIC REWARD MACHINE and ENV #######################
#########################################################################
# We need to generalize the structure for any reward machine. 
# Having defined the current one of interest, please import it.

from game_parameters import *
from environment import *
from reward_machine_pacman import *
    

task_name_string = TASK # tag which is used in exported files

# Some requirements must be fulfilled: 
# * Reward_Machine [class]
# * solve_stage_game [function]
# View reward_machine_TaskI.py for reference.
# The environment too has to be tuned in the get_labels function.
#########################################################################

         

def draw_grid(screen):
    for x in range(0, WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x + MARGIN, MARGIN), (x + MARGIN, HEIGHT + MARGIN), 2)
    for y in range(0, HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (MARGIN, y + MARGIN), (WIDTH + MARGIN, y + MARGIN), 2)

def draw_base(screen, pos, color, name):
    r, c = pos
    surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)    
    center_on_surface = (CELL_SIZE // 2, CELL_SIZE // 2)
    radius = CELL_SIZE // 4
    pygame.draw.circle(surface, (*color, BASE_ALPHA), center_on_surface, radius)
    screen.blit(surface, (c * CELL_SIZE + MARGIN, r * CELL_SIZE + MARGIN))
    font = pygame.font.Font(None, 30)
    img = font.render(name, True, color)
    text_rect = img.get_rect(center=(c * CELL_SIZE + CELL_SIZE // 2 + MARGIN, 
                                     r * CELL_SIZE + CELL_SIZE // 2 + MARGIN))
    screen.blit(img, text_rect)

def draw_adv_start(screen, pos, color, name):
    r, c = pos
    surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)    
    gray_color = (128, 128, 128, BASE_ALPHA)
    
    surface.fill(gray_color)
    screen.blit(surface, (c * CELL_SIZE + MARGIN, r * CELL_SIZE + MARGIN))
    
    font = pygame.font.Font(None, 30)
    img = font.render(name, True, color)
    text_rect = img.get_rect(center=(c * CELL_SIZE + CELL_SIZE // 2 + MARGIN, 
                                     r * CELL_SIZE + CELL_SIZE // 2 + MARGIN))
    screen.blit(img, text_rect)

def draw_dots(screen, to_visit):
    yellow_color = (252, 186, 3) 
    radius = CELL_SIZE // 8      
    
    for pos in to_visit:
        r, c = pos
        cx = c * CELL_SIZE + CELL_SIZE // 2 + MARGIN
        cy = r * CELL_SIZE + CELL_SIZE // 2 + MARGIN
    
        pygame.draw.circle(screen, yellow_color, (cx, cy), radius)


def draw_agent(screen, pos, color):
    r, c = pos
    cx = c * CELL_SIZE + CELL_SIZE // 2 + MARGIN
    cy = r * CELL_SIZE + CELL_SIZE // 2 + MARGIN
    
    size = CELL_SIZE * AGENT_RATIO
    half_size = size / 2
    
    point1 = (cx, cy - half_size)
    point2 = (cx - half_size, cy + half_size)
    point3 = (cx + half_size, cy + half_size)
    
    points = [point1, point2, point3]
    pygame.draw.polygon(screen, color, points)
    black_color = (0, 0, 0)
    pygame.draw.polygon(screen, black_color, points, 3)


def pos_to_idx(pos): 
    return pos[0]*6 + pos[1]



def visualize_trained_agents(model_path='../CODE/EXPORT/q_models.npz', 
                             checkpoint=0,
                             time_waiting=20):
    
    ## LOADING Q-tables
    try:
        # Load the checkpoint
        print(f"Loading weights from {model_path}...")
        
        models = np.load(model_path)
        q_ee = models['q_ee']
        q_ae = models['q_ae']
        q_ea = models['q_ea']
        q_aa = models['q_aa']

    except FileNotFoundError:
        print(f"Error: '{model_path}' not found.")
        return

    ## START the game
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Visualizing Fully Trained QRM-SG Agents")
    clock = pygame.time.Clock()

    ## Initialize grid env and RMs + states for game and RM
    env = PacmanGridWorld()
    rm_ego = Reward_Machine('ego')
    rm_adv = Reward_Machine('adv')

    if SAVE_GIF:
        frames = []
    
    exclude_states = {env.start_pos_a, env.start_pos_e, env.base_e}
    to_visit = [
            (i, j)
            for i in range(WINDOW_W)
            for j in range(WINDOW_H)
            if (i, j) not in set(env.exclude_states)
    ]
    
    rm_map = {'start': 0, 'v_1': 1, 'v_2': 2,'v_win': 3, 'v_end': 4} 
    
    pos_e, pos_a = env.reset()
    rm_ego.reset()
    rm_adv.reset()
    
    ## Initial Screen Game Render
    screen.fill(BG_COLOR)
    draw_grid(screen)
    draw_adv_start(screen, env.start_pos_a, ADV_COLOR, "A")
    draw_base(screen, env.base_e, EGO_COLOR, "E")
    draw_agent(screen, pos_e, EGO_COLOR) # blue
    draw_agent(screen, pos_a, ADV_COLOR) # red
    pygame.display.flip()
    
    # Freeze the frame for 1.5 seconds before starting
    pygame.time.wait(500)
    

    
    for step in range(STEP_NUM):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        ## States
        s_e, s_a = pos_to_idx(pos_e), pos_to_idx(pos_a)
        v_e, v_a = rm_map[rm_ego.state], rm_map[rm_adv.state]
        
        ## update the dots list
        if pos_e in to_visit: to_visit.remove(pos_e)

        ## Rendering (include first case)
        screen.fill(BG_COLOR)
        draw_grid(screen)
        draw_dots(screen, to_visit)
        draw_adv_start(screen, env.start_pos_a, ADV_COLOR, "A")
        draw_base(screen, env.base_e, EGO_COLOR, "E")
        draw_agent(screen, pos_e, EGO_COLOR)
        draw_agent(screen, pos_a, ADV_COLOR)
        pygame.display.flip()
        clock.tick(FPS)

        if SAVE_GIF:
            frame_str = pygame.image.tobytes(screen, 'RGB')
            frame_img = Image.frombytes('RGB', screen.get_size(), frame_str)
            frames.append(frame_img)

        ## Get the action (thanks to off-policy, we can avoid epsilon randomness)
        pi_e_ego, pi_a_ego = solve_stage_game(q_ee[s_e, s_a, v_e, v_a], q_ae[s_e, s_a, v_e, v_a])
        pi_e_adv, pi_a_adv = solve_stage_game(q_ea[s_e, s_a, v_e, v_a], q_aa[s_e, s_a, v_e, v_a])
        action_a = np.argmax(pi_a_adv)
        action_e = np.argmax(pi_e_ego)

        ## Get new states and RM current label
        pos_e, pos_a = env.step(action_e, action_a)
        labels = env.get_labels()
        
        ## Get RM reward thanks to d(current RM state, current label)
        _, r_e = rm_ego.step(labels)
        _, r_a = rm_adv.step(labels)
        

        is_terminal = (rm_ego == 'v_win')

        ## Is this the last step? (i.e. Has someone won?)
        if is_terminal: #rm_ego.state == 'v_end' or rm_adv.state == 'v_end':

            ## Rendering (include first case)
            screen.fill(BG_COLOR)
            draw_grid(screen)
            draw_dots(screen, to_visit)
            draw_adv_start(screen, env.start_pos_a, ADV_COLOR, "A")
            draw_base(screen, env.base_e, EGO_COLOR, "E")
            draw_agent(screen, pos_e, EGO_COLOR)
            draw_agent(screen, pos_a, ADV_COLOR)
            pygame.display.flip()
            clock.tick(FPS)
            
            if r_e > 0:
                print("Ego Agent Won!")
            elif r_a > 0:
                print("Adv Agent Won!")
            else:
                print("Game Over (Draw)")

            if SAVE_GIF:
                frame_str = pygame.image.tobytes(screen, 'RGB')
                frame_img = Image.frombytes('RGB', screen.get_size(), frame_str)
                frames.append(frame_img)
            
            pygame.time.wait(time_waiting)
            break
            
        # Check if they just physically crashed into each other early
    
    if SAVE_GIF:
        frame_duration = int(1000 / FPS)     
        frames[0].save(f'./Exported_gifs/{FILE_NAME}_simulation.gif', 
                    format='GIF',
                    append_images=frames[0:],
                    save_all=True,
                    duration=frame_duration,
                    loop=0) # inf loop

    pygame.quit()



if __name__ == "__main__":

    #FILE_NAME = 'q_models_task_II_ep6500'
    #FILE_NAME = 'q_models_task_I_ep16000'
    FILE_NAME = 'q_models_pacman_ep16000'
    
    visualize_trained_agents(f"../EXPORT/{FILE_NAME}.npz", 
                             checkpoint=0,
                             time_waiting=3000)
                             