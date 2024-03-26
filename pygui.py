import pygame
import numpy as np

window_size = (800,450)
scale = 0.05

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RENDER_FPS = 24

def draw_isosceles_triangle(screen, center, base_length, height, orientation):
    center_x, center_y = center

    # Calculate the coordinates of the vertices
    x1 = center_x - height / 2
    y1 = center_y - (base_length / 2)

    x2 = center_x - height / 2
    y2 = center_y + (base_length / 2)

    x3 = center_x + height / 2
    y3 = center_y

    # Rotate the triangle around its center using NumPy
    rotation_matrix = np.array([[np.cos(orientation), -np.sin(orientation)],
                                [np.sin(orientation), np.cos(orientation)]])
    
    vertex1 = np.dot(rotation_matrix, np.array([x1 - center_x, y1 - center_y]))
    vertex2 = np.dot(rotation_matrix, np.array([x2 - center_x, y2 - center_y]))
    vertex3 = np.dot(rotation_matrix, np.array([x3 - center_x, y3 - center_y]))

    x1_rot, y1_rot = vertex1 + center
    x2_rot, y2_rot = vertex2 + center
    x3_rot, y3_rot = vertex3 + center

    # Draw the rotated triangle
    pygame.draw.polygon(screen, BLUE, [(x1_rot, y1_rot), (x2_rot, y2_rot), (x3_rot, y3_rot)])

def drawPod(pod, canvas):
    return draw_isosceles_triangle(canvas, pod.pos * scale, 600*scale, 800*scale, pod.orientation)

def drawCheckpoint(p, canvas):
    return pygame.draw.circle(canvas, BLACK, p*scale, 600*scale)

def render_frame(env):
        if env.window is None and env.render_mode == "human":
            pygame.init()
            pygame.display.init()
            env.window = pygame.display.set_mode(window_size)
        if env.clock is None and env.render_mode == "human":
            env.clock = pygame.time.Clock()

        canvas = pygame.Surface(window_size)
        canvas.fill((255, 255, 255))

        
        for chk in env.checkpoints:
            drawCheckpoint(chk, canvas)
        for pod in env.pods:
            drawPod(pod, canvas)
        
        if env.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            env.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            env.clock.tick(RENDER_FPS)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )    