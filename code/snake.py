from math import *
import random

import pygame
import numpy as np

GRID_SIZE = 20
WINDOW_SIZE = 400
COLORS = {
    'background': (0, 0, 0),
    'snake': (0, 255, 0),
    'food': (255, 0, 0),
    'text': (255, 255, 255)
}


class Snake:
    def __init__(self, grid_size=20, render=False):
        self.grid_size = grid_size
        self.cell_size = WINDOW_SIZE // grid_size
        self.render = render
        self.limit_steps = 1000
        if self.render:
            pygame.init()
            self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
            pygame.display.set_caption("Snake DQN")

        self.reset()

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)
        self.food = self._place_food()
        self.done = False
        self.steps_since_apple = 0  # Сбрасываем таймер голода
        self.count_apples = 0
        self.reward = 0

        return self._get_state()

    def _place_food(self):

        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction

        dir_front = (dir_x, dir_y)
        dir_left = (-dir_y, dir_x)
        dir_right = (dir_y, -dir_x)

        def is_unsafe(point):
            x, y = point
            if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
                return 1.0

            if (x, y) in self.snake:
                return 1.0
            return 0.0

        danger_front = is_unsafe(self._next_position(head_x, head_y, dir_front))
        danger_left = is_unsafe(self._next_position(head_x, head_y, dir_left))
        danger_right = is_unsafe(self._next_position(head_x, head_y, dir_right))

        rel_food_x = (self.food[0] - head_x) / self.grid_size
        rel_food_y = (self.food[1] - head_y) / self.grid_size

        wall_front = self._get_wall_dist(head_x, head_y, dir_front) / self.grid_size
        wall_left = self._get_wall_dist(head_x, head_y, dir_left) / self.grid_size
        wall_right = self._get_wall_dist(head_x, head_y, dir_right) / self.grid_size

        state = [
            danger_front, danger_left, danger_right, 
            rel_food_x, rel_food_y,                  
            wall_front, wall_left, wall_right         
        ]

        return np.array(state, dtype=np.float32)

    def _get_wall_dist(self, head_x, head_y, direction):
        dx, dy = direction
        if dx == 1:   return (self.grid_size - 1) - head_x
        if dx == -1: return head_x
        if dy == 1:   return (self.grid_size - 1) - head_y
        return head_y

    def _next_position(self, x, y, direction):
        return x + direction[0], y + direction[1]

    def _left_direction(self):
        dx, dy = self.direction
        return -dy, dx

    def _right_direction(self):
        dx, dy = self.direction
        return dy, -dx

    def step(self, action):
        if self.done:
            return self._get_state(), self.reward, True

        if action == 0:
            new_dir = self._left_direction()
        elif action == 2:
            new_dir = self._right_direction()
        else:
            new_dir = self.direction

        self.direction = new_dir
        new_head = self._next_position(*self.snake[0], self.direction)

        step_reward = 0.0  # Награда ТОЛЬКО за текущий шаг
        self.steps_since_apple += 1

        # Проверка столкновения со стеной или хвостом
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True
            return self._get_state(), -10.0, True

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.food = self._place_food()
            self.count_apples += 1
            self.steps_since_apple = 0  
            step_reward = 10.0 
        else:
            self.snake.pop()
            step_reward = -0.05  

        if self.steps_since_apple >= self.limit_steps:
            self.done = True
            step_reward = -10.0

        self.reward += round(step_reward, 2)

        if self.render:
            self.render_frame()

        return self._get_state(), step_reward, self.done

    def render_frame(self):
        self.screen.fill(COLORS['background'])

        for segment in self.snake:
            x, y = segment
            pygame.draw.rect(self.screen, COLORS['snake'],
                             (x * self.cell_size, y * self.cell_size,
                              self.cell_size - 1, self.cell_size - 1))

        fx, fy = self.food
        pygame.draw.rect(self.screen, COLORS['food'],
                         (fx * self.cell_size, fy * self.cell_size,
                          self.cell_size - 1, self.cell_size - 1))

        font = pygame.font.SysFont(None, 30)
        text = font.render(f'Score: {self.reward}', True, COLORS['text'])
        self.screen.blit(text, (10, 10))

        pygame.display.flip()
        pygame.time.wait(50)
    

    def close(self):
        if self.render:
            pygame.quit()
