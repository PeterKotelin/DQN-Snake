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


class SnakeEnv:
    def __init__(self, grid_size=20, render=False):
        self.grid_size = grid_size
        self.cell_size = WINDOW_SIZE // grid_size
        self.render = render
        self.limit_steps = 200
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
        self.reward = 0
        self.steps = 0
        self.count_apples = 0

    def _place_food(self):
        random.seed(128)

        while True:
            food = (random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        state = []
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_x, dir_y = self.direction

        rel_food_x = food_x - head_x
        rel_food_y = food_y - head_y
        evclid_dist = sqrt(rel_food_x**2 + rel_food_y**2)
        
        state.extend([rel_food_x, rel_food_y, evclid_dis])

        # Переменные направления головы
        dir_x, dir_y = self.direction

        # 1. Расстояние ДО СТЕНЫ ВПЕРЕДИ
        if dir_x == 1:    dist_front = (self.grid_size - 1) - head_x  # ползет вправо
        elif dir_x == -1: dist_front = head_x                         # ползет влево
        elif dir_y == 1:  dist_front = (self.grid_size - 1) - head_y  # ползет вниз
        else:             dist_front = head_y                         # ползет вверх

        # 2. Расстояние ДО СТЕНЫ СЛЕВА (относительно движения змейки)
        left_x, left_y = -dir_y, dir_x
        if left_x == 1:    dist_left = (self.grid_size - 1) - head_x
        elif left_x == -1: dist_left = head_x
        elif left_y == 1:  dist_left = (self.grid_size - 1) - head_y
        else:              dist_left = head_y

        # 3. Расстояние ДО СТЕНЫ СПРАВА (относительно движения змейки)
        right_x, right_y = dir_y, -dir_x
        if right_x == 1:    dist_right = (self.grid_size - 1) - head_x
        elif right_x == -1: dist_right = head_x
        elif right_y == 1:  dist_right = (self.grid_size - 1) - head_y
        else:               dist_right = head_y

        # Нормализуем, деля на размер сетки
        wall_distances = [
            dist_front / self.grid_size,
            dist_left / self.grid_size,
            dist_right / self.grid_size
        ]

        state.extend(wall_distance)

        return np.array(state, dtype=np.float32)

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

        # Проверка столкновения
        if (new_head in self.snake or
                new_head[0] < 0 or new_head[0] >= self.grid_size or
                new_head[1] < 0 or new_head[1] >= self.grid_size):
            self.done = True

            if self.steps < 15:
                return self._get_state(), -100.0, True
            else:
                return self._get_state(), -10.0, True

        self.snake.insert(0, new_head)

        self.steps += 1

        if new_head == self.food:
            self.food = self._place_food()
            self.count_apples += 1
            self.reward += sqrt(self.count_apples) * 3.5
        elif self.steps <= self.limit_steps:
            self.snake.pop()
            self.reward -= 0.25
        else:
            self.done = True

        if self.render:
            self.render_frame()

        return self._get_state(), self.reward, self.done

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
