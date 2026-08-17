import os
import random
from collections import deque
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from snake import Snake
from buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, warmup_steps=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(50000)
        self.batch_size = batch_size
        self.gamma = 0.98
        self.learning_rate = 0.0005

        # --- Параметры Exploration ---
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_steps = 100000
        self.total_steps = 0
        self.warmup_steps = warmup_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def load_weights(self, path):
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))


    def act(self, state):
        self.total_steps += 1
        if self.total_steps > self.warmup_steps:    
            decay_rate = (1.0 - self.epsilon_min) / self.epsilon_decay_steps
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_rate)

        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.warmup_steps or len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Клиппинг градиентов для стабильности обучения
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__":
    env = Snake(grid_size=20, render=False)  # render=False ускоряет обучение в разы!
    state_size = len(env.reset())
    action_size = 3
    batch_size = 64

    agent = DQNAgent(state_size, action_size, batch_size)
    num_episodes = 2000
    target_update_freq = 1000
    episodes_scores = []
    max_score = 0

    try:
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            episode_score = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_score += reward

                # Обучение агента на каждом шаге
                agent.update()

                # Обновление Target сети по шагам
                if agent.total_steps % target_update_freq == 0:
                    agent.update_target_model()

                if env.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            exit()

            if episode_score > max_score:
                max_score = episode_score
                torch.save(agent.policy_net.state_dict(), '../model/snake_dqn.pth')

            episodes_scores.append(env.count_apples)

            if episode % 20 == 0:
                avg_apples = np.mean(episodes_scores[-20:])
                print(f"Episode: {episode} | Avg Apples (last 20): {avg_apples:.2f} | "
                      f"Epsilon: {agent.epsilon:.3f} | Steps: {agent.total_steps} | Max Score: {max_score}" )

    finally:
        env.close()
        plt.plot(np.arange(1, num_episodes), episodes_scores)