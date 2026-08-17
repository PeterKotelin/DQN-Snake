import random
from collections import deque
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from snake import SnakeEnv
from buffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(10000)
        self.batch_size = batch_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        with torch.no_grad():
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q = self.target_net(next_states).max(1)[0]
            print(current_q, next_q)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def decay_epsilon(self):
        self.epsilon = max(epsilon_min, self.epsilon * self.epsilon_decay)


# Обучение с визуализацией
if __name__ == "__main__":
    env = SnakeEnv(grid_size=20, render=True)
    state_size = len(env.reset())
    action_size = 3
    batch_size = 32

    episodes_reward = []

    agent = DQNAgent(state_size, action_size, batch_size)
    all_reward = 0
    update_target_freq = 50

    try:
        while all_reward < 1000:
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                all_reward += reward
                state = next_state
                agent.update(batch_size)

                # Обработка событий Pygame
                if env.render:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            env.close()
                            exit()

            agent.decay_epsilon()
            if e % update_target_freq == 0:
                agent.update_target_model()

            print(
                f"Episode: {e + 1}/{episodes}, Score: {env.reward}, Epsilon: {agent.epsilon:.2f}")

            episodes_reward.append(env.reward)
    finally:
        env.close()
        torch.save(agent.policy_net.state_dict(), os.path.join(os.getcwd(),'snake_dqn.pth'))

        plt.plot(np.arange(1, episodes), episodes_reward)