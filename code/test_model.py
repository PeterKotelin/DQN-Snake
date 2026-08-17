import pygame

from train import DQNAgent
from snake import Snake

env = Snake(grid_size=20, render=True)
state_size = len(env.reset())
action_size = 3
batch_size = 64

agent = DQNAgent(state_size, action_size, batch_size)
agent.load_weights("/home/asus_pc/snake_RL/DQN-Snake/model/snake_dqn.pth")

done = False
state = env.reset()

while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
                
    state = next_state

    if env.render:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()