import torch 
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 
BATCH_SIZE = 1000
LR = 0.001 # learning rate


# steps: store game, model
# state = get_state(game)
# action = get_move(state)  .. model.predict()
# reward, game_over, score = game.play_step(action)
# new_state = get_state(game)
# remember
# model.train()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # control the randomness
        self.gamma =  0.9 # discount rate (smaller than 1!)
        self.memory = deque(maxlen = MAX_MEMORY) # popleft automatically if exceed limit
        self.model = Linear_QNet(11, 256, 3) # 11 - size of states, output = 3 (3 diff nr in our actions)
        self.trainer = QTrainer(self.model, lr = LR, gammma=self.gamma)

    def get_state(self, game): 
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y) # 20 for the block size
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            dir_l, 
            dir_r,
            dir_u, 
            dir_d,

            # food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y < game.head.y, # food down
        ]

        return np.array(state, dtype = int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # store as tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample) # put all of them together <=> for all in mini_sample: train
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    

    # decide whether to explore (try random actions) or exploit (use the trained model to choose the best action)
    # return the action as a list final_move, where one element is 1 (the chosen move), and the rest are 0
    def get_action(self, state):
        # random moves: tradeoff exploration/ exploitation
        # use epsilon
        # the more games we have the smaller epsilon
        self.epsilon = 80 - self.n_games # hardcode
        final_move = [0, 0, 0]

        # over time, epsilon decreases, leading to more exploitation of the model's knowledge.
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # if the agent does not explore, it exploits its model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # model will have some floats and we need the biggest one converted to 1 and the rest to 0
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get the old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory <=> for one step
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory = experience replay memory
            # plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game', agent.n_games, 'Score:', score, 'Record:', record)

            # append the scores
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()
