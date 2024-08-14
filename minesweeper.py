import numpy as np
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import pygame
import time
import matplotlib.pyplot as plt

# Path to file for training and test networks
TRAINING_NETWORK_PATH = "minesweeper_dql_cnn.pt"
TEST_NETWORK_PATH = "minesweeper_dql_cnn.pt"

# Minesweeper board parameters
LENGTH = 10
TOTAL_MINES = 14

# Initialized for rendering minesweeper so user can see neural network playing the game
CELL_WIDTH = 32
pygame.init()
pygame.display.set_mode((CELL_WIDTH*LENGTH, CELL_WIDTH*LENGTH))
unrevealedPic = pygame.image.load("./assets/Grid.png").convert()
grid0 = pygame.image.load("./assets/empty.png").convert()
grid1 = pygame.image.load("./assets/grid1.png").convert()
grid2 = pygame.image.load("./assets/grid2.png").convert()
grid3 = pygame.image.load("./assets/grid3.png").convert()
grid4 = pygame.image.load("./assets/grid4.png").convert()
grid5 = pygame.image.load("./assets/grid5.png").convert()
grid6 = pygame.image.load("./assets/grid6.png").convert()
grid7 = pygame.image.load("./assets/grid7.png").convert()
grid8 = pygame.image.load("./assets/grid8.png").convert()
minePic = pygame.image.load("./assets/mineClicked.png").convert()

# indexes for tensor input
UNREVEALED = 9
MINES = 10

# Memory to store previous states and actions
MEMORY_LENGTH = 1000
SAMPLE_SIZE = 40

memoryBuffer = deque([], maxlen=MEMORY_LENGTH)
priorityMemoryBuffer = deque([], maxlen=MEMORY_LENGTH)

# Agent parameters
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0
SYNC_RATE = 200

def sample():
    return random.sample(memoryBuffer, SAMPLE_SIZE)

def prioritySample():
    return random.sample(priorityMemoryBuffer, SAMPLE_SIZE)

# Environment to manage minesweeper games
class MinesweeperEnvironment():
    def __init__(self, render):
        self.render = render
        if self.render:
            self.scrn = pygame.display.set_mode((CELL_WIDTH*LENGTH, CELL_WIDTH*LENGTH))
        self.length = LENGTH
        self.total_cells = LENGTH*LENGTH
        self.total_mines = TOTAL_MINES
        self.reset()

    def reset(self):
        self.board = np.ones((self.length, self.length), dtype=int) # mine = -2, revealed/unrevealed -> unrevealed = -1, revealed = nums from 0-8
        self.board *= -1
        self.state = torch.zeros(1, 11, LENGTH, LENGTH)
        self.state[0][UNREVEALED] = torch.ones(LENGTH, LENGTH)
        self.unrevealed = np.ones((self.length, self.length), dtype=int)
        self.mines = np.zeros((self.length, self.length), dtype=int)
        self.generate_mines()
        self.revealed_tiles = 0
        self.game_done = False
        self.step_count = 0
        if self.render:
            for row in range(LENGTH):
                for col in range(LENGTH):
                    self.scrn.blit(unrevealedPic, (col*CELL_WIDTH, row*CELL_WIDTH))
            pygame.display.flip()
        return self.state
        
    def generate_mines(self):
        mine_locations = np.random.choice(self.total_cells, self.total_mines, replace=False)
        for ind in mine_locations:
            row, col = divmod(ind, self.length)
            self.mines[row, col] = 1

    def count_mines_around_cell(self, row, col):
        count = 0

        bottomRow = (row + 1 == self.length)
        topRow = (row == 0)
        rightmostCol = (col + 1 == self.length)
        leftmostCol = (col == 0)

        if not bottomRow:
            count += self.mines[row + 1, col]
        if not topRow:
            count += self.mines[row - 1, col]
        if not rightmostCol:
            count += self.mines[row, col + 1]
        if not leftmostCol:
            count += self.mines[row, col - 1]
        if not (bottomRow or rightmostCol):
            count += self.mines[row + 1, col + 1]
        if not (bottomRow or leftmostCol):
            count += self.mines[row + 1, col - 1]
        if not (topRow or rightmostCol):
            count += self.mines[row - 1, col + 1]
        if not (topRow or leftmostCol):
            count += self.mines[row - 1, col - 1]
        return count
    
    def reveal(self, row, col):
        self.board[row, col] = self.count_mines_around_cell(row, col)
        self.state[0][UNREVEALED][row][col] = 0
        self.state[0][self.board[row, col]][row][col] = 1
        self.unrevealed[row, col] = 0
        self.revealed_tiles += 1
        self.renderCell(row, col, self.board[row, col])

        # Recursively call reveal() on all cells surrounding 0 cell
        if self.board[row, col] == 0:
            bottomRow = (row + 1 == self.length)
            topRow = (row == 0)
            rightmostCol = (col + 1 == self.length)
            leftmostCol = (col == 0)

            if not bottomRow and self.unrevealed[row + 1, col]:
                self.reveal(row + 1, col)
            if not topRow and self.unrevealed[row - 1, col]:
                self.reveal(row - 1, col)
            if not rightmostCol and self.unrevealed[row, col + 1]:
                self.reveal(row, col + 1)
            if not leftmostCol and self.unrevealed[row, col - 1]:
                self.reveal(row, col - 1)
            if not (bottomRow or rightmostCol) and self.unrevealed[row + 1, col + 1]:
                self.reveal(row + 1, col + 1)
            if not (bottomRow or leftmostCol) and self.unrevealed[row + 1, col - 1]:
                self.reveal(row + 1, col - 1)
            if not (topRow or rightmostCol) and self.unrevealed[row - 1, col + 1]:
                self.reveal(row - 1, col + 1)
            if not (topRow or leftmostCol) and self.unrevealed[row - 1, col - 1]:
                self.reveal(row - 1, col - 1)

    def renderCell(self, row, col, cell):
        if not self.render:
            return 
        
        match cell:
            case -1:
                self.scrn.blit(unrevealedPic, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 0:
                self.scrn.blit(grid0, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 1:
                self.scrn.blit(grid1, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 2:
                self.scrn.blit(grid2, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 3:
                self.scrn.blit(grid3, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 4:
                self.scrn.blit(grid4, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 5:
                self.scrn.blit(grid5, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 6:
                self.scrn.blit(grid6, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 7:
                self.scrn.blit(grid7, (col*CELL_WIDTH, row*CELL_WIDTH))
            case 8:
                self.scrn.blit(grid8, (col*CELL_WIDTH, row*CELL_WIDTH))

        pygame.display.flip()

    def step(self, action):
        self.step_count += 1
        row, col = divmod(action, self.length)

        if self.actionIsGuess(row, col):
            reward = -0.3
            if self.mines[row, col]:
                self.board[row, col] = -2
                self.state[0][UNREVEALED][row][col] = 0
                self.state[0][MINES][row][col] = 1
                if self.render:
                    self.scrn.blit(minePic, (col*CELL_WIDTH, row*CELL_WIDTH))
                    pygame.display.flip()
                self.game_done = True
            else:
                self.reveal(row, col)
                if self.revealed_tiles == (self.total_cells - self.total_mines):
                    reward = 1.0
                    self.game_done = True
        elif self.mines[row, col]: # if mine found
            self.board[row, col] = -2
            self.state[0][UNREVEALED][row][col] = 0
            self.state[0][MINES][row][col] = 1
            if self.render:
                self.scrn.blit(minePic, (col*CELL_WIDTH, row*CELL_WIDTH))
                pygame.display.flip()
            self.game_done = True
            reward = -1.0
        else:
            reward = 1.0
            self.reveal(row, col)
            if self.revealed_tiles == (self.total_cells - self.total_mines):
                self.game_done = True
        
        return self.state, reward, self.game_done, self.step_count, self.revealed_tiles
    
    # Checks if any of the surrounding cells are revealed to penalize guesses
    def actionIsGuess(self, row, col):
        bottomRow = (row + 1 == self.length)
        topRow = (row == 0)
        rightmostCol = (col + 1 == self.length)
        leftmostCol = (col == 0)

        if not bottomRow and not self.unrevealed[row + 1, col]:
            return False
        if not topRow and not self.unrevealed[row - 1, col]:
            return False
        if not rightmostCol and not self.unrevealed[row, col + 1]:
            return False
        if not leftmostCol and not self.unrevealed[row, col - 1]:
            return False
        if not (bottomRow or rightmostCol) and not self.unrevealed[row + 1, col + 1]:
            return False
        if not (bottomRow or leftmostCol) and not self.unrevealed[row + 1, col - 1]:
            return False
        if not (topRow or rightmostCol) and not self.unrevealed[row - 1, col + 1]:
            return False
        if not (topRow or leftmostCol) and not self.unrevealed[row - 1, col - 1]:
            return False

        return True

class DQN(nn.Module):
    def __init__(self, input_shape, out_actions):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU()
        # )

        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            # nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # x = self.conv_block3(x)
        x = self.layer_stack(x)
        return x

# Minesweeper Deep Q-Learning
class MinesweeperDQLAgent():
    def __init__(self):
        self.ACTIONS = range(LENGTH*LENGTH)
        # Loss function(Mean Squared Error)
        self.loss_fn = nn.MSELoss()

    def train(self, episodes, render=True, continueTraining=0):
        pygame.display.set_caption('Training...')
        env = MinesweeperEnvironment(render)
        num_actions = LENGTH*LENGTH

        # 100% probability to do a random action
        epsilon = 1

        policy_dqn = DQN(input_shape=11, out_actions=num_actions)
        target_dqn = DQN(input_shape=11, out_actions=num_actions)

        if continueTraining:
            policy_dqn.load_state_dict(torch.load("minesweeper_dql_cnn.pt"))
        # Copy policy network to target network
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=LEARNING_RATE)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        total_steps = 0
        steps = 0
        score_history = []

        for i in range(continueTraining, episodes):
            state = env.reset()
            gameDone = False
            stepsPerGame = 0 # True when agent takes more than 200 actions

            new_state, reward, gameDone, stepsPerGame, score = env.step(45)

            state = new_state
            steps += 1
            total_steps += 1

            if reward == 1:
                rewards_per_episode[i] += 1

            while(not gameDone and not stepsPerGame == 200):
                # select random action based on epsilon value
                if random.random() < epsilon:
                    action = random.sample(self.ACTIONS, 1)[0]
                else:
                    # select best action
                    with torch.no_grad():
                        temp = policy_dqn(state)
                        for a in range(LENGTH*LENGTH):
                            row, col = divmod(a, LENGTH)
                            if not env.unrevealed[row, col]:
                                temp[0][a] = -10
                        action = temp.argmax().item()

                new_state, reward, gameDone, stepsPerGame, score = env.step(action)
                row, col = divmod(action, LENGTH)
                # print(f"Episode: {i}, Row: {row}, Column: {col}, Reward: {reward}, Score: {score}, Done: {gameDone}")
                # Testing to see if neural network learns better when the initial guess is not saved ????
                # And when picking a revealed cell is not saved

                memoryBuffer.append((state, action, new_state, reward, gameDone))
                if reward == 1 or reward == -1:
                    priorityMemoryBuffer.append((state, action, new_state, reward, gameDone))

                state = new_state
                steps += 1
                total_steps += 1

                if reward == 1:
                    rewards_per_episode[i] += 1

            if (i + 1) % 100 == 0:
                print(f"Episode: {i + 1}, Total steps: {total_steps}, Total rewards: {np.sum(rewards_per_episode)}")
                # if(i + 1) % 1000 == 0:
                #     torch.save(policy_dqn.state_dict(), "minesweeper_dql_cnn.pt")
                #     !cp -r './minesweeper_dql_cnn.pt' '/content/gdrive/My Drive/MinesweeperResults/minesweeper_dql_cnn.pt'
                #     with open(f'/content/gdrive/My Drive/MinesweeperResults/LatestEpisode.txt', 'w') as f:
                #         f.write(f'{i + 1}')

            score_history.append(score)

            if len(priorityMemoryBuffer) > MEMORY_LENGTH/2:
                mini_batch = prioritySample()
                self.optimize(mini_batch, policy_dqn, target_dqn)

            if len(memoryBuffer) > MEMORY_LENGTH/2:
                mini_batch = sample()
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if steps > SYNC_RATE:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps = 0

        torch.save(policy_dqn.state_dict(), "minesweeper_dql_cnn.pt")

        # Create new graph
        plt.figure(1)
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(score_history)
        plt.title("Scores")

        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon decay")

        # Save plots
        plt.savefig('minesweeper_dql_cnn.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + DISCOUNT_RATE * target_dqn(new_state).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state)
            
            # Adjust the specific action to the target that was just calculated. 
            # Target_q[batch][action], hardcode batch to 0 because there is only 1 batch.
            target_q[0][action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Run the Minesweeper environment with the learned policy
    def test(self, episodes, render=True):
        pygame.display.set_caption('Testing')
        env = MinesweeperEnvironment(render)
        num_actions = LENGTH*LENGTH

        # Load learned policy
        policy_dqn = DQN(input_shape=11, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load(TEST_NETWORK_PATH))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            state = env.reset()
            gameDone = False
            stepsPerGame = 0 # True when agent takes more than 200 actions

            state, reward, gameDone, stepsPerGame, score = env.step(45)
            row, col = divmod(45, LENGTH)
            print(f"Episode: {i}, Row: {row}, Column: {col}, Reward: {reward}, Score: {score}, Done: {gameDone}")
            time.sleep(1)

            while(not gameDone and not stepsPerGame == 200):
                # Select best action
                with torch.no_grad():
                    temp = policy_dqn(state)
                    for a in range(LENGTH*LENGTH):
                        row, col = divmod(a, LENGTH)
                        if not env.unrevealed[row, col]:
                            temp[0][a] = -10
                    action = temp.argmax().item()

                # Execute action
                state, reward, gameDone, stepsPerGame, score = env.step(action)
                row, col = divmod(action, LENGTH)
                print(f"Episode: {i}, Row: {row}, Column: {col}, Reward: {reward}, Score: {score}, Done: {gameDone}")
                time.sleep(1)


if __name__ == "__main__":
    minesweeper = MinesweeperDQLAgent()
    # minesweeper.train(1000)
    minesweeper.test(10)