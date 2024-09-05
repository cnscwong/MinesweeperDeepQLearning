from torch import nn
from collections import deque
import pygame
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import time

# Minesweeper parameters
HEIGHT = 16
WIDTH = 30
TOTAL_MINES = 99

# Pixel width of each cell
CELL_WIDTH = 32
RENDER = True

torch.autograd.set_detect_anomaly(True)

if RENDER:
    pygame.init()
    pygame.display.set_mode((CELL_WIDTH*WIDTH, CELL_WIDTH*HEIGHT))
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
    flagPic = pygame.image.load("./assets/flag.jpg").convert()

UNREVEALED = 9
FLAGS = 10
PADDING = 11

# Memory to store previous states and actions
MEMORY_LENGTH = 1000
SAMPLE_SIZE = 40

# Agent parameters
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0
SYNC_RATE = 500

class DQN(nn.Module):
    def __init__(self, input_shape, out_actions):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=(3,3), stride=1, padding=0),
            nn.Sigmoid(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=0),
            nn.Sigmoid()
        )

        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.Sigmoid(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU()
        # )

        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        # x = self.conv_block2(x)
        # x = self.conv_block3(x)
        x = self.layer_stack(x)
        x = torch.sigmoid(x)
        return x.clone()
    
class MemoryBuffer():
    def __init__(self):
        self.memoryBuffer = deque([], maxlen=MEMORY_LENGTH)

    def sample(self):
        return random.sample(self.memoryBuffer, SAMPLE_SIZE)

    def append(self, state):
        self.memoryBuffer.append(state)

    def __len__(self):
        return len(self.memoryBuffer)

# Environment to manage minesweeper games
class MinesweeperEnvironment():
    def __init__(self): # Checked
        if RENDER:
            self.scrn = pygame.display.set_mode((CELL_WIDTH*WIDTH, CELL_WIDTH*HEIGHT))
        self.total_cells = HEIGHT*WIDTH
        self.total_mines = TOTAL_MINES
        self.reset()

    def reset(self): #Checked
        self.board = np.ones((HEIGHT, WIDTH), dtype=int)
        self.board *= UNREVEALED
        self.mines = np.zeros((HEIGHT, WIDTH), dtype=int)
        self.generate_mines()
        self.revealed_tiles = 0
        self.game_done = False
        if RENDER:
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    self.scrn.blit(unrevealedPic, (col*CELL_WIDTH, row*CELL_WIDTH))
            pygame.display.flip()
    
    def count_mines_around_cell(self, row, col): # Checked
        count = 0

        bottomRow = (row + 1 == HEIGHT)
        topRow = (row == 0)
        rightmostCol = (col + 1 == WIDTH)
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
    
    def reveal(self, row, col): # Checked
        self.board[row, col] = self.count_mines_around_cell(row, col)
        self.revealed_tiles += 1
        self.renderCell(row, col)

        # Recursively call reveal() on all cells surrounding 0 cell
        if self.board[row, col] == 0:
            bottomRow = (row + 1 == HEIGHT)
            topRow = (row == 0)
            rightmostCol = (col + 1 == WIDTH)
            leftmostCol = (col == 0)

            if not bottomRow and self.board[row + 1, col] == UNREVEALED:
                self.reveal(row + 1, col)
            if not topRow and self.board[row - 1, col] == UNREVEALED:
                self.reveal(row - 1, col)
            if not rightmostCol and self.board[row, col + 1] == UNREVEALED:
                self.reveal(row, col + 1)
            if not leftmostCol and self.board[row, col - 1] == UNREVEALED:
                self.reveal(row, col - 1)
            if not (bottomRow or rightmostCol) and self.board[row + 1, col + 1] == UNREVEALED:
                self.reveal(row + 1, col + 1)
            if not (bottomRow or leftmostCol) and self.board[row + 1, col - 1] == UNREVEALED:
                self.reveal(row + 1, col - 1)
            if not (topRow or rightmostCol) and self.board[row - 1, col + 1] == UNREVEALED:
                self.reveal(row - 1, col + 1)
            if not (topRow or leftmostCol) and self.board[row - 1, col - 1] == UNREVEALED:
                self.reveal(row - 1, col - 1)

    def firstStep(self, action):
        row, col = divmod(action, WIDTH)

        while self.mines[row, col]:
            self.reset()
        
        self.step(action)

    def step(self, action): # Checked
        row, col = divmod(action, WIDTH)

        if self.mines[row, col]: # if mine found
            if RENDER:
                self.scrn.blit(minePic, (col*CELL_WIDTH, row*CELL_WIDTH))
                pygame.display.flip()
            self.game_done = True
            reward = -1.0
        else:
            reward = 1.0
            self.reveal(row, col)
            if self.revealed_tiles == (self.total_cells - self.total_mines):
                self.game_done = True
        
        return reward, self.game_done, self.revealed_tiles
    
    def flag(self, action): # Reveal 
        row, col = divmod(action, WIDTH)

        self.board[row, col] = FLAGS

        if RENDER:
            self.scrn.blit(flagPic, (col*CELL_WIDTH, row*CELL_WIDTH))
            pygame.display.flip()

        if self.mines[row, col]: # if mine found
            reward = -1.0
        else:
            reward = 1.0
            self.game_done = True
        
        return reward, self.game_done, self.revealed_tiles
    
    def generate_mines(self): # Checked
        mine_locations = np.random.choice(self.total_cells, self.total_mines, replace=False)
        for ind in mine_locations:
            row, col = divmod(ind, WIDTH)
            self.mines[row, col] = 1

    def renderCell(self, row, col): # Checked
        if not RENDER:
            return 
        
        match self.board[row, col]:
            case 9:
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

    def applyPadding(self, row, col): # Checked
        tensor = torch.zeros(1,12,5,5)
        
        for i in range(5):
            if col < 2:
                tensor[0][PADDING][i][0] = 1
            if col == 0:
                tensor[0][PADDING][i][1] = 1
            if col > (WIDTH - 3):
                tensor[0][PADDING][i][4] = 1
            if col == (WIDTH - 1):
                tensor[0][PADDING][i][3] = 1

            if row == 0:
                tensor[0][PADDING][1][i] = 1
            if row < 2:
                tensor[0][PADDING][0][i] = 1
            if row == (HEIGHT - 1):
                tensor[0][PADDING][3][i] = 1
            if row > (HEIGHT - 3):
                tensor[0][PADDING][4][i] = 1

        return tensor

    def state_to_tensor(self, position): # Checked
        row, col = divmod(position, WIDTH)
        tensor = self.applyPadding(row, col)

        for r in range(-2, 3):
            for c in range(-2, 3):
                if tensor[0][PADDING][r + 2][c + 2] != 1:
                    tensor[0][self.board[row + r][col + c]][r + 2][c + 2] = 1

        return tensor
    
    def isAdjacent(self, position):
        row, col = divmod(position, WIDTH)

        if self.board[row, col] != UNREVEALED:
            return False
        
        bottomRow = (row + 1 == HEIGHT)
        topRow = (row == 0)
        rightmostCol = (col + 1 == WIDTH)
        leftmostCol = (col == 0)

        if not bottomRow and self.board[row + 1, col] != UNREVEALED:
            return True
        if not topRow and self.board[row - 1, col] != UNREVEALED:
            return True
        if not rightmostCol and self.board[row, col + 1] != UNREVEALED:
            return True
        if not leftmostCol and self.board[row, col - 1] != UNREVEALED:
            return True
        
        return False

# Minesweeper Deep Q-Learning
class MinesweeperDQLAgent():
    def __init__(self):
        self.ACTIONS = range(HEIGHT*WIDTH)
        # Loss function(Mean Squared Error)
        self.loss_fn = nn.MSELoss()

    def train(self, episodes, continueTraining=0):
        pygame.display.set_caption('Training...')
        env = MinesweeperEnvironment()

        # 100% probability to do a random action
        epsilon = 0

        policy_dqn = DQN(input_shape=12, out_actions=1)
        target_dqn = DQN(input_shape=12, out_actions=1)

        if continueTraining:
            policy_dqn.load_state_dict(torch.load("minesweeper_dql_cnn.pt"))
        # Copy policy network to target network
        target_dqn.load_state_dict(policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=LEARNING_RATE)

        epsilon_history = []
        total_steps = 0
        steps = 0
        score_history = []

        memoryBuffer = MemoryBuffer()

        for i in range(continueTraining, episodes):
            env.reset()
            gameDone = False

            action = random.sample(self.ACTIONS, 1)[0]
            env.firstStep(action)

            while(not gameDone):
                actions = []
                for pos in self.ACTIONS:
                    if env.isAdjacent(pos):
                        actions.append(pos)
                    
                # select random action based on epsilon value
                if random.random() < epsilon:
                    action = random.sample(actions, 1)[0]
                    state = env.state_to_tensor(action)
                    if random.random() < 0.5:
                        reward, gameDone, score = env.flag(action)
                    else:
                        reward, gameDone, score = env.step(action)
                else:
                    # select best action
                    with torch.no_grad():
                        minimum = 1000
                        maximum = -1000

                        for a in actions:
                            curr_state = env.state_to_tensor(a)
                            temp = policy_dqn(curr_state)

                            if temp > maximum:
                                max_state = curr_state
                                max_action = a
                                maximum = temp
                            if temp < minimum:
                                min_state = curr_state
                                min_action = a
                                minimum = temp
                            
                        if (1 - maximum) > minimum:
                            state = min_state
                            reward, gameDone, score = env.flag(min_action)
                        else:
                            state = max_state
                            reward, gameDone, score = env.step(max_action)

                # row, col = divmod(action, WIDTH)
                # print(f"Episode: {i}, Row: {row}, Column: {col}, Reward: {reward}, Score: {score}, Done: {gameDone}")
                # Testing to see if neural network learns better when the initial guess is not saved ????
                # And when picking a revealed cell is not saved

                memoryBuffer.append((state, reward, gameDone))

                steps += 1
                total_steps += 1

            if (i + 1) % 100 == 0:
                print(f"Episode: {i + 1}, Total steps: {total_steps}")
                # if(i + 1) % 1000 == 0:
                #     torch.save(policy_dqn.state_dict(), "minesweeper_dql_cnn.pt")
                #     !cp -r './minesweeper_dql_cnn.pt' '/content/gdrive/My Drive/MinesweeperResults/minesweeper_dql_cnn.pt'
                #     with open(f'/content/gdrive/My Drive/MinesweeperResults/LatestEpisode.txt', 'w') as f:
                #         f.write(f'{i + 1}')

            score_history.append(score)

            if len(memoryBuffer) > MEMORY_LENGTH/2:
                mini_batch = memoryBuffer.sample()
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

        for state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        [reward] 
                        # + DISCOUNT_RATE * target_dqn(new_state).max()
                        # Discount rate set to 0
                    )

            # Get the current set of Q values
            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state)
            
            # Adjust the specific action to the target that was just calculated. 
            # Target_q[batch][action], hardcode batch to 0 because there is only 1 batch.
            target_q[0][0] = target
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
        env = MinesweeperEnvironment()

        # Load learned policy
        policy_dqn = DQN(input_shape=12, out_actions=1)
        policy_dqn.load_state_dict(torch.load("minesweeper_dql_cnn_80000_no_epsilon.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            env.reset()
            gameDone = False

            action = random.sample(self.ACTIONS, 1)[0]
            env.firstStep(action)
            row, col = divmod(action, WIDTH)
            print(f"Episode: {i}, Row: {row}, Column: {col}, Score: {env.revealed_tiles}, Done: {gameDone}")
            time.sleep(1)

            while(not gameDone):
                actions = []
                for pos in self.ACTIONS:
                    if env.isAdjacent(pos):
                        actions.append(pos)
                # select best action
                with torch.no_grad():
                    minimum = 1000
                    maximum = -1000

                    for a in actions:
                        curr_state = env.state_to_tensor(a)
                        temp = policy_dqn(curr_state)

                        if temp > maximum:
                            max_state = curr_state
                            max_action = a
                            maximum = temp
                        if temp < minimum:
                            min_state = curr_state
                            min_action = a
                            minimum = temp
                        
                    if (1 - maximum) > minimum:
                        reward, gameDone, score = env.flag(min_action)
                        row, col = divmod(min_action, WIDTH)
                    else:
                        reward, gameDone, score = env.step(max_action)
                        row, col = divmod(max_action, WIDTH)

                print(f"Episode: {i}, Row: {row}, Column: {col}, Score: {score}, Done: {gameDone}")
                time.sleep(1)

if __name__ == "__main__":
    minesweeper = MinesweeperDQLAgent()
    minesweeper.train(20000)
    # minesweeper.test(10)