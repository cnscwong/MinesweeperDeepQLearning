from torch import nn
from collections import deque
import pygame
import numpy as np
import torch
import random

# Minesweeper parameters
HEIGHT = 16
WIDTH = 30
TOTAL_MINES = 99

# Pixel width of each cell
CELL_WIDTH = 32
RENDER = True

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

UNREVEALED = 9
FLAGS = 10
PADDING = 11

# Memory to store previous states and actions
MEMORY_LENGTH = 1000
SAMPLE_SIZE = 40

class DQN(nn.Module):
    def __init__(self, input_shape, out_actions):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
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
        # print(x)
        return x
    
class MemoryBuffer():
    def __init__(self):
        self.memoryBuffer = deque([], maxlen=MEMORY_LENGTH)

    def sample(self):
        return random.sample(self.memoryBuffer, SAMPLE_SIZE)

    def __append__(self, state):
        self.memoryBuffer.append(state)

# Environment to manage minesweeper games
class MinesweeperEnvironment():
    def __init__(self):
        if RENDER:
            self.scrn = pygame.display.set_mode((CELL_WIDTH*WIDTH, CELL_WIDTH*HEIGHT))
        self.total_cells = HEIGHT*WIDTH
        self.total_mines = TOTAL_MINES
        self.game_done = True

    def reset(self):
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
        return self.board
    
    def step(self, action): # Reveal 
        row, col = divmod(action, WIDTH)

        if self.mines[row, col]: # if mine found
            self.board[row, col] = -2
            self.state[0][UNREVEALED][row][col] = 0
            if self.render:
                self.scrn.blit(minePic, (col*CELL_WIDTH, row*CELL_WIDTH))
                pygame.display.flip()
            self.game_done = True
            reward = 0.0
        else:
            reward = 1.0
            self.reveal(row, col)
            if self.revealed_tiles == (self.total_cells - self.total_mines):
                self.game_done = True
        
        return self.state, reward, self.game_done, self.step_count, self.revealed_tiles
    
    def flag(self, action): # Reveal 
        row, col = divmod(action, WIDTH)

        if self.mines[row, col]: # if mine found
            self.board[row, col] = -2
            self.state[0][UNREVEALED][row][col] = 0
            if self.render:
                self.scrn.blit(minePic, (col*CELL_WIDTH, row*CELL_WIDTH))
                pygame.display.flip()
            self.game_done = True
            reward = 0.0
        else:
            reward = 1.0
            self.reveal(row, col)
            if self.revealed_tiles == (self.total_cells - self.total_mines):
                self.game_done = True
        
        return self.state, reward, self.game_done, self.step_count, self.revealed_tiles
    
    def generate_mines(self):
        mine_locations = np.random.choice(self.total_cells, self.total_mines, replace=False)
        for ind in mine_locations:
            row, col = divmod(ind, WIDTH)
            self.mines[row, col] = 1

    def applyPadding(self, row, col):
        tensor = torch.zeros(1,12,5,5)

        if row < 2:
            tensor[0][PADDING][0] = torch.ones(5)

        if row == 0:
            tensor[0][PADDING][1] = torch.ones(5)

        if row > (HEIGHT - 3):
            tensor[0][PADDING][4] = torch.ones(5)

        if row == (HEIGHT - 1):
            tensor[0][PADDING][3] = torch.ones(5)
        
        for i in range(5):
            if col == 0:
                

        return tensor

    def state_to_tensor(self, position):
        row, col = divmod(position, WIDTH)
        tensor = self.applyPadding(row, col)

        return tensor


env = MinesweeperEnvironment()

print(env.state_to_tensor(1))
while 1:
    pass