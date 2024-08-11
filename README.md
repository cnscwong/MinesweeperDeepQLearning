**Table of content:**
 - [Summary](#item-one)
 - [Try it yourself](#item-two)
 - [Performance Tuning Attempts](#item-three)
 - [Demo](#item-four)

<a id="item-one"></a>
# Summary
Minesweeper solver using Deep Q Learning Convolutional Neural Network

<a id="item-two"></a>
# Try it yourself:
Python version 3.12.3 (Other versions may work)

1. Git clone the repository
2. Install dependencies using:
```
pip install -r /path/to/requirements.txt
```
3. Select one of the pretrained models in the trained_networks folder and set ```TEST_NETWORK_PATH``` in minesweeper.py to the path of the desired network you want to test
4. Change code at bottom to(change 10 to amount of games you want the agent to play):
```
if __name__ == "__main__":
    minesweeper = MinesweeperDQLAgent()
    minesweeper.test(10)
```

<a id="item-three"></a>
# Performance Tuning Attempts
* Set discount rate to 0

    In minesweeper, future turns are not important if the model loses, wanted model to focus on immediate reward and pick the best move as soon as possible

* Reward value changes
    * Punish guesses(Turns that reveal a cell with no information around them)
    * Punish selecting cells that are already revealed
    * Punish selecting a mine(One that is not a guess)
    * Reward selecting a cell with no mine(and is not a guess)

    Initially had no punishment when selecting already revealed cells but the model started only picking cells that were already revealed as majority of other random moves were either mines or guesses. In other words, the model would rather get no reward to avoid picking a mine or guessing

* Network size/dimension changes

* Epsilon decay

* Learning rate

* Test to see if not adding first move of any game to memory as first move will always be a guess/random(Don't want the model to learn from move that was made off of no information)

* Board size and number of mines

<a id="item-four"></a>
# Demo
Incoming...