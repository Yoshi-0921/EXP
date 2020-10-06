# EXP-MADRL

## 0. What is this repo for?
This repository is for experiments to investigate unknown features and potential advantages of multi-agent reinforcement learning with deep neaural network.
Code is basically written in Python using PyTorch .
<p align="center"><img width="250" alt="spread_maddpg_notag" src="https://user-images.githubusercontent.com/60799014/92319743-64f73e00-f056-11ea-9bac-cdeadc4cc2bd.gif"></p>

GIF cited from [here](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
___
## 1. Experiments
Simple emprical environment has been implemented so far.
### 1.1 exp1
<p align="center"><img width="200" alt="exp1_map" src="https://user-images.githubusercontent.com/60799014/95198222-d7a23780-0815-11eb-8493-46a54997af55.png"></p>

|  Action  |  State  |  Reward  |
| ---- | ---- | ---- |
|  TD  |  TD  |  TD  |
|  TD  |  TD  |  TD  |
|  TD  |  TD  |  TD  |

___

### 1.2 exp2
The same map above is used in exp2.
___

### 1.3 exp3

Setting up such appropriate environment is quit important.
Agents interact with the environment and the other agents in the following 38x38 grid map which has 9 rooms of 10x10, separated by 2-width hallways and 1-with wall.
<p align="center"><img width="200" alt="exp3_map" src="https://user-images.githubusercontent.com/60799014/92319837-5d846480-f057-11ea-9430-2a6174093d82.jpg"></p>

## 2. Agents
___