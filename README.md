# EXP-MADRL

## 0. What is this repo for?
This repository is for experiments to investigate unknown features and potential advantages of multi-agent reinforcement learning with deep neaural network.
Code is basically written in Python using PyTorch .
<p align="center"><img width="250" alt="spread_maddpg_notag" src="https://user-images.githubusercontent.com/60799014/92319743-64f73e00-f056-11ea-9bac-cdeadc4cc2bd.gif"></p>

GIF cited from [here](https://openai.com/blog/learning-to-cooperate-compete-and-communicate/)
___
## 1. Environments
Simple empirical environment has been implemented so far.

### 1.0 Maps
Setting up such appropriate environment is quit important.
Agents interact with the environment and the other agents in the following grid map which has several rooms of 9x9, separated by 2-width hallways and 1-width wall.
<p align="center"><img width="250" alt="map_1x1" src="https://user-images.githubusercontent.com/60799014/95280381-d0b80b00-088f-11eb-9e5e-df9f64b71f1f.png">
<img width="250" alt="map_2x2" src="https://user-images.githubusercontent.com/60799014/95280393-dc0b3680-088f-11eb-9708-7bac19a03c93.png"></p>

<p align="center"><img width="375" alt="map_2x3" src="https://user-images.githubusercontent.com/60799014/95280420-e75e6200-088f-11eb-9a45-ac6ec19623e9.png">
<img width="250" alt="map_3x3" src="https://user-images.githubusercontent.com/60799014/95280409-e299ae00-088f-11eb-8731-b0c43dd7048b.png"></p>

___

### 1.1 Exp1, Exp2
Agents and events are randomly put inside the grid map. Agents aim to approach the events as fast as possible.

<p align="center"><img width="250" alt="exp1_map" src="https://user-images.githubusercontent.com/60799014/95198222-d7a23780-0815-11eb-8493-46a54997af55.png"></p>

| Action | State | Reward |
| ---- | ---- | ---- |
| 0: Stay | obs1: [agent.state.p_pos] | rew1 : -min(dist) / (SIZE_X * num_agents)
| 1: Right | obs2: entity_pos | rew2 : -(1 / num_agents) if is_collision |
| 2: Up | obs3: other_pos | rew1 + rew2 |
| 3: Left | np.concatenate(ob1 + obs2 + obs3) | - |
| 4: Down | - | - |

Still in progress.

## 2. Deep Reinforcement Learning
### 2.1 Deep Q-Network

<p align="center"><img width="500" alt="DQN" src="https://user-images.githubusercontent.com/60799014/95201229-93fdfc80-081a-11eb-9622-847856ba5f02.png"></p>

### 2.2 Deep Deterministic Policy Gradient
<p align="center"><img width="500" alt="DDPG" src="https://user-images.githubusercontent.com/60799014/95201256-a0825500-081a-11eb-867c-9cfbb3562cd2.png"></p>
