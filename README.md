<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h2 align="center">IE579 - Game Theory and Multi-Agent Reinforcement Learning </h2>
  <p align="center">
    Final Project Repository for IE579
  </p>
</div>

## 1. About The Project
### 1.1. Environment Description
We will use **MAgent-Battle** for final project. 
It is an environment for multi-agent reinforcement learning (MARL).
Each agent (small blue or red box) can move or attack enemy and The objective of each team (blue or red) is to kill all opponents in the game

We will use simplified environment. There are 5 blue and 5 red in 15X15 size map.
Description for state, observation for each agent, action and reward is as follows:

**_State_** space is a 15X15 map with channels in the table. State space will (15, 15, 37). 
Note that your actor network for submission should not use state as an input. 
Using state for training is totally fine. 
For example, you can use state as an input to train critic network.

**_Observation_** space is a 5x5 map with the channels in the table. Observation space will (5, 5, 41).
Your actor network for final submission should use observation as an input.

**_Action_** space is discrete and 21 dimensions. 13 dimensions for moving and 8 dimensions 
for attacking as shown in the below figure.

**_Reward_** is summation of multiple reward components. Note that you are free to change the reward design by yourself.
- +5 for killing an opponent.
- -0.005 for every timestep.
- -0.1 for attacking.
- +0.2 for attacking an opponent (when attack is success).
- -0.1 for dying.

### 1.2. Rules and Evaluation
To compare two models, we test 200 times with 100 different random seed and switching.
For example, if we test model A and B, test will be as follows:
- (blue=A, red=B, seed=1), (blue=B, red=A, seed=1), (blue=A, red=B, seed=2), (blue=B, red=A, seed=2), …, (blue=B, red=A, seed=100)

Each team's submitted model is tested against all other models.

Note that your model should use only the observation of your team, not the opponent's as an input.
The observation configuration should be kept. Also, only decentralized actor is allowed. 
You don't need to submit the RL-based model; a rule-based model is also acceptable.

### 1.3. Training Strategy Example
Because the game is competitive, we need a (good) opponent behavior model for training. 
However, static or unskilled opponenet might lead to overfitting.
To mitigate the issue, there are multiple approaches:
- **Self-play**: Agents can enhance their performance by playing against themselves.
- **Population-play**: Train a population of agents, with each agent playing against the others.


## 2. Getting Started
### 2.1. WSL installation (Optional, for Windows user)
WSL (Windows Subsystem for Linux) is designed to make it easy to use Linux tools within Windows, 
eliminating the need to install a virtual machine. 
Since MAgent exclusively supports Linux, if you don't have access to a Linux machine, we recommend using WSL to run MAgent.
1. Open PowerShell as an administrator
2. Install WSL
```bash
wsl --install
```
3. Type username and password

If it doesn't work, type `wsl --install -d Ubuntu-18.04`. 
You can select the version of Ubuntu by using `wsl --list --online`.
Please refer this [link](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#1-overview)
 to get more information.

### 2.2. Anaconda installation
1. Open the Ubuntu terminal or WSL
2. Install Anaconda. (You can install different version of anaconda [here](https://repo.anaconda.com/archive/))
```bash
wget -c https://repo.continuum.io/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
bash Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Accept the license term and activate the anaconda
```bash
source ~/anaconda3/bin/activate
```
4. Create env and activate created environment.
```bash
conda create --name game_theory python=3.7
conda activate game_theory
```


### 2.3. MAgent installation
Next, let's install MAgent. Begin by cloning the MAgent repository and then follow the provided instructions.
It's recommended to execute the commands step by step.
1. Clone the repo
```bash
git clone https://github.com/geek-ai/MAgent.git
```
2. Install MAgent
```bash
cd MAgent
sudo apt-get update
sudo apt-get install cmake libboost-system-dev libjsoncpp-dev libwebsocketpp-dev
sudo apt-get update && sudo apt-get install build-essential
bash build.sh
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
pip install protobuf==3.20.1
```
3. (Optional) Verify the installation of MAgent
```bash
pip install tensorflow==1.13.1
python examples/train_battle.py --train
```

## 3. Project Tips
- If you want to develop in Windows using Linux based server or WSL, use SSH or WSL interpreter.
  - If you use PyCharm ([WSL](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html),
[SSH](https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html))
  - If you use VSCode ([WSL](https://code.visualstudio.com/docs/remote/wsl), 
[SSH](https://code.visualstudio.com/docs/remote/ssh))
- Use [WandB](https://docs.wandb.ai/quickstart) as a performance visualization tool.
- Deep RL (PPO) implementation tips. ([link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/))
- MAPPO official implementation. ([link](https://github.com/zoeyuchao/mappo))

## Contact

(TA) Kanghoon Lee - leehoon@kaist.ac.kr

## References
