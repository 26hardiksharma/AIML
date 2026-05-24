import flappy_bird_gymnasium
import gymnasium as gym
import itertools
import numpy as np
import matplotlib,matplotlib.pyplot as plt
from datetime import datetime,timedelta
import random
import argparse
import yaml
import os
import torch,torch.nn as nn, torch.optim as optim
from DQN import DQN
from experience_replay import ReplayMemory
device = "cuda" if torch.cuda.is_available() else "cpu"
matplotlib.use("Agg")

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR,exist_ok=True)


class Agent:

    def __init__(self,hyperparams_set):
        with open("params.yaml","r") as f:
            params = yaml.safe_load(f)
            hyperparams = params[hyperparams_set]
            
            #Attributes
            self.hyperparameters_set = hyperparams_set
            self.replay_memory_size = hyperparams["replay_memory_size"]
            self.mini_batch_size = hyperparams["mini_batch_size"]
            self.epsilon_init = hyperparams["epsilon_init"]
            self.epsilon_decay = hyperparams["epsilon_decay"]
            self.epsilon_min = hyperparams["epsilon_min"]
            self.network_sync_rate = hyperparams["network_sync_rate"]
            self.gamma = hyperparams["gamma"]
            self.alpha = hyperparams["alpha"]
            self.loss_fn = nn.SmoothL1Loss()
            self.optimizer = None
            self.stop_on_reward = hyperparams["stop_on_reward"]

            #FILES
            self.LOG_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameters_set}.log')
            self.MODEL_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameters_set}.pt')
            self.GRAPH_FILE = os.path.join(RUNS_DIR,f'{self.hyperparameters_set}.png')

    def save_graph(self,rewards,epsilons):
        fig = plt.figure(1)

        mean_rewards = np.zeros(len(rewards))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards[max(0,x-99):(x+1)])

        plt.subplot(121)
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilons)

        plt.subplots_adjust(wspace =1.0,hspace = 1.0)
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    def run(self,is_training = True,render = False):

        env = gym.make("FlappyBird-v0",render_mode = "human" if render else None)
        state_dims = env.observation_space.shape[0]
        action_dims = env.action_space.n

        policy_dqn = DQN(state_dims,action_dims).to(device)
        rewards_list = []
        last_graph_update_time = datetime.now()
        if is_training:
            best_reward = float('-inf')
            # Init variables
            memory = ReplayMemory(self.replay_memory_size)

            epsilon_history = []
            epsilon = self.epsilon_init

            target_dqn = DQN(state_dims,action_dims).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(),lr = self.alpha)
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in itertools.count():

            state,_ = env.reset()
            state = torch.tensor(state,device=device,dtype=torch.float)
            state = state
            done = False
            episode_reward =0
            prev_score = 0

            while not done and episode_reward<self.stop_on_reward:

                # Select Action
                if is_training and random.random()<epsilon:
                    action = env.action_space.sample()

                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
                #Take Action
                new_state,reward,terminated,truncated,info = env.step(action)
                # Small reward for survival
                reward+=0.1

                # Big Reward for passing a pipe
                if info["score"]>prev_score:
                    prev_score = info["score"]
                    reward+=5
                
                #Punishment for failing
                if terminated:
                    reward -=5

                #Processing

                new_state = torch.tensor(new_state,device=device,dtype=torch.float)
                new_state = new_state
                reward= torch.tensor(reward,device=device,dtype=torch.float)

                if is_training:
                    step_count+=1

                if is_training:
                    sample_entry = (state,action,new_state,reward,terminated,truncated)
                    memory.append(sample_entry)

                # Move to new state
                state = new_state
                episode_reward+=reward.item()
                done = truncated or terminated

                if is_training and len(memory)>=1000:
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch,policy_dqn,target_dqn)

                    if step_count >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0
            if is_training:
                print(f"Episode: #{episode+1} || Reward: {episode_reward} || Epsilon: {epsilon} || Score: {info['score']}")
            else:
                print(f"Episode #{episode+1} || Score: {info['score']} || Reward: {episode_reward}")
            # if(episode%500 == 0):
            #     torch.save(policy_dqn.state_dict(),"flappybird.pt")
            rewards_list.append(episode_reward)
            if is_training:
                epsilon = max(self.epsilon_min,self.epsilon_decay*epsilon)
                epsilon_history.append(epsilon)
            
            if is_training and episode%1000 ==0:
                torch.save(policy_dqn.state_dict(),self.MODEL_FILE)

            if is_training:
                if episode_reward>best_reward:
                    log = f"Time: {datetime.now()} || Episode: #{episode+1} || New Best: {episode_reward} || Prev Best: {best_reward}"
                    best_reward= episode_reward

                    with open(self.LOG_FILE,"a") as f:
                        f.write(log+"\n")
                    
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)

                current_time = datetime.now()

                if current_time - last_graph_update_time> timedelta(minutes=1):
                    self.save_graph(rewards_list,epsilon_history)
                    last_graph_update_time = current_time

            
        env.close()
    def optimize(self,mini_batch,policy_dqn,target_dqn):
        states,actions,new_states,rewards,terminations,_ = zip(*mini_batch)
        actions = torch.tensor(actions,dtype=torch.long,device=device)
        states = torch.stack(states)
        new_states = torch.stack(new_states)
        terminations = torch.tensor(terminations,dtype=torch.float,device=device)
        rewards = torch.stack(rewards)

        with torch.no_grad():
            next_actions = policy_dqn(new_states).argmax(dim=1)

            next_q_values = target_dqn(new_states).gather(1,next_actions.unsqueeze(1)).squeeze()

        target_q = rewards + ((1 - terminations)* self.gamma* next_q_values)

        current_q = policy_dqn(states).gather(1,index=actions.unsqueeze(1)).squeeze()

        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(),max_norm=1.0)

        self.optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/Test Model"
    )

    parser.add_argument(
        "hyperparameters",
        help="Hyperparameter set name"
    )

    parser.add_argument(
        "--train",
        help="Training Mode",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(hyperparams_set=args.hyperparameters)

    dql.run(is_training=True) if args.train else dql.run(is_training=False,render=True)
    # env = gym.make("FlappyBird-v0")
    # state,_ = env.reset()
    # action = env.action_space.sample()
    # new_state,_,_,_,_ = env.step(action)
    # print(new_state)
