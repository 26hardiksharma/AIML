import flappy_bird_gymnasium
import gymnasium as gym
import yaml
import random
import argparse
from DQN import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:

    def __init__(self, param_set):

        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.replay_memory_size = params["replay_memory_size"]
        self.env_id = params["env_id"]
        self.network_sync_rate = params["network_sync_rate"]
        self.mini_batch_size = params["mini_batch_size"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.reward_threshold = params["reward_threshold"]

        # Huber Loss is more stable than MSE for RL
        self.loss_fn = nn.SmoothL1Loss()

        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def run(self, is_training=True, render=False):

        env = gym.make(
            self.env_id,
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions).to(device)
        start_episode = 0
        if is_training:

            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states,num_actions).to(device)

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            steps = 0
            best_reward = float("-inf")
            start_episode = 0
            epsilon = self.epsilon_init

            # Resume training if checkpoint exists
            if os.path.exists(self.MODEL_FILE):

                print("Loading existing model...")

                checkpoint = torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )

                policy_dqn.load_state_dict(checkpoint["model_state_dict"])

                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                epsilon = checkpoint["epsilon"]

                start_episode = checkpoint["episode"] + 1

                best_reward = checkpoint.get("best_reward",float("-inf"))

            target_dqn.load_state_dict(policy_dqn.state_dict())

        else:

            checkpoint = torch.load(
                self.MODEL_FILE,
                map_location=device
            )

            policy_dqn.load_state_dict(
                checkpoint["model_state_dict"]
            )

            policy_dqn.eval()

        # Moving average tracking
        reward_history = deque(maxlen=100)

        for episode in itertools.count(start=start_episode):

            terminated = False
            episode_reward = 0

            state, _ = env.reset()

            # Normalize state
            state = torch.tensor(
                state,
                dtype=torch.float,
                device=device
            ) / 100

            while (
                not terminated and
                episode_reward < self.reward_threshold
            ):

                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:

                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.long,
                        device=device
                    )

                else:

                    with torch.no_grad():

                        action = policy_dqn(
                            state.unsqueeze(dim=0)
                        ).squeeze().argmax()

                # Step environment
                new_state, reward, terminated, _, _ = env.step(
                    action.item()
                )

                # Reward shaping
                if terminated:
                    reward = -5
                else:
                    reward += 0.1

                reward = torch.tensor(
                    reward,
                    dtype=torch.float,
                    device=device
                )

                # Normalize next state
                new_state = torch.tensor(
                    new_state,
                    dtype=torch.float,
                    device=device
                ) / 100

                episode_reward += reward.item()

                # Store experience
                if is_training:

                    memory.append(
                        (
                            state,
                            action,
                            new_state,
                            reward,
                            terminated
                        )
                    )

                    steps += 1

                state = new_state

            reward_history.append(episode_reward)

            avg_reward = (
                sum(reward_history) /
                len(reward_history)
            )

            print(
                f"Episode: #{episode+1} || "
                f"Reward: {episode_reward:.2f} || "
                f"Avg100: {avg_reward:.2f} || "
                f"Epsilon: {epsilon:.4f}"
            )

            # Training step
            if is_training:

                epsilon = max(
                    epsilon * self.epsilon_decay,
                    self.epsilon_min
                )

                # Save best model
                if episode_reward > best_reward:

                    log_msg = (
                        f"Best Reward Achieved || "
                        f"Episode: {episode} || "
                        f"Reward: {episode_reward} || "
                        f"Prev Best: {best_reward}"
                        f"Total pipes ~ {10+episode_reward}"
                    )

                    print(log_msg)

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(
                        {
                            "model_state_dict":
                                policy_dqn.state_dict(),

                            "optimizer_state_dict":
                                self.optimizer.state_dict(),

                            "epsilon":
                                epsilon,

                            "episode":
                                episode,

                            "best_reward":
                                episode_reward
                        },
                        self.MODEL_FILE
                    )

                    best_reward = episode_reward

                # Learn from replay buffer
                if len(memory) > 5000 and steps%4 == 0:

                    mini_batch = memory.sample(
                        self.mini_batch_size
                    )

                    self.optimize(
                        mini_batch,
                        policy_dqn,
                        target_dqn
                    )

                    # Sync target network
                    if steps > self.network_sync_rate:

                        target_dqn.load_state_dict(
                            policy_dqn.state_dict()
                        )

                        steps = 0

        env.close()

    def optimize(self,mini_batch,policy_dqn,target_dqn):
        (states,actions,next_states,rewards,terminations) = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions).long()

        next_states = torch.stack(next_states)

        rewards = torch.stack(rewards)

        terminations = torch.tensor(
            terminations,
            dtype=torch.float,
            device=device
        )

        # Bellman target
        with torch.no_grad():
            next_actions = policy_dqn(next_states).argmax(dim=1)

            # Action evaluation from target network
            next_q_values = target_dqn(next_states).gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze()

            # Double DQN target
            target_q = rewards + (
                (1 - terminations)
                * self.gamma
                * next_q_values
            )
        # Current Q-values
        current_q = policy_dqn(states).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze()

        # Loss
        loss = self.loss_fn(
            current_q,
            target_q
        )

        # Backprop
        self.optimizer.zero_grad()

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            policy_dqn.parameters(),
            1.0
        )

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

    dql = Agent(
        param_set=args.hyperparameters
    )

    if args.train:
        dql.run(is_training=True)

    else:
        dql.run(
            is_training=False,
            render=True
        )