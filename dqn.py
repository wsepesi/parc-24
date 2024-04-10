import torch
import os
from poke_env.player.env_player import Gen7EnvSinglePlayer
import wandb

from itertools import count
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import random

from replay_buffer import Transition
from lib import Config

        
class FF_DQN(nn.Module):
    def __init__(self, n_observations, n_actions, device):
        super(FF_DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128, device=device)
        self.layer2 = nn.Linear(128, 128, device=device)
        self.layer3 = nn.Linear(128, n_actions, device=device)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQNAgent:
    def __init__(
            self,
            config: Config,
            player: Gen7EnvSinglePlayer,
            policy_net,
            target_net,
            optimizer,
            memory,
            model_path,
            eval_opponents,
            team,
            LOG_EVERY = 10,
            # EVAL_EVERY = 100,
            EVAL_NUM_GAMES = 25,
            FINAL_EVAL_GAMES = 100,
            EVAL_EPSILON = 0.05 # why not 0?
            ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.EPS_START = config.EPS_START
        self.EPS_END = config.EPS_END
        self.EPS_DECAY = config.EPS_DECAY
        self.BATCH_SIZE = config.BATCH_SIZE
        self.GAMMA = config.GAMMA
        self.TAU = config.TAU
        self.BURN_IN = config.BURN_IN

        self.memory = memory
        self.holdout_set = []

        self.optimizer = optimizer
        self.policy_net = policy_net
        self.target_net = target_net
        self.player = player

        self.steps_done = 0
        self.num_episodes = config.NUM_EPISODES

        self.model_path = model_path

        self.eval_opponents = eval_opponents
        self.team = team
        self.LOG_EVERY = LOG_EVERY # wandb.log cadence for logging loss, on number of env steps
        self.EVAL_EVERY = config.EVAL_EVERY # num episodes to train before eval
        self.EVAL_NUM_GAMES = EVAL_NUM_GAMES # num games to play against each opponent
        self.EVAL_EPSILON = EVAL_EPSILON # epsilon for evaluation action selection
        self.FINAL_EVAL_GAMES = FINAL_EVAL_GAMES # num games to play against each opponent in final evaluation


    def select_action(self, state, env, is_eval=False):
        if is_eval:
            with torch.no_grad():
                sample = random.random()
                if sample > self.EVAL_EPSILON:
                    return self.network(state).max(1).indices.view(1, 1)
                else:
                    return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            sample = random.random()
            eps_threshold = self.EPS_START + (self.EPS_END - self.EPS_START) * \
                (self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)


    def optimize_model(self, it):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.HuberLoss() # smooth l1? mse? 
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        if it % self.LOG_EVERY == 0:
            wandb.log({"loss": loss.item()})

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        self.create_holdout_set()
        env = self._get_env() 
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        READY_FOR_EVAL = False
        for episode in tqdm(range(self.num_episodes)):
                if episode % self.EVAL_EVERY == 0 and episode >= self.BURN_IN:
                    READY_FOR_EVAL = True

                if episode > self.BURN_IN:
                    action = self.select_action(state, env)
                else:
                    action = torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state
                
                if episode > self.BURN_IN: # only do network updates after burn-in period
                    self.optimize_model(it=episode)

                    # θ′ ← τ θ + (1 −τ )θ′ -- two network update #TODO: re implement this to fit RAINBOW paper
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    if READY_FOR_EVAL:
                        env.close()
                        self.eval()
                        env = self._get_env()
                        READY_FOR_EVAL = False
                    state, _ = env.reset()
                    state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        print('Done training...')

        # save model
        write_path = "results" #/{}/{}".format("stronger_starters", "test")
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        
        saved_model_path = os.path.join(write_path, f"{self.model_path}.torch")
        torch.save(self.policy_net.state_dict(), saved_model_path)
        
        self.memory.clean_up()

        print('Doing final evaluation...')
        self.eval(is_final=True)


    def eval(self, is_final=False):
        """
        Evaluate the agent against a set of opponents.

        Collect and log relevant metrics, being winrate, average total reward, 
        """
        results = {}

        num_games_to_eval = self.EVAL_NUM_GAMES if not is_final else self.FINAL_EVAL_GAMES
        for opponent in self.eval_opponents:
            name = opponent.__class__.__name__
            env = self._get_env(opponent)
            games_won = self.play_games(env, num_games_to_eval, name)
            env.close()
            results[name] = games_won

        # log to wandb
        results = {f"{opponent}_{metric}": value for opponent, metrics in results.items() for metric, value in metrics.items()}
        results["avg_max_q_holdout"] = self.calculate_holdout_avg_max_qs()
        wandb.log(results)
        return results
    
    def play_games(self, env, num_games, name="default"):
        metrics = {
            "winrate": 0,
            "avg_total_reward": 0,
            "avg_episode_length": 0,
            "avg_max_q_dynamic": 0,
        } # TODO: more metrics?

        episode_rewards = []
        episode_durations = []
        max_qs = []

        for _ in range(num_games):
            total_reward = 0
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = self.select_action(state, env)
                max_qs.append(self.get_max_q(state))
                observation, reward, terminated, truncated, _ = env.step(action.item())
                total_reward += reward
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
               
                state = next_state

                if done:
                    break

            # collect metrics
            episode_rewards.append(total_reward)
            episode_durations.append(t + 1)

            # determine winner and score
        
        print(f"{100 * (env.n_won_battles / num_games)}% winrate against {name}")
        
        metrics["winrate"] = env.n_won_battles / num_games
        metrics["avg_total_reward"] = sum(episode_rewards) / num_games
        metrics["avg_episode_length"] = sum(episode_durations) / num_games
        metrics["avg_max_q_dynamic"] = sum(max_qs) / len(max_qs)

        return metrics
    
    def get_max_q(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1).values.item()
    
    def calculate_holdout_avg_max_qs(self):
        max_qs = []
        for _, transition in enumerate(self.holdout_set):
            state = transition.state
            max_qs.append(self.get_max_q(state))

        avg_max_qs = sum(max_qs) / len(max_qs)
        print(f"Average max Q value on holdout set: {avg_max_qs}")
        return avg_max_qs
    
    def _get_env(self, opponent=None): # TODO: modulate opponent throughout training
        if opponent is None:
            opponent = self.eval_opponents[0]
        return self.player(battle_format="gen7ou",team=self.team, opponent=opponent, start_challenging=True)
    
    def create_holdout_set(self):
        # play 100 games against random opp, playing a random policy, and store in holdout set
        print("Creating holdout set...")
        env = self._get_env()
        for _ in tqdm(range(100)):
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            for t in count():
                action = torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                self.holdout_set.append(Transition(state, action, next_state, reward))
                state = next_state

                if done:
                    break
        env.close()
        return self.holdout_set
