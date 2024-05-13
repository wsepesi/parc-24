import torch
import asyncio
from lib import Config
import wandb
from dataclasses import asdict

import math
import torch.optim as optim

from dqn import FF_DQN, FF_DQN_2, DQNAgent
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder

from replay_buffer import ReplayMemory
from rl_players import SimpleRLPlayer
from teams import teams

async def main():
    print("Creating environment...")
    TEAM = "six"
    team = ConstantTeambuilder(teams[TEAM])
    player = SimpleRLPlayer
    opponent = RandomPlayer(battle_format="gen7ou",team=team)
    env = player(battle_format="gen7ou",team=team, opponent=opponent, start_challenging=True)
    
    NUM_EPS_BASE = 2000000 # number of turns (NOT GAMES)
    config = Config(
        BATCH_SIZE=32,
        GAMMA=0.99,
        EPS_START=0.95,
        EPS_END=0.05,
        EPS_DECAY=NUM_EPS_BASE / 10,
        TAU=0.005,
        LR=1e-4,
        NUM_EPISODES=NUM_EPS_BASE,
        REPLAY_BUFFER_SIZE=NUM_EPS_BASE / 10, 
        BURN_IN=NUM_EPS_BASE / 10, # fill replay buffer with burn in data, and then start decay process
        EVAL_EVERY=math.floor(NUM_EPS_BASE / (10 if NUM_EPS_BASE < 1000000 else 25)),
    )

    wandb.init(
        project="parc-24",
        config=asdict(config),
    )

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)
    env.close()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    policy_net = FF_DQN_2(n_observations, n_actions, device).to(device)
    target_net = FF_DQN_2(n_observations, n_actions, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=config.LR, amsgrad=True)
    memory = ReplayMemory(config.REPLAY_BUFFER_SIZE)

    agent = DQNAgent(
        config, 
        player,
        policy_net, 
        target_net, 
        optimizer, 
        memory,
        model_path="six",
        eval_opponents=[
            RandomPlayer(battle_format="gen7ou",team=team),
            MaxBasePowerPlayer(battle_format="gen7ou",team=team),
            SimpleHeuristicsPlayer(battle_format="gen7ou",team=team)
        ],
        team=team,
    )

    print("Training agent...")
    agent.train()
    print("Complete")
    wandb.finish()

if __name__ == "__main__":
    asyncio.run(main())