# parc-24

## installation

- install ps as `pokemon-showdown` at the parent directory

## training

- `node pokemon-showdown start --no-security`
- `python train.py`

## notes

- simple dqn set up, no layernorm no dropout, FF 128 neurons
- simple fwg randbats env to test convergence of learning
- simple state embedding (see `rl_players.py`)

## agenda

- work on server (TODO:)
- set up harder envs (6v6, randbats) (FIXME: set up as flag or something)
- training scheduling
- upgrade NN
- upgrade state representation
- towards rainbow
- distributed training
- self play
- actually turn on incomplete info

## long term todo

- formal method for multi agent
- formal method for incomplete info
- CFR
- better exploration policy
- MCTS or other search (patricks method)
- recurrent network style for histories > state?
