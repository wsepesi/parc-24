from dataclasses import dataclass

@dataclass
class Config:
    BATCH_SIZE: int = 128
    GAMMA: float = 0.99
    EPS_START: float = 0.9
    EPS_END: float = 0.05
    EPS_DECAY: int = 1000
    TAU: float = 0.005
    LR: float = 1e-4
    NUM_EPISODES: int = 200
    REPLAY_BUFFER_SIZE: int = 10000
    BURN_IN: int = 1000
    EVAL_EVERY: int = 1000