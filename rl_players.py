
import numpy as np
import sys
import os

from gymnasium.spaces import Box, Space

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Gen7EnvSinglePlayer, Gen4EnvSinglePlayer
from poke_env.data import GenData


base_path = os.path.dirname(os.path.realpath(__file__))
path_to_add = os.path.join(base_path, "poke-env/src/")
sys.path.append(path_to_add)

class SimpleRLPlayer(Gen7EnvSinglePlayer):
    def describe_embedding(self) -> Space:
        return Box(
            low=np.array(([-1] * 4) + ([0] * 4) + [0, 0], dtype=np.float32),
            high=np.array(([3] * 4) + ([4] * 4) + [1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def embed_battle(self, battle: AbstractBattle): 
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart = GenData.from_gen(7).type_chart
                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=5, hp_value=0.5, victory_value=30
        )


class MediumRLPlayer(Gen7EnvSinglePlayer):
    pass
class BigBoyRLPlayer(Gen7EnvSinglePlayer):
    pass
class JettRLPlayer(Gen4EnvSinglePlayer):
    pass
class JettAppxRLPlayer(Gen7EnvSinglePlayer):
    pass