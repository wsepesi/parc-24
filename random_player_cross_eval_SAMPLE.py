from poke_env import RandomPlayer, cross_evaluate
from tabulate import tabulate
import asyncio

async def main():
    # Create three random players
    players = [RandomPlayer(max_concurrent_battles=10, battle_format="gen7randombattle") for _ in range(3)]

    # Cross evaluate players: each player plays 20 games against every other player
    cross_evaluation = await cross_evaluate(players, n_challenges=20)

    # Prepare results for display
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])

    # Display results
    print(tabulate(table))

if __name__ == "__main__":
    asyncio.run(main())