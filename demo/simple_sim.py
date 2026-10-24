"""
Minimal, readable CtF simulation.

Plays a few matches between two heuristic policies, prints the board as
ASCII at the start and end of the first match, and saves an animated GIF
of that match to demo/simple_sim.gif.

    python demo/simple_sim.py --blue Roomba --red Defense --matches 5
"""

import argparse

import numpy as np
from PIL import Image

import gym
import gym_cap
import gym_cap.heuristic as policy
from gym_cap.envs.const import *

parser = argparse.ArgumentParser(description='Run a simple CtF simulation')
parser.add_argument('--blue', default='Roomba', help='blue heuristic policy class')
parser.add_argument('--red', default='Defense', help='red heuristic policy class')
parser.add_argument('--matches', type=int, default=5)
parser.add_argument('--map-size', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gif', default='demo/simple_sim.gif')
args = parser.parse_args()

GLYPH = {
    TEAM1_BACKGROUND: '.', TEAM2_BACKGROUND: ',', OBSTACLE: '#',
    TEAM1_UGV: 'b', TEAM2_UGV: 'r', TEAM1_UAV: 'B', TEAM2_UAV: 'R',
    TEAM1_FLAG: 'F', TEAM2_FLAG: 'f',
}


def print_board(env):
    board = env.get_full_state
    for row in board:
        print(' '.join(GLYPH.get(int(cell), '?') for cell in row))
    print("legend: . blue land  , red land  # wall  b/r agents  F blue flag  f red flag")


def frame(env, scale=16):
    rgb = env.get_full_state_rgb.astype(np.uint8)
    return Image.fromarray(np.kron(rgb, np.ones((scale, scale, 1), dtype=np.uint8)))


env = gym.make("cap-v0")
env.seed(args.seed)
np.random.seed(args.seed)  # flag respawn uses the global numpy RNG
env.reset(
    map_size=args.map_size,
    config_path='demo/base_settings.ini',
    policy_blue=getattr(policy, args.blue)(),
    policy_red=getattr(policy, args.red)(),
)

wins = {'blue': 0, 'red': 0, 'draw': 0}
for match in range(args.matches):
    if match > 0:
        env.reset()
    if match == 0:
        print("=== Match 1: starting board ===")
        print_board(env)
        frames = [frame(env)]

    done = False
    while not done:
        # No action passed: both teams act from their assigned policies
        observation, reward, done, info = env.step()
        if match == 0:
            frames.append(frame(env))

    if match == 0:
        print("\n=== Match 1: final board ===")
        print_board(env)
        frames[0].save(args.gif, save_all=True, append_images=frames[1:],
                       duration=100, loop=0)
        print("Saved replay to", args.gif)

    result = 'blue' if env.blue_win else 'red' if env.red_win else 'draw'
    wins[result] += 1
    print("Match %d: %3d steps | captures blue %d - red %d | blue alive %d/%d, red alive %d/%d -> %s"
          % (match + 1, env.run_step, env.blue_points, env.red_points,
             sum(a.isAlive for a in env.get_team_blue), len(env.get_team_blue),
             sum(a.isAlive for a in env.get_team_red), len(env.get_team_red),
             result.upper()))

print("\n%s (blue) vs %s (red) over %d matches: %s"
      % (args.blue, args.red, args.matches, wins))
