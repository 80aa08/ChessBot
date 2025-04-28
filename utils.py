import os
import chess
import numpy as np
import torch

# Save game to PGN
def save_pgn(env, path="./games", prefix="game"):
    os.makedirs(path, exist_ok=True)
    game = chess.pgn.Game.from_board(env.board)
    fname = f"{prefix}_{len(os.listdir(path))}.pgn"
    with open(os.path.join(path, fname), 'w') as f:
        f.write(str(game))

# Generate symmetric transformations
def get_symmetries(state: np.ndarray, pi: np.ndarray):
    syms = []
    for k in [0,1,2,3]:  # rotations
        s_rot = np.rot90(state, k, axes=(1,2))
        pi_rot = np.rot90(pi.reshape(73,64), k).reshape(-1)
        syms.append((s_rot, pi_rot))
        # horizontal flip
        s_flip = np.flip(s_rot, axis=2)
        pi_flip = np.flip(pi_rot.reshape(73,64), axis=2).reshape(-1)
        syms.append((s_flip, pi_flip))
    return syms

# Augment examples
def augment_examples(examples):
    augmented = []
    for state, pi, z in examples:
        s_np = state.numpy()
        for s2, pi2 in get_symmetries(s_np, pi.numpy()):
            augmented.append((torch.tensor(s2), torch.tensor(pi2), z))
    return augmented
