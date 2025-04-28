import datetime
import os
import chess
import numpy as np
import torch
import chess.pgn as pgn

# Save game to PGN
def save_pgn(env, path="./games", prefix="game"):
    os.makedirs(path, exist_ok=True)
    game = pgn.Game.from_board(env.board)
    game.headers["Event"] = "Self Training"
    game.headers["Time"] = datetime.datetime.now().strftime("%Y%m%d")
    fname = f"{prefix}_{len(os.listdir(path))}.pgn"
    with open(os.path.join(path, fname), 'w') as f:
        f.write(str(game))



def get_symmetries(state: np.ndarray, pi: np.ndarray):
    # pi expected length 4672
    if pi.size != 4672:
        return []
    syms = []
    pi_mat = pi.reshape(73, 64)
    for k in [0,1,2,3]:
        # rotate state and policy
        s_rot = np.rot90(state, k, axes=(1,2))
        pi_rot = np.rot90(pi_mat, k)
        syms.append((s_rot, pi_rot.reshape(-1)))
        # horizontal flip
        s_flip = np.flip(s_rot, axis=2)
        pi_flip = np.flip(pi_rot, axis=1)
        syms.append((s_flip, pi_flip.reshape(-1)))
    return syms

# Augment examples
def augment_examples(examples):
    augmented = []
    for state, pi, z in examples:
        # move tensors to CPU
        if isinstance(state, torch.Tensor):
            state_cpu = state.cpu().numpy()
        else:
            state_cpu = state
        if isinstance(pi, torch.Tensor):
            pi_cpu = pi.cpu().numpy()
        else:
            pi_cpu = np.array(pi)
        # generate symmetries
        syms = get_symmetries(state_cpu, pi_cpu)
        if not syms:
            # keep original if no symmetries
            augmented.append((torch.tensor(state_cpu), torch.tensor(pi_cpu), z))
        else:
            for s2, pi2 in syms:
                # ensure contiguous arrays
                s2_c = np.ascontiguousarray(s2)
                pi2_c = np.ascontiguousarray(pi2)
                augmented.append((torch.from_numpy(s2_c), torch.from_numpy(pi2_c), z))
    return augmented