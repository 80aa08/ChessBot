import chess
import torch
import numpy as np
from config import Config

class ChessEnv:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return self._get_state()

    def step(self, move):
        self.board.push(move)
        reward = 0
        done = self.board.is_game_over()
        if done:
            result = self.board.result()
            if result == '1-0': reward = 1
            elif result == '0-1': reward = -1
            else: reward = 0
        return self._get_state(), reward, done

    def _get_state(self):
        # one-hot encoding of pieces
        planes = np.zeros((Config.INPUT_CHANNELS, 8, 8), dtype=np.int8)
        piece_map = self.board.piece_map()
        for sq, piece in piece_map.items():
            idx = (piece.piece_type - 1) + (0 if piece.color else 6)
            row, col = divmod(sq, 8)
            planes[idx, row, col] = 1
        # extras: side to move
        planes[12, :, :] = self.board.turn
        # TODO: add castling rights, repetition, move counts
        return torch.tensor(planes, dtype=torch.float32)

    def legal_moves(self):
        return list(self.board.legal_moves)
