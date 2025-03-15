import chess
import torch


def new_game():
    return chess.Board()


def board_to_tensor(board):
    board_tensor = torch.zeros((13, 8, 8), dtype=torch.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            type_idx = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                type_idx += 6
            row, col = 7 - square // 8, square % 8
            board_tensor[type_idx, row, col] = 1.0
    board_tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    return board_tensor


def is_game_over(board):
    return board.is_game_over()


def result(board):
    if not board.is_game_over():
        return 0.5  # Domyślna wartość dla nierozstrzygniętej gry
    res = board.result()
    return 1.0 if res == "1-0" else (0.0 if res == "0-1" else 0.5)


def uci_index_map():
    board = chess.Board()
    return {move.uci(): i for i, move in enumerate(board.legal_moves)}