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


# ---------------------------
# Dodatkowe funkcje reward shaping
# ---------------------------

# Definicja wartości poszczególnych figur
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # Króla nie liczymy, bo jest bezcenny
}


def evaluate_material(board):
    """Oblicza przewagę materialną: dodatnia wartość oznacza przewagę białych, ujemna – czarnych."""
    white_material = 0
    black_material = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                white_material += value
            else:
                black_material += value
    return white_material - black_material


def evaluate_center_control(board, color):
    # Definiujemy centralne pola
    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    control = 0
    for square in center_squares:
        attackers = board.attackers(color, square)
        control += len(attackers)
    return control


def evaluate_endgame_enemy_king(board, color, piece_threshold=10, bonus=0.2):
    if len(board.piece_map()) > piece_threshold:
        return 0

    enemy_color = not color
    enemy_king_square = board.king(enemy_color)

    corners = [chess.A1, chess.H1, chess.A8, chess.H8]

    if enemy_king_square in corners:
        return bonus

    rank = chess.square_rank(enemy_king_square)
    file = chess.square_file(enemy_king_square)
    distances = [abs(rank - r) + abs(file - f) for r, f in [(0, 0), (0, 7), (7, 0), (7, 7)]]
    min_distance = min(distances)

    if min_distance <= 2:
        return bonus * (3 - min_distance)
    return 0


def evaluate_development(board, color):
    developed = 0
    # Dla białych początkowy rząd to 0 (a1, b1, …, h1); dla czarnych to 7 (a8, …, h8)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type != chess.PAWN:
            rank = chess.square_rank(square)
            if color == chess.WHITE and rank > 0:
                developed += 1
            elif color == chess.BLACK and rank < 7:
                developed += 1
    return developed


def evaluate_mobility(board, color):
    # Jeżeli to nie jest tura danego koloru, można oszacować mobilność po stronie przeciwnika
    if board.turn != color:
        return 0
    return len(list(board.legal_moves))


def evaluate_king_safety(board, color):
    king_square = board.king(color)
    if king_square is None:
        return 0
    # Przyjmujemy, że król jest bezpieczniejszy, gdy jest już zroszowany
    if color == chess.WHITE and king_square in [chess.G1, chess.C1]:
        return 1
    elif color == chess.BLACK and king_square in [chess.G8, chess.C8]:
        return 1
    else:
        return 0


def evaluate_piece_activity(board, color):
    activity = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            attacks = board.attacks(square)
            activity += len(attacks)
    return activity


def evaluate_pawn_structure(board, color):
    # Prosty model: kara za zdublowane piony
    files = {i: 0 for i in range(8)}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color and piece.piece_type == chess.PAWN:
            file = chess.square_file(square)
            files[file] += 1
    doubled = sum([count - 1 for count in files.values() if count > 1])
    return -doubled  # ujemna wartość, gdyż zdublowane piony są niekorzystne


def combined_reward(board, final_reward, alpha=0.9, color=chess.WHITE):
    """
    Łączy nagrodę końcową, nagrodę materialną i dodatkowe sygnały.
    Parametr alpha określa wagę wyniku końcowego i materialnego.
    """
    # Ocena materialna (funkcja już zaimplementowana)
    material_advantage = evaluate_material(board)
    max_material = 39
    material_reward = material_advantage / max_material

    # Dodatkowe sygnały:
    center_control = evaluate_center_control(board, color)
    development = evaluate_development(board, color)
    mobility = evaluate_mobility(board, color)
    king_safety = evaluate_king_safety(board, color)
    piece_activity = evaluate_piece_activity(board, color)
    pawn_structure = evaluate_pawn_structure(board, color)
    enemy_king_bonus = evaluate_endgame_enemy_king(board, color, piece_threshold=10, bonus=0.2)

    # Wagi dla poszczególnych komponentów – eksperymentuj, aby dobrać optymalne wartości
    w_center = 0.1
    w_development = 0.1
    w_mobility = 0.1
    w_king = 0.1
    w_activity = 0.1
    w_pawn = 0.1
    w_enemy_king = 1.0

    additional_reward = (
            w_center * center_control +
            w_development * development +
            w_mobility * mobility +
            w_king * king_safety +
            w_activity * piece_activity +
            w_pawn * pawn_structure +
            w_enemy_king * enemy_king_bonus
    )
    win_bonus = 10 if final_reward == 1.0 else 0
    base_reward = alpha * final_reward + (1 - alpha) * material_reward
    return base_reward + additional_reward + win_bonus
