# train.py - Główny plik treningowy
import torch
import torch.optim as optim
from model import ChessNet
from mcts import MCTS
from chess_helper import new_game, is_game_over, result, board_to_tensor, uci_index_map
import utils
import random

# Ustawienia treningu
NUM_ITERATIONS = 10  # liczba iteracji treningowych
GAMES_PER_ITERATION = 1  # liczba gier samouczenia na iterację
MCTS_SIMULATIONS = 100  # liczba symulacji MCTS na każdy ruch
STOCKFISH_PATH = "stockfish/stockfish.exe"  # ścieżka do Stockfisha
BATCH_SIZE = 64  # rozmiar batchu do trenowania
LEARNING_RATE = 0.001

# Parametry przejścia od Stockfisha do samodzielności
USE_STOCKFISH = True
STOCKFISH_USAGE_RATE = 1.0  # Początkowo pełna zależność od Stockfisha
STOCKFISH_DECAY = 0.1  # Co ile iteracji zmniejszamy wpływ Stockfisha
MIN_STOCKFISH_USAGE = 0.0  # Docelowa wartość (pełna samodzielność)


def self_play_game(model, engine, stockfish_usage):
    """Symuluje jedną grę self-play."""
    game_states = []
    mcts_policies = []
    rewards = []
    moves_list = []
    board = new_game()

    if random.random() < 0.2:
        for _ in range(random.randint(1, 4)):
            legal_moves = list(board.legal_moves)
            if legal_moves:
                board.push(random.choice(legal_moves))

    mcts = MCTS(model, simulations=MCTS_SIMULATIONS, engine=engine, stockfish_enabled=stockfish_usage > random.random())

    while not is_game_over(board):
        move = mcts.search(board)
        if move is None:
            break

        state_tensor = board_to_tensor(board)
        policy_probs, _ = mcts.evaluate_state(board)

        game_states.append(state_tensor)
        mcts_policies.append(policy_probs)
        if isinstance(move, str):
            move = board.parse_uci(move)
        moves_list.append(move.uci())

        rewards.append(result(board))
        board.push(move)

        if len(moves_list) >= 300:
            break

    game_result_str = "1-0" if result(board) == 1.0 else ("0-1" if result(board) == 0.0 else "1/2-1/2")
    return game_states, mcts_policies, rewards, moves_list, game_result_str


if __name__ == "__main__":
    device = utils.get_device()
    model = utils.load_or_initialize_model(ChessNet, "chess_model.pth", device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    engine = utils.init_stockfish(STOCKFISH_PATH)

    for iteration in range(1, NUM_ITERATIONS + 1):
        print(f"== Iteracja {iteration} ==")
        games_data = []

        for game_idx in range(GAMES_PER_ITERATION):
            states, policies, rewards, moves_list, game_result = self_play_game(model, engine, STOCKFISH_USAGE_RATE)
            print("GAME_IDX:", game_idx + 1)
            utils.save_game_pgn_separate(moves_list, game_result, f"{iteration}_{game_idx + 1}")
            for s, p, r in zip(states, policies, rewards):
                games_data.append((s, p, r))

        # Aktualizacja modelu
        model.train()
        random.shuffle(games_data)
        batch_count = max(1, len(games_data) // BATCH_SIZE)

        for batch_idx in range(batch_count):
            batch = games_data[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
            states_batch = [s for s, _, _ in batch]

            policy_batch = []
            for _, p, _ in batch:
                policy_vector = [0.0] * 4672
                for move_uci, prob in p.items():
                    if move_uci in uci_index_map():
                        policy_vector[uci_index_map()[move_uci]] = prob
                policy_batch.append(policy_vector)
            value_batch = [r for _, _, r in batch]

            state_tensor = torch.stack(states_batch).to(device)
            policy_tensor = torch.tensor(policy_batch, dtype=torch.float32).to(device)
            value_tensor = torch.tensor(value_batch, dtype=torch.float32).unsqueeze(1).to(device)

            pred_policy, pred_value = model(state_tensor)

            policy_loss = torch.sum(-policy_tensor * torch.log_softmax(pred_policy, dim=1)) / policy_tensor.size(0)
            value_loss = torch.mean((pred_value - value_tensor) ** 2)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Stopniowe zmniejszanie użycia Stockfisha
        STOCKFISH_USAGE_RATE = max(MIN_STOCKFISH_USAGE, STOCKFISH_USAGE_RATE - STOCKFISH_DECAY)

        utils.save_model(model, "chess_model.pth")

    utils.close_stockfish(engine)
    print("Trening zakończony. Model zapisany!")
