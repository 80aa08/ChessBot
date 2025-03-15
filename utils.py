import torch
import os
import chess.engine
import datetime
import chess.pgn


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_or_initialize_model(model_class, path, device):
    model = model_class().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
    return model


def init_stockfish(path):
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        return engine
    except Exception as e:
        print(f"Błąd inicjalizacji Stockfisha: {e}")
        return None


def close_stockfish(engine):
    if engine:
        try:
            engine.quit()
        except:
            pass


def save_game_pgn_separate(moves_list, result, game_index):
    file_name = f"game_{game_index}.pgn"
    game = chess.pgn.Game()
    game.headers["Event"] = "Self Play Training"
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["Result"] = result
    game.headers["White"] = "AI"
    game.headers["Black"] = "AI"

    node = game
    for move in moves_list:
        move_obj = chess.Move.from_uci(move)
        node = node.add_variation(move_obj)
    with open(file_name, "w") as f:
        f.write(str(game))
    print(f"[INFO] Zapisano grę w {file_name}")
