# pygame_gui.py
import pygame
import chess
from model import ChessNet
from mcts import MCTS
import chess_helper
import utils

# Ustawienia okna gry
WIDTH, HEIGHT = 600, 600
SQ_SIZE = WIDTH // 8  # rozmiar pola zakładając kwadratową planszę
FPS = 30

# Kolory
LIGHT_SQ_COLOR = (240, 217, 181)
DARK_SQ_COLOR = (181, 136, 99)
HIGHLIGHT_COLOR = (0, 255, 0)  # np. zielony do podświetleń

# Załaduj obrazy figur (zakładamy istnienie folderu "assets" z obrazkami)
pieces_images = {}
pieces = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
for p in pieces:
    # np. nazwy plików "P.png", "p.png" itd.
    try:
        img = pygame.image.load(f"assets/{p}.png")
        pieces_images[p] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
    except:
        pieces_images[p] = None  # jeśli brak pliku, pozostaw None


def draw_board(screen, board, selected_square=None, hint_move=None):
    """Rysuje szachownicę i figury. Jeśli selected_square jest ustawione, podświetla ten kwadrat.
       Jeśli hint_move (ruch podpowiedzi) jest podany, podświetla sugerowany ruch."""
    # Rysuj pola
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQ_COLOR if (rank + file) % 2 == 0 else DARK_SQ_COLOR
            rect = pygame.Rect(file * SQ_SIZE, rank * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)
    # Jeśli zaznaczono jakieś pole (wybrana figura), podświetl
    if selected_square is not None:
        s_rank = 7 - (selected_square // 8)
        s_file = selected_square % 8
        rect = pygame.Rect(s_file * SQ_SIZE, s_rank * SQ_SIZE, SQ_SIZE, SQ_SIZE)
        pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 5)
    # Jeśli jest podpowiedź ruchu, podświetl pola start i meta ruchu
    if hint_move:
        try:
            hint_move_obj = board.parse_uci(hint_move)
            start_sq = hint_move_obj.from_square
            end_sq = hint_move_obj.to_square
            for sq in [start_sq, end_sq]:
                r = 7 - (sq // 8);
                c = sq % 8
                rect = pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 5)
        except:
            pass
    # Rysuj figury na polach
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # współrzędne na ekranie
            rank = 7 - (square // 8)
            file = square % 8
            symbol = piece.symbol()  # 'P', 'p', 'N' etc.
            if pieces_images.get(symbol):
                screen.blit(pieces_images[symbol], (file * SQ_SIZE, rank * SQ_SIZE))
            else:
                # Jeśli brak obrazu, narysuj kółko jako placeholder
                color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                pygame.draw.circle(screen, color, (file * SQ_SIZE + SQ_SIZE // 2, rank * SQ_SIZE + SQ_SIZE // 2),
                                   SQ_SIZE // 3)


def main(human_color='white', model_path="model_iter_10.pth", advisor_mode=False):
    """Główna pętla gry. human_color: 'white' lub 'black' – kolor gracza człowieka.
       model_path: ścieżka do wytrenowanego modelu AI.
       advisor_mode: jeśli True, AI tylko podpowiada ruchy zamiast grać przeciwko."""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Szachy AI")
    clock = pygame.time.Clock()

    # Inicjalizacja stanu gry
    board = chess_helper.new_game()
    human_turn = (human_color == 'white')  # jeśli człowiek gra białymi, zaczyna człowiek
    selected_sq = None  # zaznaczony kwadrat (wybrana figura do ruchu)
    move_hints = None  # podpowiedziany ruch (UCI) jeśli tryb doradczy

    # Załaduj model AI
    device = utils.get_device()
    model = ChessNet()
    model = utils.load_model(ChessNet, model_path, device=device)
    mcts_player = MCTS(model, simulations=100)  # AI gracz (100 symulacji dla szybkości)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Obsługa kliknięć myszą (wybór i wykonanie ruchu przez człowieka)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # lewy przycisk
                x, y = event.pos
                file = x // SQ_SIZE
                rank = y // SQ_SIZE
                clicked_sq = chess.square(file, 7 - rank)
                if selected_sq is None:
                    # Nie wybrano jeszcze figury do ruchu -> wybór figury jeśli należy do człowieka
                    piece = board.piece_at(clicked_sq)
                    if piece and ((piece.color == chess.WHITE and human_color == 'white') or
                                  (piece.color == chess.BLACK and human_color == 'black')):
                        selected_sq = clicked_sq
                else:
                    # Jakiś ruch jest wybrany, spróbuj wykonać ruch z selected_sq na clicked_sq
                    move = chess.Move(from_square=selected_sq, to_square=clicked_sq)
                    if move in board.legal_moves:
                        board.push(move)
                        movestr = move.uci()
                        # Zapisz ruch do ewentualnego PGN (opcjonalnie)
                        # Jeśli tryb doradczy, AI nie wykonuje ruchu, więc człowiek może grać sam (lub z podpowiedziami)
                        if not advisor_mode:
                            human_turn = not human_turn  # zmiana tury, teraz ruch AI
                        move_hints = None  # zresetuj podpowiedź po wykonaniu ruchu
                    # reset wyboru
                    selected_sq = None
            # Obsługa klawiatury (np. wywołanie trybu doradczego/hint lub rezygnacja)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_h and advisor_mode:
                    # Klawisz 'h' wywołuje podpowiedź ruchu AI dla aktualnej pozycji
                    best_move = mcts_player.search(board)
                    move_hints = best_move.uci() if best_move else None
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Jeśli nie tryb doradczy i jest tura AI, wykonaj ruch AI
        if not advisor_mode:
            if (board.turn == chess.WHITE and human_color == 'black') or (
                    board.turn == chess.BLACK and human_color == 'white'):
                # AI's turn
                ai_move = mcts_player.search(board)
                if ai_move is not None:
                    board.push(ai_move)
                human_turn = not human_turn  # wraca tura do człowieka

        # Rysuj interfejs
        draw_board(screen, board, selected_square=selected_sq, hint_move=move_hints)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    # Uruchom grę: domyślnie człowiek gra białymi, AI czarnymi.
    # Aby uruchomić tryb doradczy, można przekazać advisor_mode=True.
    main(human_color='white', model_path='model_iter_10.pth', advisor_mode=False)
