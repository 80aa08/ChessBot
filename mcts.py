import math
import torch
import numpy as np
from chess_helper import board_to_tensor


class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.wins = 0.0
        self.visits = 0
        self.prior = prior

    def expand(self, policy_probs):
        for move in self.state.legal_moves:
            uci = move.uci()
            if uci not in self.children:
                new_state = self.state.copy()
                new_state.push(move)
                self.children[uci] = MCTSNode(new_state, parent=self, prior=policy_probs.get(uci, 0.001))

    def best_child(self, c_puct=1.0):
        best_score, best_child = -float("inf"), None
        for child in self.children.values():
            q_value = (child.wins / (child.visits + 1e-6)) if child.visits else 0
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            if score > best_score:
                best_score, best_child = score, child
        return best_child


class MCTS:
    def __init__(self, model, simulations=100, engine=None, stockfish_enabled=True):
        self.model = model
        self.model.eval()
        self.simulations = simulations
        self.engine = engine
        self.stockfish_enabled = stockfish_enabled

    def search(self, board):
        root = MCTSNode(board)

        # 📌 Dodanie szumu Dirichleta dla większej różnorodności
        if root.children:
            dirichlet_noise = np.random.dirichlet([0.03] * len(root.children))
            epsilon = 0.25  # Współczynnik eksploracji
            for i, (move, child) in enumerate(root.children.items()):
                child.prior = (1 - epsilon) * child.prior + epsilon * dirichlet_noise[i]

        for _ in range(self.simulations):
            value = self._simulate(root)
            self.backpropagate(root, value)

        return max(root.children.items(), key=lambda x: x[1].visits)[0]

    def _simulate(self, node):
        while node.children:
            node = node.best_child()
        policy_probs, value = self.evaluate_state(node.state)
        node.expand(policy_probs)
        return value

    def evaluate_state(self, board):
        tensor = board_to_tensor(board).unsqueeze(0)
        policy_logits, value_tensor = self.model(tensor)
        policy = torch.softmax(policy_logits, dim=1)[0].detach().cpu().numpy()
        return {m.uci(): policy[i] for i, m in enumerate(board.legal_moves)}, value_tensor.item()

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.wins += value
            value *= -0.95
            node = node.parent
