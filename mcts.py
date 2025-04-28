import math
import torch
import numpy as np
from copy import deepcopy
from config import Config

class Node:
    def __init__(self, prior):
        self.P = prior
        self.N = 0
        self.W = 0
        self.Q = 0
        self.children = {}

    def expand(self, priors, legal_moves):
        for p, move in zip(priors, legal_moves):
            self.children[move] = Node(p)

    def backup(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

class MCTS:
    def __init__(self, model, config, device):
        self.model = model.eval()
        self.config = config
        self.device = device
        self.batch_queue = []

    def search(self, env):
        root = Node(0)
        self.batch_queue.append((deepcopy(env), root))
        self._evaluate_batch()
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA] * len(moves))
        for m, n in zip(moves, noise):
            child = root.children[m]
            child.P = (1 - self.config.EPSILON) * child.P + self.config.EPSILON * n
        for _ in range(self.config.NUM_SIMULATIONS):
            env_copy = deepcopy(env)
            self._simulate(env_copy, root)
            if len(self.batch_queue) >= self.config.INFER_BATCH_SIZE:
                self._evaluate_batch()
        self._evaluate_batch()
        return max(root.children.items(), key=lambda item: item[1].N)[0]

    def _simulate(self, env, node):
        if not node.children:
            self.batch_queue.append((deepcopy(env), node))
            return
        total_N = sum(child.N for child in node.children.values())
        best_score, best_move, best_child = -float('inf'), None, None
        for move, child in node.children.items():
            U = self.config.C_PUCT * child.P * math.sqrt(total_N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        env.step(best_move)
        self._simulate(env, best_child)
        best_child.backup(-best_child.Q)

    def _evaluate_batch(self):
        if not self.batch_queue:
            return
        chunk = self.batch_queue[:self.config.INFER_BATCH_SIZE]
        states = torch.stack([env._get_state() for env, _ in chunk]).to(self.device)
        with torch.no_grad():
            logits, values = self.model(states)
            priors = torch.softmax(logits, dim=1).cpu().numpy()
            values = values.cpu().numpy()
        for (env_state, node), prior, value in zip(chunk, priors, values):
            legal = env_state.legal_moves()
            node.expand(prior[:len(legal)], legal)
            node.backup(value.item())
        self.batch_queue = self.batch_queue[self.config.INFER_BATCH_SIZE:]