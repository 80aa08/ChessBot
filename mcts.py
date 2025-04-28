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

class MCTS:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def search(self, env):
        root = Node(0)
        state = env._get_state().unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        priors = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        moves = env.legal_moves()
        # expand root
        for idx, move in enumerate(moves):
            root.children[move] = Node(priors[idx])
        # Dirichlet noise
        noise = np.random.dirichlet([self.config.DIRICHLET_ALPHA]*len(moves))
        for idx, move in enumerate(moves):
            root.children[move].P = (1 - self.config.EPSILON)*priors[idx] + self.config.EPSILON*noise[idx]
        # simulations
        for _ in range(self.config.NUM_SIMULATIONS):
            self._simulate(env, root)
        # pick most visited
        return max(root.children.items(), key=lambda item: item[1].N)[0]

    def _simulate(self, env, node):
        if not node.children:
            state = env._get_state().unsqueeze(0).to(self.device)
            logits, value = self.model(state)
            priors = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
            for idx, move in enumerate(env.legal_moves()):
                node.children[move] = Node(priors[idx])
            return value.item()
        total_N = sum(child.N for child in node.children.values())
        best_score, best_move, best_child = -float('inf'), None, None
        for move, child in node.children.items():
            U = self.config.C_PUCT * child.P * math.sqrt(total_N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        env_copy = deepcopy(env)
        env_copy.step(best_move)
        v = self._simulate(env_copy, best_child)
        best_child.N += 1
        best_child.W += v
        best_child.Q = best_child.W / best_child.N
        return -v
