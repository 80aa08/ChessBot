import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import os
from config import Config
from model import ChessNet
from chess_env import ChessEnv
from mcts import MCTS
from utils import save_pgn, augment_examples

class SelfPlayDataset(Dataset):
    def __init__(self, examples):
        self.states, self.policies, self.values = zip(*examples)
    def __len__(self): return len(self.states)
    def __getitem__(self, idx): return self.states[idx], self.policies[idx], self.values[idx]

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()
    model = ChessNet().to(device)
    policy_dim = model.policy_fc.out_features
    if config.USE_TORCHSCRIPT:
        model.eval()
        example = torch.zeros(1, config.INPUT_CHANNELS, 8, 8, device=device)
        model = torch.jit.trace(model, example)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter(config.LOG_DIR)

    best_val_loss = float('inf')
    patience = 0

    for iteration in range(1, config.NUM_EPOCHS + 1):
        examples = []
        for _ in range(config.NUM_SELFPLAY_GAMES):
            env = ChessEnv(device=device)
            mcts = MCTS(model, config, device)
            state = env.reset()
            history = []
            done = False
            while not done:
                move = mcts.search(env)
                pi = torch.zeros(policy_dim)
                state, reward, done = env.step(move)
                history.append((state, pi, reward))
            save_pgn(env)
            examples.extend(history)

        if config.USE_AUGMENT_SYMMETRIES:
            examples = augment_examples(examples)

        dataset = SelfPlayDataset(examples)
        loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        model.train()
        for epoch in range(config.NUM_EPOCHS):
            for i, (s, pi, z) in enumerate(loader):
                s, pi, z = s.to(device), pi.to(device), z.to(device)
                logits, value = model(s)
                loss_p = -torch.mean(torch.sum(pi * torch.log_softmax(logits, dim=1), dim=1))
                loss_v = torch.mean((z - value.view(-1))**2)
                loss = loss_p + loss_v
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 10 == 0:
                    writer.add_scalar('Loss/total', loss.item(), iteration * len(loader) + i)

        val_examples = []
        for _ in range(config.NUM_VALIDATION_GAMES):
            env = ChessEnv(device=device)
            mcts = MCTS(model, config, device)
            state = env.reset()
            done = False
            history = []
            while not done:
                move = mcts.search(env)
                pi = torch.zeros(policy_dim)
                state, reward, done = env.step(move)
                history.append((state, pi, reward))
            save_pgn(env, path="./validation")
            val_examples.extend(history)

        model.eval()
        total_val_loss = 0
        if val_examples:
            val_dataset = SelfPlayDataset(val_examples)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
            with torch.no_grad():
                for s, pi, z in val_loader:
                    s, pi, z = s.to(device), pi.to(device), z.to(device)
                    logits, value = model(s)
                    loss_p = -torch.mean(torch.sum(pi * torch.log_softmax(logits, dim=1), dim=1))
                    loss_v = torch.mean((z - value.view(-1))**2)
                    total_val_loss += (loss_p + loss_v).item() * s.size(0)
            val_loss = total_val_loss / len(val_dataset)
            writer.add_scalar('Loss/validation', val_loss, iteration)
        else:
            val_loss = float('nan')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        if patience > config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at iteration {iteration}")
            break

        if iteration % config.SAVE_INTERVAL == 0:
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.MODEL_DIR}/model_iter_{iteration}.pt")
    writer.close()