class Config:
    # MCTS parameters
    NUM_SIMULATIONS = 100           # number of MCTS rollouts per move
    C_PUCT = 1.0                    # exploration vs exploitation constant

    # Dirichlet noise for root exploration
    DIRICHLET_ALPHA = 0.3
    EPSILON = 0.25

    # Inference batching
    INFER_BATCH_SIZE = 16          # batch size for model inference in MCTS

    # Neural network training
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 64
    NUM_EPOCHS = 1

    # Network architecture
    INPUT_CHANNELS = 18            # e.g., 12 piece planes + extras
    RESIDUAL_BLOCKS = 5
    CHANNELS = 128

    # Self-play
    NUM_SELFPLAY_GAMES = 1      # games per iteration
    MAX_GAME_LENGTH = 50

    # Validation (early stopping)
    NUM_VALIDATION_GAMES = 1
    EARLY_STOPPING_PATIENCE = 3

    # Data augmentation (board symmetries)
    USE_AUGMENT_SYMMETRIES = True

    # Logging / saving
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    SAVE_INTERVAL = 1             # iterations

    # GPU optimizations
    USE_TORCHSCRIPT = True