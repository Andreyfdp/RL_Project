from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

DATASET_PATH = DATA_DIR / "offline_dataset.npz"
MODEL_PATH = CHECKPOINT_DIR / "bcq_final.pt"


def create_project_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RANDOMNESS
# ============================================================

SEED = 42


# ============================================================
# ENVIRONMENT
# ============================================================

MAP_WIDTH = 10.0
MAP_HEIGHT = 10.0

START_POSITION = (1.0, 1.0)
GOAL_POSITION = (9.0, 9.0)

# Rectangle:
# (x_min, y_min, x_max, y_max)
OBSTACLES = [
    (4.2, 2.2, 5.8, 7.8),
]

# Maximum movement during one step.
# Action itself will always be in [-1, 1]^2.
STEP_SIZE = 0.35

AGENT_RADIUS = 0.15
GOAL_RADIUS = 0.45

MAX_EPISODE_STEPS = 120


# ============================================================
# REWARD
# ============================================================

# Reward for getting closer to the goal.
PROGRESS_REWARD_SCALE = 5.0

# Small penalty for every step.
STEP_PENALTY = 0.02

# Terminal rewards.
GOAL_REWARD = 10.0
COLLISION_PENALTY = 5.0


# ============================================================
# OFFLINE DATASET
# ============================================================

DATASET_EPISODES = 1000

# Standard deviation of Gaussian noise in behavior policy.
BEHAVIOR_NOISE_STD = 0.30


# ============================================================
# BCQ TRAINING
# ============================================================

BATCH_SIZE = 256

TRAINING_STEPS = 100_000

GAMMA = 0.99
TAU = 0.005

LEARNING_RATE = 1e-3

# Maximum perturbation applied by BCQ actor.
PHI = 0.05

# Number of VAE actions sampled when BCQ selects an action.
NUM_ACTION_SAMPLES = 100

# VAE latent dimension.
LATENT_DIM = 4


# ============================================================
# EVALUATION
# ============================================================

EVALUATION_EPISODES = 100


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    create_project_dirs()

    print("BCQ project configuration loaded successfully.")
    print()
    print(f"Base directory:       {BASE_DIR}")
    print(f"Dataset path:         {DATASET_PATH}")
    print(f"Model path:           {MODEL_PATH}")
    print()
    print(f"Map size:             {MAP_WIDTH} x {MAP_HEIGHT}")
    print(f"Start:                {START_POSITION}")
    print(f"Goal:                 {GOAL_POSITION}")
    print(f"Obstacles:            {OBSTACLES}")
    print(f"Max episode steps:    {MAX_EPISODE_STEPS}")
    print()
    print("Directories are ready.")