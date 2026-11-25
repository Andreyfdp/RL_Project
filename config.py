from pathlib import Path

# PROJECT PATHS

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
EXPERIMENT_DIR = RESULTS_DIR / "experiments"

DATASET_PATH = DATA_DIR / "offline_dataset.npz"
MODEL_PATH = CHECKPOINT_DIR / "bcq_final.pt"


def create_project_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# RANDOMNESS

SEED = 42


# ENVIRONMENT

MAP_WIDTH = 10.0
MAP_HEIGHT = 10.0

START_POSITION = (1.0, 1.0)
GOAL_POSITION = (9.0, 9.0)

# Obstacle bounds: (x_min, y_min, x_max, y_max).
OBSTACLES = [
    (4.2, 2.2, 5.8, 7.8),
]

# Scale bounded actions to map movement.
STEP_SIZE = 0.35

AGENT_RADIUS = 0.15
GOAL_RADIUS = 0.45

MAX_EPISODE_STEPS = 120


# REWARD

# Reward for getting closer to the goal.
PROGRESS_REWARD_SCALE = 5.0

# Small penalty for every step.
STEP_PENALTY = 0.02

# Terminal rewards.
GOAL_REWARD = 10.0
COLLISION_PENALTY = 5.0


# OFFLINE DATASET

DATASET_EPISODES = 1000

# Standard deviation of Gaussian noise in behavior policy.
BEHAVIOR_NOISE_STD = 0.30

# Probability of a completely random action in behavior policy.
RANDOM_ACTION_PROB = 0.05

# Experiment datasets (collect_dataset.py --mode ...)

# Random shift of the start position (uniform in [-x, x] for both coordinates).
START_NOISE = 0.30

# Configure action noise, random actions and good-policy fractions.
DATASET_MODES = {
    "easy": {
        "noise_std": 0.30,
        "random_prob": 0.05,
        "good_fraction": 1.0,
        "episodes": 1000,
    },
    "noisy": {
        "noise_std": 0.60,
        "random_prob": 0.25,
        "good_fraction": 1.0,
        "episodes": 1000,
    },
    "mixed": {
        "noise_std": 0.30,
        "random_prob": 0.05,
        "good_fraction": 0.4,
        "episodes": 1000,
    },
    "small": {
        "noise_std": 0.30,
        "random_prob": 0.05,
        "good_fraction": 0.4,
        "episodes": 100,
    },
}


# BCQ TRAINING

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


# EVALUATION

EVALUATION_EPISODES = 100


# QUICK TEST

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
