import random

import numpy as np
import torch

from bcq import BCQ
from config import (
    DATASET_PATH,
    MODEL_PATH,
    RESULTS_DIR,
    CHECKPOINT_DIR,
    BATCH_SIZE,
    TRAINING_STEPS,
    SEED,
    create_project_dirs,
)
from dataset import OfflineDataset
from env import NavigationEnv


# Evaluation
EVAL_INTERVAL = 5000
EVAL_EPISODES = 30

# Early stopping
EARLY_STOP_START = 10000
EARLY_STOP_PATIENCE = 2

# Minimum meaningful improvement
MIN_SUCCESS_IMPROVEMENT = 0.01
MIN_REWARD_IMPROVEMENT = 0.10

# Safety
MAX_CRITIC_LOSS = 1000.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent, episodes=EVAL_EPISODES):
    # Preserve training RNG state so evaluation does not affect training
    cpu_rng_state = torch.get_rng_state()

    cuda_rng_state = None
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state_all()

    # Use the same random sequence for every evaluation
    torch.manual_seed(SEED + 1000)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 1000)

    env = NavigationEnv(seed=SEED + 1000)

    episode_rewards = []
    episode_steps = []

    successes = 0
    collisions = 0

    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)

            total_reward += reward

            if done:
                episode_rewards.append(total_reward)
                episode_steps.append(info["steps"])

                if info["success"]:
                    successes += 1

                if info["collision"]:
                    collisions += 1

                break

    # Restore training RNG state
    torch.set_rng_state(cpu_rng_state)

    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    return {
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "average_reward": float(np.mean(episode_rewards)),
        "average_steps": float(np.mean(episode_steps)),
    }


def is_better_model(metrics, best_success, best_reward):
    success = metrics["success_rate"]
    reward = metrics["average_reward"]

    # Success rate is the primary metric
    if success >= best_success + MIN_SUCCESS_IMPROVEMENT:
        return True

    # If success rate is effectively equal, use reward
    if abs(success - best_success) < MIN_SUCCESS_IMPROVEMENT:
        if reward >= best_reward + MIN_REWARD_IMPROVEMENT:
            return True

    return False


def train():
    create_project_dirs()
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OfflineDataset(DATASET_PATH, device=device)

    agent = BCQ(
        state_dim=4,
        action_dim=2,
        max_action=1.0,
        device=device,
    )

    vae_losses = []
    critic_losses = []
    actor_losses = []

    eval_steps = []
    eval_success = []
    eval_rewards = []
    eval_average_steps = []
    eval_collisions = []

    best_success = -1.0
    best_reward = -float("inf")

    no_improvement_count = 0
    stopped_early = False

    best_model_path = CHECKPOINT_DIR / "bcq_best.pt"

    print("Training started")
    print(f"Device: {device}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Maximum training steps: {TRAINING_STEPS}")
    print()

    print("Expected training behavior:")
    print("VAE loss       -> decrease and stabilize")
    print("Critic loss    -> fluctuate, then stabilize; must not explode")
    print("Actor loss     -> usually become more negative, then stabilize")
    print("Success rate   -> increase")
    print("Average reward -> increase")
    print("Average steps  -> decrease")
    print("Collision rate -> decrease")
    print()

    for step in range(1, TRAINING_STEPS + 1):
        batch = dataset.sample(BATCH_SIZE)
        losses = agent.train(batch)

        vae_loss = losses["vae_loss"]
        critic_loss = losses["critic_loss"]
        actor_loss = losses["actor_loss"]

        # Numerical stability checks
        if not np.isfinite(vae_loss):
            raise RuntimeError(
                f"VAE loss became invalid at step {step}: {vae_loss}"
            )

        if not np.isfinite(critic_loss):
            raise RuntimeError(
                f"Critic loss became invalid at step {step}: {critic_loss}"
            )

        if not np.isfinite(actor_loss):
            raise RuntimeError(
                f"Actor loss became invalid at step {step}: {actor_loss}"
            )

        if critic_loss > MAX_CRITIC_LOSS:
            raise RuntimeError(
                f"Critic loss exploded at step {step}: {critic_loss:.4f}"
            )

        vae_losses.append(vae_loss)
        critic_losses.append(critic_loss)
        actor_losses.append(actor_loss)

        # Training log
        if step % 1000 == 0:
            recent_vae = float(np.mean(vae_losses[-1000:]))
            recent_critic = float(np.mean(critic_losses[-1000:]))
            recent_actor = float(np.mean(actor_losses[-1000:]))

            print(
                f"Step {step:6d}/{TRAINING_STEPS} | "
                f"VAE {recent_vae:8.4f} | "
                f"Critic {recent_critic:8.4f} | "
                f"Actor {recent_actor:9.4f}"
            )

        # Policy evaluation
        if step % EVAL_INTERVAL == 0:
            metrics = evaluate(agent)

            success = metrics["success_rate"]
            reward = metrics["average_reward"]
            avg_steps = metrics["average_steps"]
            collision = metrics["collision_rate"]

            eval_steps.append(step)
            eval_success.append(success)
            eval_rewards.append(reward)
            eval_average_steps.append(avg_steps)
            eval_collisions.append(collision)

            print()
            print("------------------------------------------------------------")
            print(f"EVALUATION @ STEP {step}")
            print(f"Success rate:   {success * 100:6.2f}%")
            print(f"Average reward: {reward:8.3f}")
            print(f"Average steps:  {avg_steps:8.2f}")
            print(f"Collision rate: {collision * 100:6.2f}%")
            print("------------------------------------------------------------")

            if is_better_model(metrics, best_success, best_reward):
                best_success = success
                best_reward = reward
                no_improvement_count = 0

                agent.save(best_model_path)

                print("NEW BEST MODEL")
                print(f"Best success rate: {best_success * 100:.2f}%")
                print(f"Best average reward: {best_reward:.3f}")

            elif step >= EARLY_STOP_START:
                no_improvement_count += 1

                print(
                    f"No meaningful improvement: "
                    f"{no_improvement_count}/{EARLY_STOP_PATIENCE}"
                )

            print()

            # Stop when evaluation stops improving
            if (
                step >= EARLY_STOP_START
                and no_improvement_count >= EARLY_STOP_PATIENCE
            ):
                print("EARLY STOPPING")
                print(f"Stopped at step: {step}")
                stopped_early = True
                break

        # Periodic backup
        if step % 10000 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"bcq_{step}.pt"
            agent.save(checkpoint_path)

    # Restore the best policy found during training
    if best_model_path.exists():
        agent.load(best_model_path)

    agent.save(MODEL_PATH)

    # Save complete training history
    np.savez(
        RESULTS_DIR / "training_history.npz",
        vae_loss=np.asarray(vae_losses, dtype=np.float32),
        critic_loss=np.asarray(critic_losses, dtype=np.float32),
        actor_loss=np.asarray(actor_losses, dtype=np.float32),
        eval_steps=np.asarray(eval_steps, dtype=np.int32),
        eval_success=np.asarray(eval_success, dtype=np.float32),
        eval_rewards=np.asarray(eval_rewards, dtype=np.float32),
        eval_average_steps=np.asarray(eval_average_steps, dtype=np.float32),
        eval_collisions=np.asarray(eval_collisions, dtype=np.float32),
    )

    print()
    print("============================================================")
    print("TRAINING FINISHED")
    print("============================================================")
    print(f"Early stopping: {stopped_early}")
    print(f"Best success rate: {best_success * 100:.2f}%")
    print(f"Best average reward: {best_reward:.3f}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Final model: {MODEL_PATH}")
    print(f"Training history: {RESULTS_DIR / 'training_history.npz'}")


if __name__ == "__main__":
    train()