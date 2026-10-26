
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DQN_PATH = PROJECT_DIR / "dqn_lander.pt"
REINFORCE_PATH = PROJECT_DIR / "reinforce_lander.pt"

OUTPUT_DIR = PROJECT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

ENV_NAME = "LunarLander-v3"
NUM_EVAL_EPISODES = 100
VIDEO_SEED = 2026

DEVICE = torch.device("cpu")
torch.set_num_threads(1)


# ---------------------------------------------------------
# MODEL ARCHITECTURE
# Must match the architecture used during training.
# ---------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Copy the trained .pt file into the project folder."
        )

    model = Net().to(DEVICE)

    try:
        state_dict = torch.load(
            path, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        state_dict = torch.load(path, map_location=DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint: {path.name}")
    return model


# ---------------------------------------------------------
# ACTION SELECTION
# DQN: greedy action (highest Q-value)
# REINFORCE: sample an action from policy logits
# ---------------------------------------------------------

def choose_action(model, observation, agent_name, rng=None):
    state = torch.as_tensor(
        observation, dtype=torch.float32, device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        output = model(state).squeeze(0)

        if agent_name == "DQN":
            return int(torch.argmax(output).item())

        if agent_name == "REINFORCE":
            # Match the stochastic policy used during evaluation.
            # Seeded generator makes this evaluation reproducible.
            probs = torch.softmax(output, dim=-1).cpu().numpy()
            return int(rng.choice(len(probs), p=probs))

    raise ValueError(f"Unknown agent: {agent_name}")


# ---------------------------------------------------------
# NUMERICAL EVALUATION
# ---------------------------------------------------------

def evaluate_agent(model, agent_name, episodes=100):
    rows = []

    for episode in range(episodes):
        seed = 1000 + episode
        rng = np.random.default_rng(seed)

        env = gym.make(ENV_NAME)
        observation, info = env.reset(seed=seed)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = choose_action(
                model, observation, agent_name, rng
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)
            steps += 1

        env.close()

        # Explicit, simple success proxy:
        # environment terminated naturally and episode reward > 0.
        # This may differ from a custom "clean landing" metric
        # used in your original notebook.
        success = bool(terminated and total_reward > 0)

        rows.append({
            "agent": agent_name,
            "episode": episode + 1,
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success_proxy": success,
        })

    results = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / f"{agent_name.lower()}_evaluation.csv"
    results.to_csv(csv_path, index=False)

    summary = {
        "agent": agent_name,
        "episodes": len(results),
        "mean_reward": results["total_reward"].mean(),
        "std_reward": results["total_reward"].std(ddof=0),
        "mean_steps": results["steps"].mean(),
        "success_proxy_rate_percent": (
            results["success_proxy"].mean() * 100
        ),
        "csv_file": str(csv_path),
    }

    return results, summary


# ---------------------------------------------------------
# RECORD A SINGLE GAMEPLAY VIDEO
# ---------------------------------------------------------

def record_video(model, agent_name, seed=2026):
    video_path = OUTPUT_DIR / f"{agent_name.lower()}_lander.mp4"
    rng = np.random.default_rng(seed)

    env = gym.make(ENV_NAME, render_mode="rgb_array")

    observation, info = env.reset(seed=seed)
    frames = [env.render()]

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = choose_action(
            model, observation, agent_name, rng
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += float(reward)
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    env.close()

    # macro_block_size=1 avoids resizing frames to multiples of 16.
    with imageio.get_writer(
        video_path,
        fps=30,
        codec="libx264",
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    print(f"\n{agent_name} video saved: {video_path}")
    print(f"Video episode reward: {total_reward:.2f}")
    print(f"Video episode steps: {steps}")
    print(f"Video episode terminated: {terminated}")
    print(f"Video episode truncated: {truncated}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    print("Device:", DEVICE)
    print("Environment:", ENV_NAME)
    print("Training will NOT be run.")

    dqn_model = load_checkpoint(DQN_PATH)
    reinforce_model = load_checkpoint(REINFORCE_PATH)

    all_results = []
    summaries = []

    for agent_name, model in [
        ("DQN", dqn_model),
        ("REINFORCE", reinforce_model),
    ]:
        print(f"\nEvaluating {agent_name} over "
              f"{NUM_EVAL_EPISODES} episodes...")

        results, summary = evaluate_agent(
            model, agent_name, NUM_EVAL_EPISODES
        )

        all_results.append(results)
        summaries.append(summary)

    combined = pd.concat(all_results, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_evaluation_results.csv"
    combined.to_csv(combined_path, index=False)

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n========== NUMERICAL SUMMARY ==========")
    print(summary_df.to_string(index=False))
    print(f"\nAll episode results: {combined_path}")
    print(f"Summary CSV: {summary_path}")

    print("\nRecording DQN video...")
    record_video(dqn_model, "DQN", seed=VIDEO_SEED)

    print("\nRecording REINFORCE video...")
    record_video(
        reinforce_model, "REINFORCE", seed=VIDEO_SEED
    )

    print("\nFinished. Check the results folder:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(" -", path.name)


if __name__ == "__main__":
    main()
