
import os
import tempfile
from pathlib import Path

import streamlit as st
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------
# PATHS AND DEVICE
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

DQN_PATH = BASE_DIR / "dqn_lander.pt"
REINFORCE_PATH = BASE_DIR / "reinforce_lander.pt"

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# MODEL ARCHITECTURE
# Must match the saved checkpoint architecture exactly.
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


def load_checkpoint(filename):
    if not filename.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {filename}\n"
            "Place the checkpoint beside app.py."
        )

    model = Net().to(DEVICE)

    state = torch.load(
        filename,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()
    return model


# Load pretrained models. No training takes place.
dqn = load_checkpoint(DQN_PATH)
reinforce = load_checkpoint(REINFORCE_PATH)


# ---------------------------------------------------------
# LIVE SIMULATION
# ---------------------------------------------------------

def fly(agent_name, seed, noise_sigma, max_steps):
    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
    )

    frames = []
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    try:
        obs, _ = env.reset(seed=int(seed))

        initial_frame = env.render()
        if initial_frame is not None:
            frames.append(initial_frame)

        while not (terminated or truncated) and steps < int(max_steps):
            observed = np.asarray(obs, dtype=np.float32).copy()

            if float(noise_sigma) > 0:
                observed += np.random.normal(
                    0.0,
                    float(noise_sigma),
                    observed.shape,
                ).astype(np.float32)

            state = torch.as_tensor(
                observed,
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                if agent_name == "Random":
                    action = env.action_space.sample()

                elif agent_name == "DQN (greedy)":
                    action = int(
                        torch.argmax(dqn(state), dim=1).item()
                    )

                else:
                    logits = reinforce(state)
                    action = int(
                        Categorical(logits=logits).sample().item()
                    )

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)
            steps += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)

    finally:
        env.close()

    if not frames:
        raise RuntimeError(
            "The environment did not return any video frames."
        )

    # Create a unique MP4 filename.
    handle = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False,
    )
    video_path = handle.name
    handle.close()

    imageio.mimsave(
        video_path,
        frames,
        fps=30,
        macro_block_size=1,
    )

    if terminated:
        status = "Episode terminated"
    elif truncated:
        status = "Episode truncated by the environment"
    else:
        status = "Stopped at the configured step limit"

    result = (
        f"Agent: {agent_name}\n"
        f"Total reward: {total_reward:.2f}\n"
        f"Steps: {steps}\n"
        f"Status: {status}\n"
        f"Seed: {int(seed)}\n"
        f"Sensor noise: {float(noise_sigma):.2f}\n\n"
        "Note: termination alone does not prove a successful landing."
    )

    return video_path, result


# ---------------------------------------------------------
# EVALUATION CSV LOADING
# ---------------------------------------------------------

def find_column(df, candidates):
    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None

def load_evaluation_data():
    """Load evaluation data and return exactly two values:
    (DataFrame, source file path).
    """
    summary_file = RESULTS_DIR / "summary.csv"
    all_results_file = RESULTS_DIR / "all_evaluation_results.csv"

    if summary_file.exists():
        return pd.read_csv(summary_file), str(summary_file)

    if all_results_file.exists():
        return pd.read_csv(all_results_file), str(all_results_file)

    raise FileNotFoundError(
        f"No evaluation CSV found in: {RESULTS_DIR}"
    )

def calculate_summary(df):
    """Create a consistent model-comparison table from either evaluation CSV format."""

    expected_columns = [
        "agent",
        "episodes",
        "mean_reward",
        "std_reward",
        "mean_steps",
        "success_proxy_rate_percent",
    ]

    if df is None or df.empty:
        return pd.DataFrame(columns=expected_columns)

    # Case 1: The CSV already contains summary statistics.
    if {"agent", "mean_reward", "mean_steps"}.issubset(df.columns):
        result = df.copy()

        if "episodes" not in result.columns:
            result["episodes"] = 0

        if "std_reward" not in result.columns:
            result["std_reward"] = 0.0

        if "success_proxy_rate_percent" not in result.columns:
            result["success_proxy_rate_percent"] = 0.0

        result = result[expected_columns].copy()

        for col in [
            "episodes",
            "mean_reward",
            "std_reward",
            "mean_steps",
            "success_proxy_rate_percent",
        ]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        return result

    # Case 2: Calculate statistics from individual episode results.
    required = {"agent", "total_reward", "steps"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Evaluation CSV is missing required columns. "
            f"Found: {df.columns.tolist()}"
        )

    data = df.copy()
    data["total_reward"] = pd.to_numeric(
        data["total_reward"], errors="coerce"
    )
    data["steps"] = pd.to_numeric(data["steps"], errors="coerce")

    rows = []

    for agent, group in data.groupby("agent"):
        rewards = group["total_reward"].dropna()
        steps = group["steps"].dropna()

        if "success_proxy" in group.columns:
            success = (
                group["success_proxy"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
            success_rate = success.mean() * 100
        else:
            success_rate = 0.0

        rows.append({
            "agent": agent,
            "episodes": len(group),
            "mean_reward": rewards.mean() if not rewards.empty else 0.0,
            "std_reward": rewards.std() if len(rewards) > 1 else 0.0,
            "mean_steps": steps.mean() if not steps.empty else 0.0,
            "success_proxy_rate_percent": success_rate,
        })

    return pd.DataFrame(rows, columns=expected_columns)


def make_bar_chart(summary, column, title, ylabel):
    if summary.empty or column not in summary.columns:
        return None

    plot_data = summary.dropna(subset=[column])

    if plot_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(
        plot_data["agent"],
        plot_data[column],
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    return fig


def refresh_evaluation():
    df, source_path = load_evaluation_data()

    if df.empty:
        message = (
            "### No evaluation data found\n\n"
            "Run `python run_models.py` first. This evaluates your "
            "saved models and creates the results CSV files; it does "
            "not retrain them."
        )
        return message, None, None, None

    summary = calculate_summary(df)

    if summary.empty:
        message = (
            f"Found `{source_path.name}`, but its columns did not match "
            "the expected agent and metric names.\n\n"
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
        return message, None, None, None

    display = summary.rename(columns={
        "agent": "Agent",
        "episodes": "Episodes",
        "mean_reward": "Mean reward",
        "std_reward": "Reward std. dev.",
        "mean_steps": "Mean steps",
        "success_proxy_rate_percent": "Success proxy (%)",
    }).copy()

    for column in [
        "Mean reward",
        "Reward std. dev.",
        "Mean steps",
        "Success proxy (%)",
    ]:
        display[column] = display[column].apply(
            lambda value: f"{value:.2f}"
            if pd.notna(value)
            else "N/A"
        )

    table = display.to_markdown(index=False)

    message = (
        f"### Evaluation summary\n\n"
        f"Source: `{source_path}`\n\n"
        f"{table}\n\n"
        "Success proxy values use the definition stored in the evaluation "
        "data. They are not automatically equivalent to an official "
        "Gymnasium success rate."
    )

    reward_plot = make_bar_chart(
        summary,
        "mean_reward",
        "Mean Evaluation Reward",
        "Mean reward",
    )

    steps_plot = make_bar_chart(
        summary,
        "mean_steps",
        "Average Episode Length",
        "Steps",
    )

    success_plot = make_bar_chart(
        summary,
        "success_proxy_rate_percent",
        "Success Proxy Comparison",
        "Percent",
    )

    return message, reward_plot, steps_plot, success_plot


# ---------------------------------------------------------
# HISTORICAL TRAINING CURVE
# Only use real logs. Never fabricate training history.
# ---------------------------------------------------------

def training_history():
    candidates = [
        BASE_DIR / "training_history.csv",
        BASE_DIR / "training_rewards.csv",
        RESULTS_DIR / "training_history.csv",
        RESULTS_DIR / "training_rewards.csv",
        RESULTS_DIR / "training_log.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        episode_col = find_column(
            df,
            ["episode", "episodes", "timestep", "step"],
        )

        reward_col = find_column(
            df,
            ["reward", "episode_reward", "total_reward", "return"],
        )

        if episode_col is None or reward_col is None:
            continue

        episodes = pd.to_numeric(
            df[episode_col],
            errors="coerce",
        )

        rewards = pd.to_numeric(
            df[reward_col],
            errors="coerce",
        )

        valid = episodes.notna() & rewards.notna()

        if not valid.any():
            continue

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            episodes[valid],
            rewards[valid],
            alpha=0.45,
            label="Episode reward",
        )

        if valid.sum() >= 10:
            rolling = rewards[valid].rolling(
                10,
                min_periods=1,
            ).mean()

            ax.plot(
                episodes[valid],
                rolling,
                label="10-episode rolling mean",
            )

            ax.legend()

        ax.set_title("Historical Training Reward")
        ax.set_xlabel(episode_col)
        ax.set_ylabel(reward_col)
        ax.grid(alpha=0.25)
        fig.tight_layout()

        return (
            f"Loaded genuine training history from `{path.name}`.",
            fig,
        )

    evaluation_files = [
        (RESULTS_DIR / "all_evaluation_results.csv", None),
        (RESULTS_DIR / "dqn_evaluation.csv", "DQN"),
        (RESULTS_DIR / "reinforce_evaluation.csv", "REINFORCE"),
    ]

    for path, default_agent in evaluation_files:
        if not path.is_file():
            continue

        df = pd.read_csv(path)
        episode_col = find_column(
            df,
            ["episode", "episodes", "timestep", "step"],
        )
        reward_col = find_column(
            df,
            ["reward", "episode_reward", "total_reward", "return"],
        )
        agent_col = find_column(df, ["agent", "model"])

        if episode_col is None or reward_col is None:
            continue

        episodes = pd.to_numeric(df[episode_col], errors="coerce")
        rewards = pd.to_numeric(df[reward_col], errors="coerce")
        valid = episodes.notna() & rewards.notna()

        if not valid.any():
            continue

        plot_data = df.loc[valid].copy()
        plot_data["_episode"] = episodes[valid]
        plot_data["_reward"] = rewards[valid]

        fig, ax = plt.subplots(figsize=(8, 4))
        if agent_col is not None:
            groups = plot_data.groupby(agent_col)
        else:
            groups = [(default_agent or path.stem, plot_data)]

        for agent, group in groups:
            group = group.sort_values("_episode")
            ax.plot(
                group["_episode"],
                group["_reward"],
                marker=".",
                label=str(agent),
            )

        ax.set_title("Evaluation Episode Rewards (Not Training History)")
        ax.set_xlabel("Evaluation episode")
        ax.set_ylabel("Episode reward")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()

        return (
            "No original training log was found, so the training learning "
            "curve cannot be reconstructed. Showing existing evaluation "
            "episode rewards instead; these are not training results. "
            "No retraining has been performed.",
            fig,
        )

    return (
        "No historical training log or per-episode evaluation results were "
        "found. The final checkpoint files do not contain enough "
        "information to reconstruct the original learning curve. Add the "
        "original training log CSV to display it here. No retraining has "
        "been performed.",
        None,
    )


# ---------------------------------------------------------
# SAVED DEMONSTRATION VIDEOS
# ---------------------------------------------------------

def existing_video(filename):
    path = RESULTS_DIR / filename
    return str(path) if path.is_file() else None

# ---------------------------------------------------------
# STREAMLIT INTERFACE
# Run with: streamlit run streamlit_app.py
# ---------------------------------------------------------

st.set_page_config(
    page_title="LunarLander-v3: DQN vs REINFORCE",
    page_icon="🚀",
    layout="wide",
)

st.title("LunarLander-v3: DQN vs REINFORCE")
st.markdown(
    """
This dashboard runs saved PyTorch checkpoints in the real Gymnasium
`LunarLander-v3` environment. It does **not** train or update either model.

Evaluation charts read existing CSV results. Historical training curves are
shown only when genuine training logs are available.
"""
)

tab_live, tab_compare, tab_history, tab_videos = st.tabs(
    [
        "Live Simulation",
        "Model Comparison",
        "Training History",
        "Saved Demonstration Videos",
    ]
)

with tab_live:
    st.subheader("Run a live landing episode")
    col1, col2 = st.columns(2)

    with col1:
        agent = st.selectbox(
            "Select agent",
            ["DQN (greedy)", "REINFORCE (sampled)", "Random"],
            index=0,
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=2_147_483_647,
            value=42, step=1,
        )

    with col2:
        noise = st.slider(
            "Sensor noise σ", min_value=0.0, max_value=0.5,
            value=0.0, step=0.05,
        )
        max_steps = st.slider(
            "Maximum episode steps", min_value=200, max_value=1500,
            value=1000, step=100,
        )

    if st.button("Run Landing Episode", type="primary", key="run_episode"):
        with st.spinner("Running the LunarLander episode and recording video..."):
            try:
                video_path, result = fly(agent, seed, noise, max_steps)
                st.session_state["live_result"] = result
                with open(video_path, "rb") as video_file:
                    st.session_state["live_video_bytes"] = video_file.read()
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            except Exception as exc:
                st.error(f"Episode failed: {exc}")

    if st.session_state.get("live_result"):
        st.text(st.session_state["live_result"])
    if st.session_state.get("live_video_bytes"):
        st.video(st.session_state["live_video_bytes"])

with tab_compare:
    st.subheader("Model comparison and evaluation")
    st.write(
        "Statistics are calculated from the existing evaluation CSV files "
        "in the `results` folder."
    )
    if st.button("Refresh Evaluation Results", key="refresh_eval"):
        st.session_state["evaluation_loaded"] = True

    # Show the evaluation by default and when the refresh button is clicked.
    if st.session_state.get("evaluation_loaded", True):
        try:
            message, reward_plot, steps_plot, success_plot = refresh_evaluation()
            st.markdown(message)
            chart_col1, chart_col2 = st.columns(2)
            if reward_plot is not None:
                chart_col1.pyplot(reward_plot, clear_figure=True)
            if steps_plot is not None:
                chart_col2.pyplot(steps_plot, clear_figure=True)
            if success_plot is not None:
                st.pyplot(success_plot, clear_figure=True)
        except Exception as exc:
            st.warning(f"Could not load evaluation results: {exc}")
            st.info(
                "Check that `results/summary.csv` or "
                "`results/all_evaluation_results.csv` exists and has the "
                "expected columns."
            )

with tab_history:
    st.subheader("Training history")
    st.write(
        "Original training logs show the learning curve. If they are "
        "unavailable, existing evaluation episode rewards are shown "
        "separately and are not presented as training history."
    )
    if st.button("Load Available History", key="load_history"):
        st.session_state["history_loaded"] = True

    if st.session_state.get("history_loaded", True):
        try:
            history_message_text, history_plot = training_history()
            st.markdown(history_message_text)
            if history_plot is not None:
                st.pyplot(history_plot, clear_figure=True)
        except Exception as exc:
            st.warning(f"Could not load history: {exc}")

with tab_videos:
    st.subheader("Saved demonstration videos")
    st.write("These are previously generated videos from your `results` folder.")
    video_col1, video_col2 = st.columns(2)

    dqn_video_path = existing_video("dqn_lander.mp4")
    reinforce_video_path = existing_video("reinforce_lander.mp4")

    with video_col1:
        st.markdown("**DQN demonstration**")
        if dqn_video_path:
            st.video(dqn_video_path)
        else:
            st.info("`results/dqn_lander.mp4` was not found.")

    with video_col2:
        st.markdown("**REINFORCE demonstration**")
        if reinforce_video_path:
            st.video(reinforce_video_path)
        else:
            st.info("`results/reinforce_lander.mp4` was not found.")

    st.caption(
        "If a saved video is missing, run `python run_models.py` locally to "
        "recreate evaluation results and recordings. This does not retrain "
        "the saved models."
    )
