import os
import tempfile

import gradio as gr
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# The uploaded notebook checkpoints use this exact 8 -> 128 -> 128 -> 4 layout.
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
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Missing {filename}. Upload it beside app.py in your Hugging Face Space."
        )
    model = Net()
    state = torch.load(filename, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


dqn = load_checkpoint("dqn_lander.pt")
reinforce = load_checkpoint("reinforce_lander.pt")


def fly(agent_name, seed, noise_sigma, max_steps):
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=int(seed))
    frames = []
    total_reward = 0.0
    terminated = False
    truncated = False
    last_reward = 0.0
    steps = 0

    try:
        frames.append(env.render())
        while not (terminated or truncated) and steps < int(max_steps):
            observed = np.asarray(obs, dtype=np.float32).copy()
            if float(noise_sigma) > 0:
                observed += np.random.normal(
                    0.0, float(noise_sigma), observed.shape
                ).astype(np.float32)

            state = torch.as_tensor(observed, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                if agent_name == "Random":
                    action = env.action_space.sample()
                elif agent_name == "DQN (greedy)":
                    action = int(torch.argmax(dqn(state), dim=1).item())
                else:
                    logits = reinforce(state)
                    action = int(Categorical(logits=logits).sample().item())

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            last_reward = float(reward)
            steps += 1
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    finally:
        env.close()

    # Save a playable MP4. A small fallback handles an unusually short episode.
    video_path = tempfile.mktemp(suffix=".mp4")
    if frames:
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
    else:
        raise RuntimeError("The environment did not return any video frames.")

    landed = bool(terminated and last_reward >= 100.0)
    status = "Landed successfully" if landed else "Did not land successfully"
    result = (
        f"Agent: {agent_name}\n"
        f"Total reward: {total_reward:.1f}\n"
        f"Steps: {steps}\n"
        f"Result: {status}\n"
        f"Seed: {int(seed)} | Sensor noise σ: {float(noise_sigma):.2f}"
    )
    return video_path, result


demo_description = """
This app runs the uploaded PyTorch checkpoints inside the real Gymnasium
`LunarLander-v3` environment. It is separate from the companion HTML demo's
simplified JavaScript physics simulator, so their scores are not directly
interchangeable.
"""

with gr.Blocks(title="LunarLander-v3: DQN vs REINFORCE") as demo:
    gr.Markdown("# 🚀 LunarLander-v3: DQN vs REINFORCE")
    gr.Markdown(demo_description)
    with gr.Row():
        agent = gr.Radio(
            ["Random", "DQN (greedy)", "REINFORCE (sampled)"],
            value="DQN (greedy)",
            label="Agent",
        )
        seed = gr.Number(value=42, precision=0, label="Random seed")
    with gr.Row():
        noise = gr.Slider(
            minimum=0.0, maximum=0.5, value=0.0, step=0.05,
            label="Sensor noise σ",
        )
        max_steps = gr.Slider(
            minimum=200, maximum=1500, value=1000, step=100,
            label="Maximum episode steps",
        )
    run = gr.Button("Run landing episode", variant="primary")
    video = gr.Video(label="Lunar Lander episode")
    result = gr.Textbox(label="Episode result", lines=5)
    run.click(
        fn=fly,
        inputs=[agent, seed, noise, max_steps],
        outputs=[video, result],
    )

if __name__ == "__main__":
    demo.launch()
