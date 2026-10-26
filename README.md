# LunarLander RL Lab

An interactive Deep Reinforcement Learning application featuring **DQN (Deep Q-Network)** and **REINFORCE** agents trained on Gymnasium's `LunarLander-v3` environment. Built with Python, PyTorch, and Streamlit, this project allows users to explore trained reinforcement learning agents, run simulations, and examine evaluation results.

## Live Demo

**[Launch LunarLander RL Lab](https://lunar-lander-rl-lab-ocesbugdun9mfrzvpf5eak.streamlit.app/)**

Explore the deployed application to interact with the trained agents and visualize their performance.

## Features

- **DQN Agent:** Explore a trained value-based reinforcement learning policy.
- **REINFORCE Agent:** Explore a policy-gradient reinforcement learning approach.
- **Interactive Dashboard:** Use the Streamlit interface to run simulations and explore evaluation results.
- **Pretrained Models:** Load saved DQN and REINFORCE checkpoints.
- **Model Comparison:** Examine evaluation results for both agents.
- **Training History:** Visualize available training metrics.
- **Recorded Demonstrations:** Watch saved agent evaluation videos.
- **Cloud Deployment:** Access the application through Streamlit Community Cloud.

## Tech Stack

- Python
- PyTorch
- Gymnasium
- Box2D
- Streamlit
- NumPy
- Pandas
- Matplotlib

## Project Structure

```text
lunar-lander-rl-lab/
├── streamlit_app.py
├── app.py
├── run_models.py
├── dqn_lander.pt
├── reinforce_lander.pt
├── requirements.txt
├── packages.txt
├── README.md
├── LICENSE
└── results/
    ├── all_evaluation_results.csv
    ├── dqn_evaluation.csv
    ├── reinforce_evaluation.csv
    ├── summary.csv
    ├── dqn_lander.mp4
    └── reinforce_lander.mp4
```

*Note: The listed results files should be present in the repository for the corresponding evaluation and video features to work. The application entry point for the deployed Streamlit dashboard is `streamlit_app.py`.*

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/chillakalyan/lunar-lander-rl-lab.git
cd lunar-lander-rl-lab
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that the dependencies required by Gymnasium's LunarLander environment, including Box2D, are installed.

### 4. Launch the Streamlit dashboard

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL printed in the terminal to access the application.

## Evaluate the Trained Agents

To run the project's evaluation script:

```bash
python run_models.py
```

The script's actual evaluation settings determine which agents are evaluated and which reports or recordings are generated. Review `run_models.py` before running it, particularly if you want to reproduce the reported results.

## Evaluation Results

The `results/` directory can contain the following evaluation artifacts:

| File | Description |
|---|---|
| `all_evaluation_results.csv` | Combined evaluation results |
| `dqn_evaluation.csv` | DQN evaluation data |
| `reinforce_evaluation.csv` | REINFORCE evaluation data |
| `summary.csv` | Summary of evaluation results |
| `dqn_lander.mp4` | Recorded DQN demonstration |
| `reinforce_lander.mp4` | Recorded REINFORCE demonstration |

These artifacts help compare the agents under the configured evaluation conditions. Results may vary with random seeds, environment settings, and evaluation procedures.

## Deployment with Streamlit Community Cloud

The application is deployed using Streamlit Community Cloud.

**Live application:** [LunarLander RL Lab](https://lunar-lander-rl-lab-ocesbugdun9mfrzvpf5eak.streamlit.app/)

To deploy your own instance:

1. Push the application code and required files to GitHub.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Create a new application and select `chillakalyan/lunar-lander-rl-lab`.
5. Select the `main` branch.
6. Set the main file path to `streamlit_app.py`.
7. Deploy and wait for the application to build.

### Deployment considerations

- Include Streamlit and the application's required Python packages in `requirements.txt`.
- Configure system dependencies for Box2D in `packages.txt` if needed.
- Ensure that required model checkpoints are available at runtime.
- Include the evaluation data and demonstration videos required by the interface.
- Consider the memory, CPU, and execution-time limits of free hosting.
- Never commit virtual environments, secrets, or credentials.

## Reinforcement Learning Environment

This project uses Gymnasium's `LunarLander-v3` environment for its reinforcement learning experiments. DQN and REINFORCE represent different approaches to learning a policy.

Results from this environment should not be directly compared with results from simplified browser-based LunarLander simulations unless their environment dynamics, observation spaces, rewards, and evaluation procedures are equivalent.

## License

See [LICENSE](LICENSE) for licensing information.

## Author

**Chilla Kalyan Sai Kumar Reddy**

- **GitHub:** [@chillakalyan](https://github.com/chillakalyan)
- **Repository:** [LunarLander RL Lab](https://github.com/chillakalyan/lunar-lander-rl-lab)
- **Live Demo:** [Open the application](https://lunar-lander-rl-lab-ocesbugdun9mfrzvpf5eak.streamlit.app/)
