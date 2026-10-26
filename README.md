# LunarLander RL Lab

An interactive Deep Reinforcement Learning application featuring **DQN (Deep Q-Network)** and **REINFORCE** agents trained on Gymnasium's `LunarLander-v3` environment. Built with Python, PyTorch, and Streamlit, this project provides an interface to explore trained agents and examine their evaluation results.

## Features

- **DQN agent:** Explore a trained value-based reinforcement learning policy.
- **REINFORCE agent:** Explore a policy-gradient reinforcement learning approach.
- **Interactive web interface:** Built with Streamlit.
- **Pretrained checkpoints:** Load the saved DQN and REINFORCE models.
- **Evaluation reports:** Review CSV files containing model evaluation results.
- **Recorded demonstrations:** Watch the saved agent evaluation videos.
- **Free deployment option:** Host the application on Streamlit Community Cloud.

## Tech Stack

- Python
- PyTorch
- Gymnasium
- Box2D
- Streamlit
- NumPy and Pandas

## Project Structure

```text
lunar-lander-rl-demo/
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

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/chillakalyan/lunar-lander-rl-demo.git
cd lunar-lander-rl-demo
```

### 2. Create a virtual environment

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

Ensure the Box2D dependencies required by Gymnasium's LunarLander environment are installed.

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal. Open that URL in your browser to use the application.

## Evaluate the Trained Agents

Run the evaluation script:

```bash
python run_models.py
```

The script runs the evaluations configured in the project and saves the resulting reports and recordings in the `results/` directory. Check `run_models.py` for the evaluation settings and supported options.

## Evaluation Results

The `results/` directory contains the following artifacts:

| File | Description |
|---|---|
| `all_evaluation_results.csv` | Combined evaluation results |
| `dqn_evaluation.csv` | DQN evaluation data |
| `reinforce_evaluation.csv` | REINFORCE evaluation data |
| `summary.csv` | Summary of evaluation results |
| `dqn_lander.mp4` | Recorded DQN evaluation |
| `reinforce_lander.mp4` | Recorded REINFORCE evaluation |

Use these results to compare the agents under the recorded evaluation conditions. Performance can vary depending on the environment configuration, random seeds, and evaluation procedure.

## Deploy for Free with Streamlit Community Cloud

1. Push the application code and required files to your GitHub repository.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account and authorize access to the repository.
4. Create a new app and select `chillakalyan/lunar-lander-rl-demo`.
5. Set the branch to `main` and the main file path to `app.py`.
6. Deploy the application and wait for the build to complete.
7. Open the generated public URL and add it to this README.

**Deployment considerations:**
- Ensure `requirements.txt` includes Streamlit and all Python packages imported by the app.
- Check whether additional operating-system dependencies are needed for Box2D.
- Include the trained model checkpoints if the app loads them at runtime.
- Keep resource limits in mind when running simulations or loading PyTorch models on free hosting.
- Do not commit API keys, credentials, or virtual-environment folders.

## Important Note

This project uses Gymnasium's actual `LunarLander-v3` environment for its reinforcement learning experiments. If you compare it with a separate HTML browser demo that uses a simplified JavaScript simulator, their scores are not directly comparable because the environments and dynamics differ.

## License

See the [LICENSE](LICENSE) file for licensing information.

## Author

**Chilla Kalyan Sai Kumar Reddy**

- GitHub: [@chillakalyan](https://github.com/chillakalyan)
- Project repository: [LunarLander RL Lab](https://github.com/chillakalyan/lunar-lander-rl-demo)
