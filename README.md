# LunarLander-v3 Gradio app

## Files
- `app.py`: Gradio interface for the real Gymnasium LunarLander-v3 environment.
- `dqn_lander.pt`: trained DQN checkpoint from your notebook.
- `reinforce_lander.pt`: trained REINFORCE checkpoint from your notebook.
- `requirements.txt`: Python dependencies.
- `packages.txt`: system package required to build Box2D.

## Run locally
```bash
pip install -r requirements.txt
python app.py
```

## Publish on Hugging Face Spaces
1. Create a new Space and choose **Gradio** as the SDK.
2. Upload all five files in this folder to the Space repository.
3. Wait for the build to finish, then open the Space URL.

Note: the HTML browser demo uses a simplified JavaScript simulator. This Gradio app uses the actual Gymnasium `LunarLander-v3` environment, so the two environments' scores are not directly comparable.
