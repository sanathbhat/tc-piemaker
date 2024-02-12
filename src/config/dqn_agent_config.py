class PieMakerDQNConfig:
    # Environment settings
    ENV_NAME = 'PieMakerEnv-v0'

    # Agent settings
    hyper_parameters = {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "gradient_steps": -1
    }
