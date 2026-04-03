from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_latest_checkpoint(model_folder="models", model_basename="tmodel_"):
    model_path = Path(model_folder)
    if not model_path.exists():
        return None

    checkpoint_files = sorted(model_path.glob(f"{model_basename}*.pt"))
    if not checkpoint_files:
        return None

    latest_file = checkpoint_files[-1]
    epoch_str = latest_file.stem.replace(model_basename, "")
    return epoch_str

def get_config():
    """Returns only path and training configuration.
    Model architecture is loaded from config.json for HuggingFace compatibility.
    """
    return {
        # Training parameters
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "lang_src": "en",
        "lang_tgt": "it",
        # Path configuration
        "model_folder": "models",
        "model_basename": "tmodel_",
        "preload": get_latest_checkpoint(),
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "config_file": "config/config.json"
    }

def get_weights_file_path(config,epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)