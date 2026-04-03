import os
from dotenv import load_dotenv
from huggingface_hub import create_repo, upload_folder
import torch
import json
from tokenizers import Tokenizer
from src.model.model import build_transformer
# Load token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Your repo name (must be unique)
REPO_ID = "your-username/transformer-translation-custom"

def upload_to_hf():
    create_repo(repo_id=REPO_ID, token=HF_TOKEN, exist_ok=True)

    # Step 2: Upload entire folder
    upload_folder(
        folder_path="transformer_artifact",
        repo_id=REPO_ID,
        token=HF_TOKEN
    )

    print("Upload successful")


def load_from_hf():
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id="your-username/transformer-translation-custom"
    )

    print(local_path)


def load_model():


    # Paths
    BASE_PATH = "downloaded_folder_path"

    # Load metadata
    with open(f"{BASE_PATH}/config/model_metadata.json") as f:
        metadata = json.load(f)

    config = metadata["config"]

    # Load tokenizers
    tokenizer_src = Tokenizer.from_file(f"{BASE_PATH}/tokenizers/tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file(f"{BASE_PATH}/tokenizers/tokenizer_it.json")

    # Build model
    model = build_transformer(
        metadata["src_vocab_size"],
        metadata["tgt_vocab_size"],
        config["seq_len"],
        config["seq_len"],
        config["d_model"]
    )

    # Load weights
    state_dict = torch.load(f"{BASE_PATH}/weights/final_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()