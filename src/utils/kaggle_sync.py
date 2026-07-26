import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

def sync_to_kaggle_dataset(dataset_slug, dataset_title, folder_to_sync="kaggle_sync", update_message="Updating weights"):
    """
    Syncs training artifacts to a Kaggle dataset.
    
    Args:
        dataset_slug (str): Your kaggle dataset slug (e.g., 'yourusername/trans-trans-weights')
        dataset_title (str): Title for the dataset (only used if creating new)
        folder_to_sync (str): Temporary folder to gather files before upload
        update_message (str): Message for the dataset version update
    """
    sync_dir = Path(folder_to_sync)
    
    # 1. Gather all required files into the sync directory
    print(f"Gathering files into {sync_dir}...")
    if sync_dir.exists():
        shutil.rmtree(sync_dir)
    sync_dir.mkdir(parents=True)

    # Directories/Files to save
    items_to_save = [
        "models",           # Model weights and metadata
        "runs",             # TensorBoard logs
        "config",           # Model configurations
        "tokenizer_en.json",
        "tokenizer_it.json"
    ]

    for item in items_to_save:
        item_path = Path(item)
        if not item_path.exists():
            print(f"Warning: {item} does not exist. Skipping.")
            continue
            
        dest_path = sync_dir / item
        if item_path.is_dir():
            # Use os.link to create hard links instead of copying data (saves disk space)
            shutil.copytree(item_path, dest_path, copy_function=os.link)
        else:
            os.link(item_path, dest_path)
            
    print("Files gathered successfully.")

    # 2. Check if Kaggle API is accessible
    try:
        import kaggle
    except (ImportError, OSError):
        print("Error: Kaggle API not configured properly.")
        print("Make sure you have kaggle installed (pip install kaggle) and your kaggle.json is in ~/.kaggle/")
        return

    # 3. Create dataset-metadata.json
    metadata = {
        "title": dataset_title,
        "id": dataset_slug,
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(sync_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # 4. Upload to Kaggle
    print(f"Uploading to Kaggle Dataset: {dataset_slug}...")
    
    # Check if dataset exists by trying to list its files
    try:
        subprocess.run(["kaggle", "datasets", "status", dataset_slug], check=True, capture_output=True)
        dataset_exists = True
    except subprocess.CalledProcessError:
        dataset_exists = False

    try:
        if dataset_exists:
            print("Dataset exists. Pushing new version...")
            subprocess.run(["kaggle", "datasets", "version", "-p", str(sync_dir), "-m", update_message, "--dir-mode", "zip"], check=True)
        else:
            print("Dataset does not exist. Creating new dataset...")
            subprocess.run(["kaggle", "datasets", "create", "-p", str(sync_dir), "--dir-mode", "zip"], check=True)
            
        print("Upload complete!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to upload to Kaggle: {e}")

if __name__ == "__main__":
    # REPLACE THIS WITH YOUR KAGGLE USERNAME AND DESIRED DATASET NAME
    KAGGLE_USERNAME = "thorfromasgard" 
    DATASET_NAME = "trans-trans-checkpoints"
    
    DATASET_SLUG = f"{KAGGLE_USERNAME}/{DATASET_NAME}"
    DATASET_TITLE = "Transformer Translation Checkpoints"
    
    sync_to_kaggle_dataset(DATASET_SLUG, DATASET_TITLE)
