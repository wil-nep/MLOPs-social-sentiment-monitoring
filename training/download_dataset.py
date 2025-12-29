#!/usr/bin/env python3
"""
Script to download and prepare the TweetEval sentiment dataset for fine-tuning.
"""

from datasets import load_dataset
import os

def download_dataset():
    # Load the TweetEval sentiment dataset
    dataset = load_dataset("tweet_eval", "sentiment")
    print("Dataset loaded successfully.")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")
    print(f"Test size: {len(dataset['test'])}")

    # Save to local directory
    output_dir = "training/data"
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)
    print(f"Dataset saved to {output_dir}")

if __name__ == "__main__":
    download_dataset()