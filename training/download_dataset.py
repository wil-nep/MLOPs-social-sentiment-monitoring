#!/usr/bin/env python3
"""
Dataset download and preparation script.
Handles downloading and caching sentiment analysis datasets.
"""

import os
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATASETS_DIR = Path("data")
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "datasets"


def download_tweet_eval_dataset(cache_dir: Optional[Path] = None) -> DatasetDict:
    """
    Download the Tweet Eval sentiment dataset.
    
    Args:
        cache_dir: Custom cache directory for datasets
        
    Returns:
        DatasetDict with train/test splits
    """
    logger.info("Downloading tweet_eval/sentiment dataset...")
    
    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    
    try:
        dataset = load_dataset(
            "tweet_eval",
            "sentiment",
            **cache_kwargs
        )
        logger.info(f"Dataset downloaded successfully")
        logger.info(f"Dataset splits: {list(dataset.keys())}")
        logger.info(f"Dataset info: {dataset}")
        
        return dataset
    except Exception as e:
        logger.error(f"Error downloading tweet_eval dataset: {e}")
        raise


def download_twitter_sentiment_2015(cache_dir: Optional[Path] = None) -> DatasetDict:
    """
    Download Twitter Sentiment 2015/2016 dataset.
    
    Args:
        cache_dir: Custom cache directory for datasets
        
    Returns:
        DatasetDict with train/test splits
    """
    logger.info("Downloading twitter-sentiment-analysis dataset...")
    
    cache_kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    
    try:
        dataset = load_dataset(
            "aesdd/twitter-sentiment-analysis",
            **cache_kwargs
        )
        logger.info(f"Dataset downloaded successfully")
        logger.info(f"Dataset info: {dataset}")
        
        return dataset
    except Exception as e:
        logger.error(f"Error downloading sentiment dataset: {e}")
        raise


def save_dataset_info(dataset: DatasetDict, output_dir: Path) -> None:
    """
    Save dataset information and statistics.
    
    Args:
        dataset: Dataset to analyze
        output_dir: Directory to save info
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    info_file = output_dir / "dataset_info.txt"
    
    with open(info_file, "w") as f:
        f.write("Dataset Information\n")
        f.write("=" * 50 + "\n\n")
        
        for split_name, split_data in dataset.items():
            f.write(f"Split: {split_name}\n")
            f.write(f"Number of samples: {len(split_data)}\n")
            f.write(f"Features: {split_data.features}\n")
            f.write(f"Column names: {split_data.column_names}\n")
            f.write("\n")
            
            # Sample statistics
            if "label" in split_data.column_names:
                labels = split_data["label"]
                f.write(f"Label distribution:\n")
                from collections import Counter
                label_counts = Counter(labels)
                for label, count in sorted(label_counts.items()):
                    f.write(f"  Label {label}: {count} ({100*count/len(labels):.1f}%)\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    logger.info(f"Dataset info saved to: {info_file}")


def prepare_datasets(
    output_dir: Path = DATASETS_DIR,
    cache_dir: Optional[Path] = None
) -> DatasetDict:
    """
    Download and prepare all datasets.
    
    Args:
        output_dir: Directory to save dataset info
        cache_dir: Custom cache directory
        
    Returns:
        Combined DatasetDict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting dataset download and preparation...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Cache directory: {cache_dir or CACHE_DIR}")
    
    try:
        # Try to download primary dataset
        dataset = download_tweet_eval_dataset(cache_dir)
        save_dataset_info(dataset, output_dir)
        
        logger.info("Dataset preparation completed successfully")
        return dataset
        
    except Exception as e:
        logger.warning(f"Primary dataset download failed: {e}")
        logger.info("Attempting alternative dataset...")
        
        try:
            dataset = download_twitter_sentiment_2015(cache_dir)
            save_dataset_info(dataset, output_dir)
            return dataset
        except Exception as e2:
            logger.error(f"Both dataset downloads failed: {e2}")
            raise


def verify_model_cache() -> bool:
    """
    Verify that the pretrained model is cached.
    
    Returns:
        True if model is cached, False otherwise
    """
    logger.info("Verifying pretrained model cache...")
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    try:
        logger.info(f"Attempting to load model: {MODEL_NAME}")
        # This will download the model if not cached
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        
        logger.info(f"Model cached and verified successfully")
        return True
    except Exception as e:
        logger.error(f"Error verifying model cache: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "model":
        # Verify and cache the pretrained model
        if verify_model_cache():
            logger.info("Model verification completed successfully")
        else:
            logger.error("Model verification failed")
            sys.exit(1)
    else:
        # Download and prepare datasets
        try:
            dataset = prepare_datasets()
            logger.info("All datasets prepared successfully")
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            sys.exit(1)