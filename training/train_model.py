#!/usr/bin/env python3
"""
Model training and fine-tuning script.
Implements automatic retraining with periodic validation.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = Path("models")
METRICS_FILE = Path("metrics/training_metrics.json")
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 2


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }


def load_training_data() -> DatasetDict:
    """Load sentiment dataset for fine-tuning."""
    try:
        # Try to load a public sentiment dataset (Twitter sentiment 2015)
        logger.info("Loading sentiment dataset...")
        dataset = load_dataset(
            "tweet_eval",
            "sentiment",
            split="train"
        )
        
        # Use only a small subset for quick fine-tuning (production-like scenario)
        dataset = dataset.shuffle(seed=42).select(range(min(1000, len(dataset))))
        
        # Split into train/validation
        dataset = dataset.train_test_split(
            test_size=VALIDATION_SPLIT,
            seed=42
        )
        
        logger.info(f"Dataset loaded: {len(dataset['train'])} training samples, {len(dataset['test'])} validation samples")
        return dataset
    except Exception as e:
        logger.warning(f"Could not load remote dataset: {e}. Using synthetic data.")
        # Fallback: create synthetic training data for demonstration
        return create_synthetic_dataset()


def create_synthetic_dataset() -> DatasetDict:
    """Create a synthetic sentiment dataset for demonstration."""
    from datasets import Dataset
    
    synthetic_data = {
        "text": [
            "I love this product, it's amazing!",
            "This is terrible, worst experience ever",
            "It's okay, nothing special",
            "Great quality and fast shipping",
            "Horrible service, never coming back",
            "Exactly what I was looking for",
            "Not worth the money",
            "Highly recommend to everyone",
            "Could be better",
            "Absolutely terrible, stay away",
        ] * 100,  # Repeat to get reasonable dataset size
        "label": [2, 0, 1, 2, 0, 2, 0, 2, 1, 0] * 100,  # 0=negative, 1=neutral, 2=positive
    }
    
    dataset = Dataset.from_dict(synthetic_data)
    split_dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=42)
    logger.info(f"Synthetic dataset created: {len(split_dataset['train'])} training, {len(split_dataset['test'])} validation")
    return split_dataset


def tokenize_function(examples, tokenizer):
    """Tokenize text inputs."""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def train_model(resume_from_checkpoint: bool = False) -> Tuple[str, Dict]:
    """
    Fine-tune the sentiment model with validation.
    
    Args:
        resume_from_checkpoint: Whether to resume from a previous checkpoint
        
    Returns:
        Tuple of (model_path, metrics_dict)
    """
    logger.info(f"Starting model training/fine-tuning...")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )
    
    # Load and prepare dataset
    dataset = load_training_data()
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    # Rename label column if needed
    if "label" not in tokenized_datasets["train"].column_names:
        tokenized_datasets = tokenized_datasets.rename_column("sentiment", "label")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoint"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=42,
        report_to=["tensorboard"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE
            )
        ],
    )
    
    # Train
    logger.info("Training starting...")
    checkpoint = None
    if resume_from_checkpoint:
        latest_checkpoint = max(
            (OUTPUT_DIR / "checkpoint").glob("checkpoint-*"),
            default=None,
            key=lambda p: int(p.name.split("-")[1]) if p.name != "best_model" else -1
        )
        if latest_checkpoint:
            checkpoint = str(latest_checkpoint)
            logger.info(f"Resuming from checkpoint: {checkpoint}")
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Evaluate
    logger.info("Running final evaluation...")
    eval_result = trainer.evaluate()
    
    # Save model
    model_path = OUTPUT_DIR / f"model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(str(model_path))
    logger.info(f"Model saved to: {model_path}")
    
    # Compile metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "training_metrics": train_result.metrics,
        "evaluation_metrics": eval_result,
        "model_name": MODEL_NAME,
    }
    
    # Save metrics
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {METRICS_FILE}")
    
    return str(model_path), metrics


def validate_model(model_path: str) -> Dict[str, float]:
    """
    Validate a trained model on test data.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Validating model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    dataset = load_training_data()
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["text"]
    )
    
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )
    
    eval_result = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Validation metrics: {eval_result}")
    
    return eval_result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        if len(sys.argv) > 2:
            metrics = validate_model(sys.argv[2])
        else:
            latest_model = max(
                OUTPUT_DIR.glob("model-*"),
                default=None,
                key=lambda p: p.name
            )
            if latest_model:
                metrics = validate_model(str(latest_model))
            else:
                logger.error("No model found to validate")
                sys.exit(1)
    else:
        resume = len(sys.argv) > 1 and sys.argv[1] == "resume"
        model_path, metrics = train_model(resume_from_checkpoint=resume)
        logger.info(f"Training completed. Model: {model_path}")
        logger.info(f"Final metrics: {json.dumps(metrics, indent=2)}")