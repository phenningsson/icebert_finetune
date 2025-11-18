"""
IceBERT Old Icelandic Training - Version 2
Optimized for larger dataset (17,000+ sentences)
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
set_seed(42)

# ============================================================================
# CONFIG - OPTIMIZED FOR LARGER DATASET
# ============================================================================

DATA_PATH = './dataset_cleaned.txt'  # Use cleaned corpus
OUTPUT_DIR = './icebert-old-icelandic-v2'

# Training parameters - optimized for ~17k sentences
BATCH_SIZE = 2           # Keep at 2 for M2 Air stability
GRAD_ACCUMULATION = 8    # Effective batch size = 16
LEARNING_RATE = 3e-5     # Slightly lower for better convergence
EPOCHS = 8               # More epochs with more data
MAX_LENGTH = 256         # Good balance of context and memory
WARMUP_RATIO = 0.06      # 6% warmup (standard for BERT)

# Validation and checkpointing
VAL_SPLIT = 0.10         # 10% validation
EVAL_STEPS = 300         # Evaluate every 300 steps
SAVE_STEPS = 300         # Save every 300 steps
LOGGING_STEPS = 50       # Log every 50 steps

# Early stopping
EARLY_STOPPING_PATIENCE = 0  # DISABLED - train all epochs despite NaN eval
EARLY_STOPPING_THRESHOLD = 0.001

# Advanced options
USE_WHOLE_WORD_MASKING = False  # Disabled - not compatible with IceBERT tokenizer
MLM_PROBABILITY = 0.15          # Standard BERT masking rate

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_data(path):
    """Load Old Icelandic text"""
    path = Path(path)
    texts = []
    
    if path.is_file():
        with open(path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif path.is_dir():
        for txt_file in sorted(path.glob("**/*.txt")):
            with open(txt_file, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])
    else:
        raise FileNotFoundError(f"Not found: {path}")
    
    logger.info(f"✓ Loaded {len(texts):,} sentences")
    logger.info(f"  Total characters: {sum(len(t) for t in texts):,}")
    
    return texts

def main():
    logger.info("="*70)
    logger.info("IceBERT → Old Icelandic Training (Version 2)")
    logger.info("Optimized for larger datasets")
    logger.info("="*70)
    
    logger.info("\nImprovements in this version:")
    logger.info("  • Standard token masking (IceBERT compatible)")
    logger.info("  • Optimized learning rate (3e-5)")
    logger.info("  • 8 epochs with early stopping")
    logger.info("  • Better evaluation metrics")
    logger.info("")
    
    # Force CPU mode
    logger.info("✓ Running in CPU mode")
    
    # Load model
    logger.info("\n[1/4] Loading IceBERT...")
    tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")
    model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")
    model.gradient_checkpointing_enable()
    model = model.to("cpu")
    logger.info("✓ Model loaded")
    
    # Load data
    logger.info("\n[2/4] Loading Old Icelandic corpus...")
    try:
        texts = load_data(DATA_PATH)
        
        if len(texts) < 10000:
            logger.warning(f"⚠ Only {len(texts):,} sentences found")
            logger.warning("  Recommend collecting at least 17,000 sentences")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    except FileNotFoundError:
        logger.error(f"❌ File not found: {DATA_PATH}")
        logger.info("\nMake sure your expanded corpus is ready:")
        logger.info("  • Original: ~7,600 sentences")
        logger.info("  • New data: ~10,000+ sentences")
        logger.info("  • Total: ~17,000+ sentences")
        return
    
    # Prepare dataset
    logger.info("\n[3/4] Preparing dataset...")
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,  # Dynamic padding handled by collator
            return_attention_mask=True,
            return_special_tokens_mask=True
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Split train/validation
    split = tokenized.train_test_split(test_size=VAL_SPLIT, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    
    logger.info(f"✓ Train: {len(train_data):,} | Validation: {len(eval_data):,}")
    
    # Save the splits as text files for inspection
    logger.info("\nSaving train/validation splits...")
    
    # Get original text indices
    train_indices = train_data['input_ids']
    eval_indices = eval_data['input_ids']
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Decode and save training set
    train_texts = []
    for i in range(len(train_data)):
        # Decode the tokenized text back to string
        decoded = tokenizer.decode(train_data[i]['input_ids'], skip_special_tokens=True)
        train_texts.append(decoded)
    
    train_split_file = os.path.join(OUTPUT_DIR, 'train_split.txt')
    with open(train_split_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    logger.info(f"  ✓ Training set saved: {train_split_file}")
    
    # Decode and save validation set
    eval_texts = []
    for i in range(len(eval_data)):
        decoded = tokenizer.decode(eval_data[i]['input_ids'], skip_special_tokens=True)
        eval_texts.append(decoded)
    
    val_split_file = os.path.join(OUTPUT_DIR, 'validation_split.txt')
    with open(val_split_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_texts))
    logger.info(f"  ✓ Validation set saved: {val_split_file}")
    
    # Save split statistics
    stats_file = os.path.join(OUTPUT_DIR, 'split_info.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"Data Split Information\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Total sentences: {len(texts):,}\n")
        f.write(f"Training sentences: {len(train_texts):,} ({len(train_texts)/len(texts)*100:.1f}%)\n")
        f.write(f"Validation sentences: {len(eval_texts):,} ({len(eval_texts)/len(texts)*100:.1f}%)\n")
        f.write(f"\nRandom seed: 42\n")
        f.write(f"Split ratio: {1-VAL_SPLIT:.0%} train / {VAL_SPLIT:.0%} validation\n")
    logger.info(f"  ✓ Split info saved: {stats_file}")
    logger.info("")
    
    # Setup data collator with standard masking
    logger.info("\n[4/4] Setting up training...")
    
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_data) // (BATCH_SIZE * GRAD_ACCUMULATION)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    logger.info(f"  Steps per epoch: {steps_per_epoch}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # Training
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,  # Clip gradients to prevent NaN
        
        # Learning rate schedule (cosine decay)
        lr_scheduler_type="cosine",
        
        # Evaluation
        per_device_eval_batch_size=1,  # Smaller batch for stability
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        
        # Logging & Saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Optimization
        fp16=False,
        use_cpu=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,  # Prevent memory issues
        
        # Misc
        seed=42,
        report_to="none",
        disable_tqdm=False,
    )
    
    # Setup early stopping (if enabled)
    callbacks = []
    if EARLY_STOPPING_PATIENCE > 0:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=EARLY_STOPPING_THRESHOLD
        )
        callbacks.append(early_stopping)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    logger.info(f"Estimated time: {total_steps * 6 / 60:.0f}-{total_steps * 10 / 60:.0f} minutes")
    if EARLY_STOPPING_PATIENCE > 0:
        logger.info("Early stopping enabled - training may finish early")
    else:
        logger.info("Early stopping DISABLED - will train all 8 epochs")
    logger.info("="*70 + "\n")
    
    try:
        trainer.train()
        
        # Save
        logger.info("\n✓ Training complete! Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Final evaluation
        logger.info("\nFinal validation metrics:")
        metrics = trainer.evaluate()
        logger.info(f"  Validation Loss: {metrics['eval_loss']:.4f}")
        
        # Compare to initial loss if available
        if hasattr(trainer.state, 'log_history') and len(trainer.state.log_history) > 0:
            initial_logs = [log for log in trainer.state.log_history if 'eval_loss' in log]
            if len(initial_logs) > 1:
                initial_loss = initial_logs[0]['eval_loss']
                final_loss = metrics['eval_loss']
                improvement = ((initial_loss - final_loss) / initial_loss) * 100
                logger.info(f"  Initial Loss: {initial_loss:.4f}")
                logger.info(f"  Improvement: {improvement:.1f}%")
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ Model saved to: {OUTPUT_DIR}")
        logger.info("\nNext steps:")
        logger.info("  1. Evaluate on external test set:")
        logger.info("     python standardized_eval.py")
        logger.info("  2. Use in Binder NER:")
        logger.info(f"     encoder = AutoModel.from_pretrained('{OUTPUT_DIR}')")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Training interrupted - saving checkpoint...")
        trainer.save_model(OUTPUT_DIR)
        logger.info("✓ Saved")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        if "out of memory" in str(e).lower():
            logger.info("\n💡 Memory solutions:")
            logger.info("  BATCH_SIZE = 1")
            logger.info("  MAX_LENGTH = 128")
        raise

if __name__ == "__main__":
    main()