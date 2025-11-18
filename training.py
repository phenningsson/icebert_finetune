"""
IceBERT Old Icelandic Training
Clean version for Python 3.10.11 + M2 MacBook Air
CPU-ONLY MODE - No GPU/MPS usage
"""

import os
# DISABLE MPS COMPLETELY - Force CPU only
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

import torch
# Verify MPS is disabled
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)
from datasets import Dataset
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
set_seed(42)

# ============================================================================
# CONFIG - Change these settings
# ============================================================================

DATA_PATH = 'kombinerad.txt'  # Your Old Icelandic text file
OUTPUT_DIR = './icebert-old-icelandic'

# Memory-optimized for M2 MacBook Air - CPU ONLY
BATCH_SIZE = 2  # Larger on CPU since we have more RAM available
GRAD_ACCUMULATION = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
MAX_LENGTH = 128  # Reduced to save memory

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
    
    logger.info(f"✓ Loaded {len(texts)} sentences")
    return texts

def main():
    logger.info("="*70)
    logger.info("IceBERT → Old Icelandic Finetuning")
    logger.info("CPU-ONLY MODE")
    logger.info("="*70)
    
    logger.info("✓ FORCED CPU MODE (no GPU)")
    logger.info("  Slower but stable - no memory crashes")
    
    # Load model
    logger.info("\n[1/4] Loading IceBERT...")
    tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")
    model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")
    model.gradient_checkpointing_enable()  # Save memory
    model = model.to("cpu")  # FORCE CPU
    logger.info("✓ Model loaded on CPU")
    
    # Load data
    logger.info("\n[2/4] Loading Old Icelandic text...")
    try:
        texts = load_data(DATA_PATH)
    except FileNotFoundError:
        logger.error(f"❌ File not found: {DATA_PATH}")
        logger.info("\nCreate a text file with your Old Icelandic text (one sentence per line)")
        return
    
    # Prepare dataset
    logger.info("\n[3/4] Preparing dataset...")
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Split
    if len(tokenized) > 10:
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        train_data = split["train"]
        eval_data = split["test"]
        logger.info(f"✓ Train: {len(train_data)} | Eval: {len(eval_data)}")
    else:
        train_data = tokenized
        eval_data = None
        logger.info(f"✓ Train: {len(train_data)}")
    
    # Setup training
    logger.info("\n[4/4] Training...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=500 if eval_data else None,
        fp16=False,
        use_cpu=True,  # FORCE CPU
        dataloader_num_workers=0,
        seed=42,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("="*70)
    steps = len(train_data) // (BATCH_SIZE * GRAD_ACCUMULATION)
    logger.info(f"Total steps: {steps * EPOCHS}")
    logger.info(f"Estimated time on CPU: {steps * EPOCHS * 8 / 60:.0f}-{steps * EPOCHS * 12 / 60:.0f} minutes")
    logger.info("="*70 + "\n")
    
    try:
        trainer.train()
        
        # Save
        logger.info("\n✓ Training complete! Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Final eval
        if eval_data:
            metrics = trainer.evaluate()
            logger.info(f"\nFinal Loss: {metrics['eval_loss']:.4f}")
        
        logger.info("\n" + "="*70)
        logger.info(f"✓ Model saved to: {OUTPUT_DIR}")
        logger.info("\nTo use in Binder NER:")
        logger.info(f"  from transformers import AutoModel")
        logger.info(f"  encoder = AutoModel.from_pretrained('{OUTPUT_DIR}')")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Training interrupted - saving checkpoint...")
        trainer.save_model(OUTPUT_DIR)
        logger.info("✓ Saved")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        if "out of memory" in str(e).lower():
            logger.info("\n💡 Further reduce memory:")
            logger.info("  BATCH_SIZE = 1")
            logger.info("  MAX_LENGTH = 64")
        raise

if __name__ == "__main__":
    main()