"""
IceBERT Old Icelandic Training - Version 3
Partial parameter freezing for efficient adaptation

Note: IceBERT uses RoBERTa architecture (not BERT), so we access model.roberta.*

Freezing strategy based on V1/V2 results:
- Embeddings: FROZEN (vocabulary mostly shared)
- Layers 1-4: FROZEN (basic morphological features)
- Layers 5-12: TRAINABLE (syntax, semantics, context)

Rationale: V1/V2 showed that substantial adaptation is needed (4.18% → 41.79% accuracy),
but the large vocabulary overlap between Modern and Old Icelandic means we can preserve
embeddings and low-level features while focusing training on higher-level linguistic patterns.

Expected: ~55-60% trainable parameters (middle ground between PEFT and full fine-tuning)

Fixes for NaN evaluation:
- Gradient checkpointing disabled (conflicts with frozen layers)
- Eval batch size increased to 4 (handles short sequences better)
- Short sequences filtered (< 10 tokens cause NaN with masking)
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
# CONFIG
# ============================================================================

DATA_PATH = './dataset_cleaned.txt'
OUTPUT_DIR = './icebert-old-icelandic-v3-partial-freeze'

# Training parameters - same as V2
BATCH_SIZE = 2
GRAD_ACCUMULATION = 8
LEARNING_RATE = 3e-5
EPOCHS = 8
MAX_LENGTH = 256
WARMUP_RATIO = 0.06

# Validation and checkpointing
VAL_SPLIT = 0.10
EVAL_STEPS = 300
SAVE_STEPS = 300
LOGGING_STEPS = 50

# Early stopping
EARLY_STOPPING_PATIENCE = 0  # DISABLED
EARLY_STOPPING_THRESHOLD = 0.001

# MLM parameters
MLM_PROBABILITY = 0.15

# ============================================================================
# FREEZING CONFIGURATION
# ============================================================================

# Based on V1/V2 results showing substantial improvement (4.18% → 41.79%),
# we freeze lower layers while allowing upper layers to adapt

FREEZE_EMBEDDINGS = True      # Freeze token/position embeddings
FREEZE_LAYERS = [0, 1, 2, 3]  # Freeze first 4 layers (0-indexed)
# Trainable layers: 4-11 (8 out of 12 layers)

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

def freeze_parameters(model, freeze_embeddings=True, freeze_layers=None):
    """
    Freeze specific parameters in the model
    
    Args:
        model: The model to freeze parameters in
        freeze_embeddings: Whether to freeze embedding layers
        freeze_layers: List of layer indices to freeze (0-indexed)
    
    Returns:
        Tuple of (trainable_params, total_params, percentage)
    """
    if freeze_layers is None:
        freeze_layers = []
    
    # Freeze embeddings (IceBERT uses RoBERTa architecture)
    if freeze_embeddings:
        logger.info("\nFreezing embeddings layer...")
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        logger.info("  ✓ Embeddings frozen")
    
    # Freeze specific transformer layers (RoBERTa structure)
    if freeze_layers:
        logger.info(f"\nFreezing transformer layers: {freeze_layers}")
        for layer_idx in freeze_layers:
            for param in model.roberta.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
            logger.info(f"  ✓ Layer {layer_idx} frozen")
    
    # Count parameters
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / total_params
    
    logger.info("\n" + "="*70)
    logger.info("Parameter Freezing Summary:")
    logger.info("="*70)
    logger.info(f"  Total parameters:      {total_params:,}")
    logger.info(f"  Trainable parameters:  {trainable_params:,}")
    logger.info(f"  Frozen parameters:     {total_params - trainable_params:,}")
    logger.info(f"  Trainable percentage:  {percentage:.2f}%")
    logger.info("="*70)
    
    # Show layer-by-layer breakdown
    logger.info("\nLayer-by-layer status:")
    logger.info("  Embeddings: " + ("FROZEN ❄️" if freeze_embeddings else "TRAINABLE ✓"))
    for i in range(12):  # RoBERTa (IceBERT) has 12 layers
        if i in freeze_layers:
            logger.info(f"  Layer {i:2d}:    FROZEN ❄️")
        else:
            logger.info(f"  Layer {i:2d}:    TRAINABLE ✓")
    logger.info("  MLM Head:   TRAINABLE ✓")
    logger.info("")
    
    return trainable_params, total_params, percentage

def main():
    logger.info("="*70)
    logger.info("IceBERT → Old Icelandic Training (Version 3)")
    logger.info("Partial Parameter Freezing Strategy")
    logger.info("="*70)
    
    logger.info("\nV3 Strategy:")
    logger.info("  • Freeze embeddings (vocabulary mostly shared)")
    logger.info("  • Freeze layers 0-3 (basic morphological features)")
    logger.info("  • Train layers 4-11 (syntax, semantics, context)")
    logger.info("  • Expected: ~55-60% trainable parameters")
    logger.info("")
    
    logger.info("Rationale from V1/V2 results:")
    logger.info("  • Original → V2: 4.18% → 41.79% accuracy (+900%)")
    logger.info("  • Pseudo-perplexity: 36,720 → 29.30 (99.9% improvement)")
    logger.info("  • Shows substantial adaptation needed at high levels")
    logger.info("  • But vocabulary overlap suggests low levels can be frozen")
    logger.info("")
    
    # Force CPU mode
    logger.info("✓ Running in CPU mode")
    
    # Load model
    logger.info("\n[1/5] Loading IceBERT...")
    tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")
    model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")
    # Gradient checkpointing disabled - conflicts with frozen layers causing NaN
    # model.gradient_checkpointing_enable()
    model = model.to("cpu")
    logger.info("✓ Model loaded")
    
    # Apply parameter freezing
    logger.info("\n[2/5] Applying parameter freezing...")
    trainable, total, percent = freeze_parameters(
        model, 
        freeze_embeddings=FREEZE_EMBEDDINGS,
        freeze_layers=FREEZE_LAYERS
    )
    
    if percent < 40:
        logger.warning("⚠ Warning: Less than 40% trainable - might underfit")
    elif percent > 80:
        logger.warning("⚠ Warning: More than 80% trainable - close to full fine-tuning")
    else:
        logger.info("✓ Freezing configuration looks good")
    
    # Load data
    logger.info("\n[3/5] Loading Old Icelandic corpus...")
    try:
        texts = load_data(DATA_PATH)
        
        if len(texts) < 10000:
            logger.warning(f"⚠ Only {len(texts):,} sentences found")
            logger.warning("  Recommend at least 17,000 sentences")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    except FileNotFoundError:
        logger.error(f"❌ File not found: {DATA_PATH}")
        return
    
    # Prepare dataset
    logger.info("\n[4/5] Preparing dataset...")
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Filter out very short sequences (cause NaN during evaluation)
    logger.info("\nFiltering short sequences...")
    original_count = len(tokenized)
    
    def filter_short(example):
        # Keep sequences with at least 10 tokens (after tokenization)
        # Short sequences can cause NaN when 15% masking leaves 0-1 tokens
        return len(example['input_ids']) >= 10
    
    tokenized = tokenized.filter(filter_short)
    filtered_count = len(tokenized)
    removed = original_count - filtered_count
    
    logger.info(f"✓ Filtered: {filtered_count:,} kept, {removed:,} removed ({removed/original_count*100:.1f}%)")
    
    # Split train/validation
    split = tokenized.train_test_split(test_size=VAL_SPLIT, seed=42)
    train_data = split["train"]
    eval_data = split["test"]
    
    logger.info(f"✓ Train: {len(train_data):,} | Validation: {len(eval_data):,}")
    
    # Save splits
    logger.info("\nSaving train/validation splits...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_texts = []
    for i in range(len(train_data)):
        decoded = tokenizer.decode(train_data[i]['input_ids'], skip_special_tokens=True)
        train_texts.append(decoded)
    
    train_split_file = os.path.join(OUTPUT_DIR, 'train_split.txt')
    with open(train_split_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_texts))
    logger.info(f"  ✓ Training set: {train_split_file}")
    
    eval_texts = []
    for i in range(len(eval_data)):
        decoded = tokenizer.decode(eval_data[i]['input_ids'], skip_special_tokens=True)
        eval_texts.append(decoded)
    
    val_split_file = os.path.join(OUTPUT_DIR, 'validation_split.txt')
    with open(val_split_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(eval_texts))
    logger.info(f"  ✓ Validation set: {val_split_file}")
    
    # Save configuration info
    stats_file = os.path.join(OUTPUT_DIR, 'training_info.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"V3 Training Configuration - Partial Parameter Freezing\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Data:\n")
        f.write(f"  Total sentences: {len(texts):,}\n")
        f.write(f"  Training: {len(train_texts):,} ({len(train_texts)/len(texts)*100:.1f}%)\n")
        f.write(f"  Validation: {len(eval_texts):,} ({len(eval_texts)/len(texts)*100:.1f}%)\n\n")
        f.write(f"Parameter Freezing:\n")
        f.write(f"  Total parameters: {total:,}\n")
        f.write(f"  Trainable: {trainable:,} ({percent:.2f}%)\n")
        f.write(f"  Frozen: {total - trainable:,} ({100-percent:.2f}%)\n\n")
        f.write(f"Frozen components:\n")
        f.write(f"  - Embeddings: {'Yes' if FREEZE_EMBEDDINGS else 'No'}\n")
        f.write(f"  - Layers: {FREEZE_LAYERS}\n\n")
        f.write(f"Trainable components:\n")
        f.write(f"  - Layers: {[i for i in range(12) if i not in FREEZE_LAYERS]}\n")
        f.write(f"  - MLM Head: Yes\n\n")
        f.write(f"Rationale:\n")
        f.write(f"  Based on V1/V2 results showing 4.18% → 41.79% accuracy improvement,\n")
        f.write(f"  substantial adaptation is needed. However, vocabulary overlap between\n")
        f.write(f"  Modern and Old Icelandic allows freezing embeddings and lower layers\n")
        f.write(f"  while training upper layers for syntactic/semantic adaptation.\n")
    logger.info(f"  ✓ Config info: {stats_file}\n")
    
    # Setup training
    logger.info("\n[5/5] Setting up training...")
    
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
    logger.info(f"  Parameters being updated: {trainable:,} ({percent:.2f}%)")
    
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
        max_grad_norm=1.0,
        
        # Learning rate schedule
        lr_scheduler_type="cosine",
        
        # Evaluation
        per_device_eval_batch_size=4,  # Larger batch to handle short sequences
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
        dataloader_pin_memory=False,
        
        # Misc
        seed=42,
        report_to="none",
        disable_tqdm=False,
    )
    
    # Setup callbacks
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
    logger.info("Starting V3 training with partial parameter freezing...")
    logger.info("="*70)
    logger.info(f"Training {percent:.2f}% of parameters ({trainable:,} / {total:,})")
    logger.info(f"Estimated time: {total_steps * 4 / 60:.0f}-{total_steps * 8 / 60:.0f} minutes")
    logger.info("(Faster than full fine-tuning due to fewer parameters)")
    logger.info("="*70 + "\n")
    
    try:
        trainer.train()
        
        # Save
        logger.info("\n✓ Training complete! Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        logger.info(f"✓ Model saved to: {OUTPUT_DIR}")
        
        # Final evaluation
        logger.info("\nFinal validation metrics:")
        metrics = trainer.evaluate()
        logger.info(f"  Validation Loss: {metrics['eval_loss']:.4f}")
        
        # Compare to V2 if we can
        logger.info("\nExpected comparison to V2:")
        logger.info("  V2 (100% params): 41.79% accuracy, 29.30 pseudo-perplexity")
        logger.info(f"  V3 ({percent:.1f}% params): [evaluate to find out]")
        logger.info("")
        logger.info("  Hypothesis: V3 should achieve 38-40% accuracy")
        logger.info("  (slightly lower than V2, but with 40-45% fewer parameters trained)")
        
        logger.info("\n" + "="*70)
        logger.info("NEXT STEPS")
        logger.info("="*70)
        logger.info("\n1. Evaluate V3 with deep_eval.py")
        logger.info("   Compare: Original vs V1 vs V2 vs V3")
        logger.info("")
        logger.info("2. Analyze trade-offs:")
        logger.info("   • V2: 100% params, best accuracy (41.79%)")
        logger.info(f"   • V3: {percent:.1f}% params, [check accuracy]")
        logger.info("   • Trade-off: Efficiency vs performance")
        logger.info("")
        logger.info("3. Use in Binder NER:")
        logger.info(f"   encoder = AutoModel.from_pretrained('{OUTPUT_DIR}')")
        logger.info("")
        logger.info("4. For thesis:")
        logger.info("   'We explored partial parameter freezing (V3) as a middle")
        logger.info("   ground between full fine-tuning and PEFT, freezing")
        logger.info(f"   embeddings and lower layers while training {percent:.0f}% of")
        logger.info("   parameters for syntactic/semantic adaptation.'")
        logger.info("="*70)
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Training interrupted - saving checkpoint...")
        trainer.save_model(OUTPUT_DIR)
        logger.info("✓ Checkpoint saved")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        if "out of memory" in str(e).lower():
            logger.info("\n💡 Memory solutions:")
            logger.info("  BATCH_SIZE = 1")
            logger.info("  MAX_LENGTH = 128")
        raise

if __name__ == "__main__":
    main()