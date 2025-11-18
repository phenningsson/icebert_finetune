"""
Diagnose data issues that might cause NaN losses
"""

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from pathlib import Path

DATA_PATH = 'dataset.txt'
VAL_SPLIT = 0.10

print("="*70)
print("DATA DIAGNOSTICS")
print("="*70)

# Load data
print("\n[1/5] Loading data...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Total sentences: {len(texts):,}")

# Check for problematic sentences
print("\n[2/5] Checking for problematic sentences...")
issues = []

for i, text in enumerate(texts):
    # Check length
    if len(text) < 3:
        issues.append((i, "too short", text))
    elif len(text) > 5000:
        issues.append((i, "too long", text[:50]))
    
    # Check for weird characters
    if any(ord(c) > 65535 for c in text):
        issues.append((i, "invalid unicode", text[:50]))
    
    # Check if mostly numbers or symbols
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < len(text) * 0.3:
        issues.append((i, "too few letters", text[:50]))

if issues:
    print(f"⚠ Found {len(issues)} problematic sentences:")
    for idx, issue_type, sample in issues[:10]:
        print(f"  Line {idx}: {issue_type} - {sample}...")
else:
    print("✓ No obvious issues found")

# Check tokenization
print("\n[3/5] Testing tokenization...")
tokenizer = AutoTokenizer.from_pretrained("mideind/IceBERT")

sample_texts = texts[:100]
tokenization_issues = []

for i, text in enumerate(sample_texts):
    try:
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )
        
        # Check for issues
        if len(tokens['input_ids']) == 0:
            tokenization_issues.append((i, "empty tokens", text[:50]))
        elif len(tokens['input_ids']) < 3:
            tokenization_issues.append((i, "too few tokens", text[:50]))
            
    except Exception as e:
        tokenization_issues.append((i, str(e), text[:50]))

if tokenization_issues:
    print(f"⚠ Found {len(tokenization_issues)} tokenization issues:")
    for idx, issue, sample in tokenization_issues[:5]:
        print(f"  Sample {idx}: {issue} - {sample}...")
else:
    print("✓ Tokenization looks good")

# Test model inference
print("\n[4/5] Testing model inference...")
model = AutoModelForMaskedLM.from_pretrained("mideind/IceBERT")
model.eval()

inference_issues = []
for i, text in enumerate(sample_texts[:20]):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
            if not torch.isfinite(torch.tensor(loss)):
                inference_issues.append((i, f"loss={loss}", text[:50]))
                
    except Exception as e:
        inference_issues.append((i, str(e), text[:50]))

if inference_issues:
    print(f"⚠ Found {len(inference_issues)} inference issues:")
    for idx, issue, sample in inference_issues[:5]:
        print(f"  Sample {idx}: {issue} - {sample}...")
else:
    print("✓ Model inference looks good")

# Check validation split
print("\n[5/5] Checking validation split...")
dataset = Dataset.from_dict({"text": texts})

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding=False,
        return_attention_mask=True,
        return_special_tokens_mask=True
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
split = tokenized.train_test_split(test_size=VAL_SPLIT, seed=42)
train_data = split["train"]
eval_data = split["test"]

print(f"Training: {len(train_data):,} samples")
print(f"Validation: {len(eval_data):,} samples")

# Test a batch from validation
print("\n[6/5] Testing validation batch...")
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Get first few validation samples
test_batch = [eval_data[i] for i in range(min(8, len(eval_data)))]

try:
    batch = data_collator(test_batch)
    
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss.item()
        
    print(f"✓ Validation batch test:")
    print(f"  Loss: {loss}")
    
    if not torch.isfinite(torch.tensor(loss)):
        print("  ⚠ NaN/Inf detected in validation batch!")
    else:
        print("  ✓ Loss is valid")
        
except Exception as e:
    print(f"❌ Error testing validation batch: {e}")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nRecommendations:")
if issues or tokenization_issues or inference_issues:
    print("  1. Clean your data - remove problematic sentences")
    print("  2. Check encoding (should be UTF-8)")
    print("  3. Remove very short (<5 chars) or very long (>1000 chars) sentences")
else:
    print("  Data looks okay. The NaN issue might be:")
    print("  1. A bug in the training loop")
    print("  2. Numerical instability during evaluation")
    print("  3. Try training WITHOUT early stopping first")

print("\n" + "="*70)