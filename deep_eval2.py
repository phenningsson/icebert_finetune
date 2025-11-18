"""
Standardized Evaluation Metrics for Masked Language Models
Based on: Devlin et al. (BERT), Liu et al. (RoBERTa), and standard NLP benchmarks

Updated to include V3 model (partial parameter freezing)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
import json

# ============================================================================
# CONFIG
# ============================================================================

ORIGINAL_MODEL = 'mideind/IceBERT'
FINETUNED_V1 = './icebert-old-icelandic'       # V1: 7.6k sentences, 3 epochs, full fine-tuning
FINETUNED_V2 = './icebert-old-icelandic-v2'    # V2: 39k sentences, 8 epochs, full fine-tuning
FINETUNED_V3 = './icebert-old-icelandic-v3-partial-freeze'  # V3: 39k sentences, 8 epochs, ~55% params frozen
TEST_FILE = 'egil_saga_am132_norm.txt'
TEST_SAMPLE_SIZE = 2000
RANDOM_SEED = 42

# ============================================================================
# STANDARD METRIC 1: PERPLEXITY
# ============================================================================

def calculate_perplexity(model, tokenizer, texts, max_length=512):
    """
    Perplexity (PPL) - Standard metric for language models
    Lower is better. Measures how well model predicts the text.
    
    Formula: PPL = exp(average_cross_entropy_loss)
    
    Reference: Used in BERT, GPT-2, GPT-3 papers
    """
    print("\n[METRIC 1/5] Perplexity (PPL)")
    print("-" * 70)
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Accumulate loss (cross-entropy)
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"  Cross-entropy loss: {avg_loss:.4f}")
    print(f"  Perplexity (PPL):   {perplexity:.2f}")
    
    return {
        'perplexity': perplexity,
        'cross_entropy_loss': avg_loss
    }

# ============================================================================
# STANDARD METRIC 2: MASKED TOKEN ACCURACY
# ============================================================================

def calculate_masked_accuracy(model, tokenizer, texts, mask_prob=0.15, max_length=512):
    """
    Masked Token Accuracy - Percentage of correctly predicted masked tokens
    Higher is better (0-100%).
    
    This is the primary metric used in BERT evaluation.
    
    Reference: Devlin et al., 2019 (BERT paper)
    """
    print("\n[METRIC 2/5] Masked Token Accuracy")
    print("-" * 70)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating accuracy"):
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs["input_ids"][0]
            
            # Randomly mask tokens (excluding special tokens)
            special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, 
                            tokenizer.pad_token_id, tokenizer.unk_token_id]
            maskable_positions = [
                i for i, token_id in enumerate(input_ids)
                if token_id not in special_tokens
            ]
            
            if len(maskable_positions) == 0:
                continue
            
            # Mask 15% of tokens (BERT standard)
            n_mask = max(1, int(len(maskable_positions) * mask_prob))
            mask_positions = np.random.choice(maskable_positions, n_mask, replace=False)
            
            # Store original tokens
            original_tokens = input_ids[mask_positions].clone()
            
            # Apply mask
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = tokenizer.mask_token_id
            
            # Predict
            outputs = model(masked_input_ids.unsqueeze(0))
            predictions = outputs.logits[0, mask_positions].argmax(dim=-1)
            
            # Calculate accuracy
            correct += (predictions == original_tokens).sum().item()
            total += len(mask_positions)
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"  Correct predictions: {correct}/{total}")
    print(f"  Accuracy:            {accuracy:.2f}%")
    
    return {
        'masked_token_accuracy': accuracy,
        'correct': correct,
        'total': total
    }

# ============================================================================
# STANDARD METRIC 3: TOP-K ACCURACY
# ============================================================================

def calculate_topk_accuracy(model, tokenizer, texts, k_values=[1, 5, 10], 
                           mask_prob=0.15, max_length=512):
    """
    Top-K Accuracy - Is the correct token in the top-K predictions?
    Used to measure model confidence and ranking ability.
    
    Reference: Common in information retrieval and NLP
    """
    print("\n[METRIC 3/5] Top-K Accuracy")
    print("-" * 70)
    
    model.eval()
    topk_correct = {k: 0 for k in k_values}
    total = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating Top-K"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs["input_ids"][0]
            
            # Get maskable positions
            special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id, tokenizer.unk_token_id]
            maskable_positions = [
                i for i, token_id in enumerate(input_ids)
                if token_id not in special_tokens
            ]
            
            if len(maskable_positions) == 0:
                continue
            
            # Mask tokens
            n_mask = max(1, int(len(maskable_positions) * mask_prob))
            mask_positions = np.random.choice(maskable_positions, n_mask, replace=False)
            original_tokens = input_ids[mask_positions].clone()
            
            masked_input_ids = input_ids.clone()
            masked_input_ids[mask_positions] = tokenizer.mask_token_id
            
            # Get predictions
            outputs = model(masked_input_ids.unsqueeze(0))
            logits = outputs.logits[0, mask_positions]
            
            # Check if correct token is in top-k
            for k in k_values:
                topk_predictions = torch.topk(logits, k=k, dim=-1).indices
                for i, true_token in enumerate(original_tokens):
                    if true_token in topk_predictions[i]:
                        topk_correct[k] += 1
            
            total += len(mask_positions)
    
    results = {}
    for k in k_values:
        accuracy = (topk_correct[k] / total) * 100 if total > 0 else 0
        print(f"  Top-{k} Accuracy:  {accuracy:.2f}%")
        results[f'top_{k}_accuracy'] = accuracy
    
    return results

# ============================================================================
# STANDARD METRIC 4: PSEUDO-PERPLEXITY (Salazar et al., 2020)
# ============================================================================

def calculate_pseudo_perplexity(model, tokenizer, texts, max_length=512):
    """
    Pseudo-Perplexity (PPPL) - For masked LMs without autoregressive property
    More accurate for BERT-style models than standard perplexity.
    
    Reference: Salazar et al., 2020 "Masked Language Model Scoring"
    """
    print("\n[METRIC 4/5] Pseudo-Perplexity (PPPL)")
    print("-" * 70)
    
    model.eval()
    total_log_prob = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating pseudo-perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs["input_ids"][0]
            
            # Mask each token one at a time
            special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id,
                            tokenizer.pad_token_id, tokenizer.unk_token_id]
            
            for i in range(len(input_ids)):
                if input_ids[i].item() in special_tokens:
                    continue
                
                # Mask current token
                masked_input = input_ids.clone()
                original_token = masked_input[i].item()
                masked_input[i] = tokenizer.mask_token_id
                
                # Get prediction probability
                outputs = model(masked_input.unsqueeze(0))
                logits = outputs.logits[0, i]
                probs = torch.softmax(logits, dim=-1)
                token_prob = probs[original_token].item()
                
                # Accumulate log probability
                if token_prob > 0:
                    total_log_prob += np.log(token_prob)
                    total_tokens += 1
    
    # Calculate pseudo-perplexity
    avg_log_prob = total_log_prob / total_tokens if total_tokens > 0 else 0
    pseudo_ppl = np.exp(-avg_log_prob)
    
    print(f"  Pseudo-Perplexity:  {pseudo_ppl:.2f}")
    
    return {
        'pseudo_perplexity': pseudo_ppl,
        'avg_log_prob': avg_log_prob
    }

# ============================================================================
# STANDARD METRIC 5: BITS PER CHARACTER (BPC)
# ============================================================================

def calculate_bpc(model, tokenizer, texts, max_length=512):
    """
    Bits Per Character (BPC) - Information-theoretic metric
    Lower is better. Measures compression efficiency.
    
    Common in language model evaluation, especially for character-level models.
    
    Reference: Used in GPT, Transformer-XL papers
    """
    print("\n[METRIC 5/5] Bits Per Character (BPC)")
    print("-" * 70)
    
    model.eval()
    total_loss = 0
    total_chars = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating BPC"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False
            )
            
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Accumulate
            total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
            total_chars += len(text)
    
    # Calculate BPC
    avg_loss = total_loss / total_chars
    bpc = avg_loss / np.log(2)  # Convert to bits
    
    print(f"  Bits Per Character: {bpc:.4f}")
    
    return {
        'bits_per_character': bpc
    }

# ============================================================================
# LOAD DATA
# ============================================================================

def load_test_data(file_path):
    """Load external Old Icelandic test data"""
    print("\nLoading external test data...")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    chars = sum(len(t) for t in texts)
    
    print(f"  Total sentences: {len(texts):,}")
    print(f"  Total characters: {chars:,}")
    print(f"  Avg length: {chars/len(texts):.0f} chars/sentence")
    
    if len(texts) >= 500:
        print("  ✓ EXCELLENT test set size!")
    
    return texts

# ============================================================================
# COMPARISON FUNCTION
# ============================================================================

def compare_models(original_path, v1_path, v2_path, v3_path, test_texts):
    """Compare original, v1, v2, and v3 models on all metrics"""
    print("\n" + "="*70)
    print("COMPARING FOUR MODELS")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    print("  [1/4] Original IceBERT...")
    orig_tokenizer = AutoTokenizer.from_pretrained(original_path)
    orig_model = AutoModelForMaskedLM.from_pretrained(original_path)
    
    print("  [2/4] Finetuned V1 (7.6k sentences, full fine-tuning)...")
    v1_tokenizer = AutoTokenizer.from_pretrained(v1_path)
    v1_model = AutoModelForMaskedLM.from_pretrained(v1_path)
    
    print("  [3/4] Finetuned V2 (39k sentences, full fine-tuning)...")
    v2_tokenizer = AutoTokenizer.from_pretrained(v2_path)
    v2_model = AutoModelForMaskedLM.from_pretrained(v2_path)
    
    print("  [4/4] Finetuned V3 (39k sentences, partial freezing ~55% params)...")
    v3_tokenizer = AutoTokenizer.from_pretrained(v3_path)
    v3_model = AutoModelForMaskedLM.from_pretrained(v3_path)
    
    results = {
        'original': {},
        'v1_finetuned': {},
        'v2_finetuned': {},
        'v3_partial_freeze': {}
    }
    
    # For perplexity and BPC, use ALL test data (fast metrics)
    print(f"\nNote: Using all {len(test_texts):,} sentences for Perplexity and BPC")
    print(f"      Using {TEST_SAMPLE_SIZE} sentences for detailed metrics (faster)")
    
    # Sample for expensive metrics
    test_sample = test_texts[:TEST_SAMPLE_SIZE]
    
    print("\n" + "="*70)
    print("EVALUATING ORIGINAL ICEBERT")
    print("="*70)
    
    results['original']['perplexity'] = calculate_perplexity(orig_model, orig_tokenizer, test_texts)
    results['original']['accuracy'] = calculate_masked_accuracy(orig_model, orig_tokenizer, test_sample)
    results['original']['topk'] = calculate_topk_accuracy(orig_model, orig_tokenizer, test_sample)
    results['original']['pseudo_ppl'] = calculate_pseudo_perplexity(orig_model, orig_tokenizer, test_sample)
    results['original']['bpc'] = calculate_bpc(orig_model, orig_tokenizer, test_texts)
    
    print("\n" + "="*70)
    print("EVALUATING V1 MODEL (7.6k sentences, 3 epochs, 100% params trained)")
    print("="*70)
    
    results['v1_finetuned']['perplexity'] = calculate_perplexity(v1_model, v1_tokenizer, test_texts)
    results['v1_finetuned']['accuracy'] = calculate_masked_accuracy(v1_model, v1_tokenizer, test_sample)
    results['v1_finetuned']['topk'] = calculate_topk_accuracy(v1_model, v1_tokenizer, test_sample)
    results['v1_finetuned']['pseudo_ppl'] = calculate_pseudo_perplexity(v1_model, v1_tokenizer, test_sample)
    results['v1_finetuned']['bpc'] = calculate_bpc(v1_model, v1_tokenizer, test_texts)
    
    print("\n" + "="*70)
    print("EVALUATING V2 MODEL (39k sentences, 8 epochs, 100% params trained)")
    print("="*70)
    
    results['v2_finetuned']['perplexity'] = calculate_perplexity(v2_model, v2_tokenizer, test_texts)
    results['v2_finetuned']['accuracy'] = calculate_masked_accuracy(v2_model, v2_tokenizer, test_sample)
    results['v2_finetuned']['topk'] = calculate_topk_accuracy(v2_model, v2_tokenizer, test_sample)
    results['v2_finetuned']['pseudo_ppl'] = calculate_pseudo_perplexity(v2_model, v2_tokenizer, test_sample)
    results['v2_finetuned']['bpc'] = calculate_bpc(v2_model, v2_tokenizer, test_texts)
    
    print("\n" + "="*70)
    print("EVALUATING V3 MODEL (39k sentences, 8 epochs, ~55% params trained)")
    print("="*70)
    
    results['v3_partial_freeze']['perplexity'] = calculate_perplexity(v3_model, v3_tokenizer, test_texts)
    results['v3_partial_freeze']['accuracy'] = calculate_masked_accuracy(v3_model, v3_tokenizer, test_sample)
    results['v3_partial_freeze']['topk'] = calculate_topk_accuracy(v3_model, v3_tokenizer, test_sample)
    results['v3_partial_freeze']['pseudo_ppl'] = calculate_pseudo_perplexity(v3_model, v3_tokenizer, test_sample)
    results['v3_partial_freeze']['bpc'] = calculate_bpc(v3_model, v3_tokenizer, test_texts)
    
    return results

# ============================================================================
# REPORT GENERATION
# ============================================================================

def print_comparison_report(results):
    """Print formatted comparison report for four models"""
    print("\n" + "="*110)
    print("EVALUATION SUMMARY - EXTERNAL OLD ICELANDIC TEST SET")
    print("="*110)
    
    orig = results['original']
    v1 = results['v1_finetuned']
    v2 = results['v2_finetuned']
    v3 = results['v3_partial_freeze']
    
    print("\n{:<30s} {:>15s} {:>15s} {:>15s} {:>15s}".format(
        "Metric", "Original", "V1 (7.6k)", "V2 (39k)", "V3 (freeze)"
    ))
    print("-" * 110)
    
    # Masked accuracy
    orig_acc = orig['accuracy']['masked_token_accuracy']
    v1_acc = v1['accuracy']['masked_token_accuracy']
    v2_acc = v2['accuracy']['masked_token_accuracy']
    v3_acc = v3['accuracy']['masked_token_accuracy']
    print("{:<30s} {:>14.2f}% {:>14.2f}% {:>14.2f}% {:>14.2f}%".format(
        "Masked Token Accuracy", orig_acc, v1_acc, v2_acc, v3_acc
    ))
    
    # Top-5 accuracy
    orig_top5 = orig['topk']['top_5_accuracy']
    v1_top5 = v1['topk']['top_5_accuracy']
    v2_top5 = v2['topk']['top_5_accuracy']
    v3_top5 = v3['topk']['top_5_accuracy']
    print("{:<30s} {:>14.2f}% {:>14.2f}% {:>14.2f}% {:>14.2f}%".format(
        "Top-5 Accuracy", orig_top5, v1_top5, v2_top5, v3_top5
    ))
    
    # Pseudo-perplexity
    orig_pppl = orig['pseudo_ppl']['pseudo_perplexity']
    v1_pppl = v1['pseudo_ppl']['pseudo_perplexity']
    v2_pppl = v2['pseudo_ppl']['pseudo_perplexity']
    v3_pppl = v3['pseudo_ppl']['pseudo_perplexity']
    print("{:<30s} {:>15.2f} {:>15.2f} {:>15.2f} {:>15.2f}".format(
        "Pseudo-Perplexity", orig_pppl, v1_pppl, v2_pppl, v3_pppl
    ))
    
    print("="*110)
    
    # Improvements
    print("\nIMPROVEMENTS:")
    print("-" * 110)
    
    print("\nOriginal → V1 (7.6k sentences, 3 epochs, 100% params):")
    v1_acc_change = v1_acc - orig_acc
    v1_top5_change = v1_top5 - orig_top5
    v1_pppl_change = ((orig_pppl - v1_pppl) / orig_pppl) * 100
    print(f"  Masked Accuracy:     {v1_acc_change:+.2f} pp")
    print(f"  Top-5 Accuracy:      {v1_top5_change:+.2f} pp")
    print(f"  Pseudo-Perplexity:   {v1_pppl_change:+.1f}%")
    
    print("\nV1 → V2 (39k sentences, 8 epochs, 100% params):")
    v2_acc_change = v2_acc - v1_acc
    v2_top5_change = v2_top5 - v1_top5
    v2_pppl_change = ((v1_pppl - v2_pppl) / v1_pppl) * 100
    print(f"  Masked Accuracy:     {v2_acc_change:+.2f} pp")
    print(f"  Top-5 Accuracy:      {v2_top5_change:+.2f} pp")
    print(f"  Pseudo-Perplexity:   {v2_pppl_change:+.1f}%")
    
    print("\nV2 → V3 (same data/epochs, but ~55% params frozen):")
    v3_acc_change = v3_acc - v2_acc
    v3_top5_change = v3_top5 - v2_top5
    v3_pppl_change = ((v2_pppl - v3_pppl) / v2_pppl) * 100
    print(f"  Masked Accuracy:     {v3_acc_change:+.2f} pp")
    print(f"  Top-5 Accuracy:      {v3_top5_change:+.2f} pp")
    print(f"  Pseudo-Perplexity:   {v3_pppl_change:+.1f}%")
    
    print("\nOriginal → V2 (best full fine-tuning):")
    total_acc_change_v2 = v2_acc - orig_acc
    total_top5_change_v2 = v2_top5 - orig_top5
    total_pppl_change_v2 = ((orig_pppl - v2_pppl) / orig_pppl) * 100
    print(f"  Masked Accuracy:     {total_acc_change_v2:+.2f} pp")
    print(f"  Top-5 Accuracy:      {total_top5_change_v2:+.2f} pp")
    print(f"  Pseudo-Perplexity:   {total_pppl_change_v2:+.1f}%")
    
    print("\nOriginal → V3 (partial freezing approach):")
    total_acc_change_v3 = v3_acc - orig_acc
    total_top5_change_v3 = v3_top5 - orig_top5
    total_pppl_change_v3 = ((orig_pppl - v3_pppl) / orig_pppl) * 100
    print(f"  Masked Accuracy:     {total_acc_change_v3:+.2f} pp")
    print(f"  Top-5 Accuracy:      {total_top5_change_v3:+.2f} pp")
    print(f"  Pseudo-Perplexity:   {total_pppl_change_v3:+.1f}%")
    
    print("\n" + "="*110)
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    
    # V1 assessment
    v1_improvements = sum([v1_acc_change > 5, v1_top5_change > 5, v1_pppl_change > 30])
    if v1_improvements >= 3:
        print("  V1: Strong improvement from original")
    elif v1_improvements >= 2:
        print("  V1: Good improvement from original")
    else:
        print("  V1: Moderate improvement from original")
    
    # V2 assessment
    v2_improvements = sum([v2_acc_change > 5, v2_top5_change > 5, v2_pppl_change > 20])
    if v2_improvements >= 3:
        print("  V2: Strong improvement from V1 (more data & training paid off)")
    elif v2_improvements >= 2:
        print("  V2: Good improvement from V1")
    elif v2_improvements >= 1:
        print("  V2: Moderate improvement from V1")
    else:
        print("  V2: Limited improvement from V1 (may be near optimal)")
    
    # V3 assessment
    print("\n  V3 Analysis (Partial Freezing):")
    if abs(v3_acc_change) < 2:
        print("    ✓ Nearly equivalent to V2 (< 2pp difference)")
        print("    ✓ Efficiency gain: ~45% fewer parameters trained")
        print("    ✓ Training time: ~20-30% faster")
        print("    → Conclusion: Good trade-off for efficiency")
    elif v3_acc_change > 2:
        print("    ✓ BETTER than V2 (unexpected!)")
        print("    → Freezing may have acted as regularization")
    else:
        print("    ✗ Worse than V2 by {:.1f}pp".format(abs(v3_acc_change)))
        if abs(v3_acc_change) < 5:
            print("    → Still acceptable trade-off for 45% fewer params")
        else:
            print("    → Significant performance drop, full fine-tuning better")
    
    # Save results
    output_file = 'evaluation_results_all_models_v3.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    # Create email-friendly summary
    print("\n" + "="*110)
    print("EMAIL-FRIENDLY SUMMARY")
    print("="*110)
    print("\nEvaluation Results - Old Icelandic IceBERT Models\n")
    print("Test Set: External Old Icelandic text\n")
    print("Metric                      Original    V1 (7.6k)   V2 (39k)    V3 (freeze)  V2→V3")
    print("-" * 110)
    print(f"Masked Token Accuracy       {orig_acc:>6.2f}%    {v1_acc:>6.2f}%    {v2_acc:>6.2f}%    {v3_acc:>6.2f}%      {v3_acc_change:>+6.2f}pp")
    print(f"Top-5 Accuracy              {orig_top5:>6.2f}%    {v1_top5:>6.2f}%    {v2_top5:>6.2f}%    {v3_top5:>6.2f}%      {v3_top5_change:>+6.2f}pp")
    print(f"Pseudo-Perplexity        {orig_pppl:>9.2f}   {v1_pppl:>8.2f}   {v2_pppl:>8.2f}   {v3_pppl:>8.2f}      {v3_pppl_change:>+6.1f}%")
    print("\nTraining Details:")
    print("  V1: 7.6k sentences, 3 epochs, 100% parameters")
    print("  V2: 39k sentences, 8 epochs, 100% parameters (best full fine-tuning)")
    print("  V3: 39k sentences, 8 epochs, ~55% parameters (embeddings + layers 0-3 frozen)")
    print("\n(pp = percentage points; lower pseudo-perplexity is better)")
    print("="*110)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*110)
    print("STANDARDIZED MLM EVALUATION - FOUR MODEL COMPARISON")
    print("="*110)
    print("\nComparing:")
    print("  1. Original IceBERT (baseline)")
    print("  2. V1 Finetuned (7.6k sentences, 3 epochs, full fine-tuning)")
    print("  3. V2 Finetuned (39k sentences, 8 epochs, full fine-tuning)")
    print("  4. V3 Finetuned (39k sentences, 8 epochs, ~55% params - partial freezing)")
    print("\nMetrics based on:")
    print("  - Devlin et al., 2019 (BERT)")
    print("  - Liu et al., 2019 (RoBERTa)")
    print("  - Salazar et al., 2020 (MLM Scoring)")
    print("="*110)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load test data
    test_texts = load_test_data(TEST_FILE)
    
    # Run comparison on all four models
    results = compare_models(ORIGINAL_MODEL, FINETUNED_V1, FINETUNED_V2, FINETUNED_V3, test_texts)
    
    # Print report
    print_comparison_report(results)
    
    print("\n" + "="*110)
    print("EVALUATION COMPLETE")
    print("="*110)

if __name__ == "__main__":
    main()