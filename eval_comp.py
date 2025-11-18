"""
Standardized Evaluation Metrics for Masked Language Models
Based on: Devlin et al. (BERT), Liu et al. (RoBERTa), and standard NLP benchmarks
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

FINETUNED_MODEL = './icebert-old-icelandic'
ORIGINAL_MODEL = 'mideind/IceBERT'
TEST_FILE = 'eval_set.txt'  # UPDATE THIS to your test file name
TEST_SAMPLE_SIZE = 200  # Use subset for faster detailed metrics (Top-K, Pseudo-PPL)
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

def compare_models(original_model_path, finetuned_model_path, test_texts):
    """Compare original vs finetuned model on all metrics"""
    print("\n" + "="*70)
    print("COMPARING MODELS")
    print("="*70)
    
    # Load models
    print("\nLoading models...")
    orig_tokenizer = AutoTokenizer.from_pretrained(original_model_path)
    orig_model = AutoModelForMaskedLM.from_pretrained(original_model_path)
    
    fine_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    fine_model = AutoModelForMaskedLM.from_pretrained(finetuned_model_path)
    
    results = {
        'original': {},
        'finetuned': {}
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
    print("EVALUATING YOUR OLD ICELANDIC MODEL")
    print("="*70)
    
    results['finetuned']['perplexity'] = calculate_perplexity(fine_model, fine_tokenizer, test_texts)
    results['finetuned']['accuracy'] = calculate_masked_accuracy(fine_model, fine_tokenizer, test_sample)
    results['finetuned']['topk'] = calculate_topk_accuracy(fine_model, fine_tokenizer, test_sample)
    results['finetuned']['pseudo_ppl'] = calculate_pseudo_perplexity(fine_model, fine_tokenizer, test_sample)
    results['finetuned']['bpc'] = calculate_bpc(fine_model, fine_tokenizer, test_texts)
    
    return results

# ============================================================================
# REPORT GENERATION
# ============================================================================

def print_comparison_report(results):
    """Print formatted comparison report"""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY - EXTERNAL OLD ICELANDIC TEST SET")
    print("="*70)
    
    orig = results['original']
    fine = results['finetuned']
    
    print("\n{:<30s} {:>15s} {:>15s} {:>10s}".format(
        "Metric", "Original", "Finetuned", "Change"
    ))
    print("-" * 70)
    
    # Perplexity
    orig_ppl = orig['perplexity']['perplexity']
    fine_ppl = fine['perplexity']['perplexity']
    change_ppl = ((orig_ppl - fine_ppl) / orig_ppl) * 100
    print("{:<30s} {:>15.2f} {:>15.2f} {:>9.1f}%".format(
        "Perplexity (lower=better)", orig_ppl, fine_ppl, change_ppl
    ))
    
    # Masked accuracy
    orig_acc = orig['accuracy']['masked_token_accuracy']
    fine_acc = fine['accuracy']['masked_token_accuracy']
    change_acc = fine_acc - orig_acc
    print("{:<30s} {:>14.2f}% {:>14.2f}% {:>9.1f}%".format(
        "Masked Token Accuracy", orig_acc, fine_acc, change_acc
    ))
    
    # Top-5 accuracy
    orig_top5 = orig['topk']['top_5_accuracy']
    fine_top5 = fine['topk']['top_5_accuracy']
    change_top5 = fine_top5 - orig_top5
    print("{:<30s} {:>14.2f}% {:>14.2f}% {:>9.1f}%".format(
        "Top-5 Accuracy", orig_top5, fine_top5, change_top5
    ))
    
    # Pseudo-perplexity
    orig_pppl = orig['pseudo_ppl']['pseudo_perplexity']
    fine_pppl = fine['pseudo_ppl']['pseudo_perplexity']
    change_pppl = ((orig_pppl - fine_pppl) / orig_pppl) * 100
    print("{:<30s} {:>15.2f} {:>15.2f} {:>9.1f}%".format(
        "Pseudo-Perplexity", orig_pppl, fine_pppl, change_pppl
    ))
    
    # BPC
    orig_bpc = orig['bpc']['bits_per_character']
    fine_bpc = fine['bpc']['bits_per_character']
    change_bpc = ((orig_bpc - fine_bpc) / orig_bpc) * 100
    print("{:<30s} {:>15.4f} {:>15.4f} {:>9.1f}%".format(
        "Bits Per Character", orig_bpc, fine_bpc, change_bpc
    ))
    
    print("="*70)
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    improvements = sum([
        change_ppl > 0,  # Lower perplexity is better
        change_acc > 0,  # Higher accuracy is better
        change_top5 > 0,  # Higher top-k is better
        change_pppl > 0,  # Lower pseudo-ppl is better
        change_bpc > 0   # Lower BPC is better
    ])
    
    if improvements >= 4:
        print("  ✓ EXCELLENT - Strong improvement on Old Icelandic")
    elif improvements >= 3:
        print("  ✓ GOOD - Clear improvement on Old Icelandic")
    elif improvements >= 2:
        print("  ⚠ MODERATE - Some improvement")
    else:
        print("  ⚠ LIMITED - Consider more training data or epochs")
    
    # Save results
    output_file = 'evaluation_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to: {output_file}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("STANDARDIZED MLM EVALUATION")
    print("External Old Icelandic Test Set")
    print("="*70)
    print("\nMetrics based on:")
    print("  - Devlin et al., 2019 (BERT)")
    print("  - Liu et al., 2019 (RoBERTa)")
    print("  - Salazar et al., 2020 (MLM Scoring)")
    print("="*70)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load test data
    test_texts = load_test_data(TEST_FILE)
    
    # Run comparison
    results = compare_models(ORIGINAL_MODEL, FINETUNED_MODEL, test_texts)
    
    # Print report
    print_comparison_report(results)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()