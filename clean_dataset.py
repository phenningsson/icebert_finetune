"""
Clean Old Icelandic corpus by removing problematic sentences
"""

INPUT_FILE = 'dataset.txt'
OUTPUT_FILE = 'dataset_cleaned.txt'

print("="*70)
print("CLEANING OLD ICELANDIC CORPUS")
print("="*70)

# Load data
print(f"\nLoading: {INPUT_FILE}")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"Original: {len(texts):,} sentences")

# Filter problematic sentences
cleaned = []
removed = []

for i, text in enumerate(texts):
    # Skip if too short
    if len(text) < 5:
        removed.append((i, "too short", text))
        continue
    
    # Skip if too long
    if len(text) > 2000:
        removed.append((i, "too long", text[:50]))
        continue
    
    # Skip if has weird unicode
    if any(ord(c) > 65535 for c in text):
        removed.append((i, "invalid unicode", text[:50]))
        continue
    
    # Skip if mostly numbers/symbols (less than 30% letters)
    alpha_count = sum(c.isalpha() for c in text)
    if alpha_count < len(text) * 0.3:
        removed.append((i, "too few letters", text[:50]))
        continue
    
    # Keep this sentence
    cleaned.append(text)

# Save cleaned data
print(f"\nCleaned: {len(cleaned):,} sentences")
print(f"Removed: {len(removed):,} sentences ({len(removed)/len(texts)*100:.1f}%)")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned))

print(f"\n✓ Saved to: {OUTPUT_FILE}")

# Show samples of what was removed
if removed:
    print(f"\nSample of removed sentences (first 20):")
    for idx, reason, text in removed[:20]:
        print(f"  Line {idx}: {reason} - '{text}'")

# Show statistics
print("\n" + "="*70)
print("STATISTICS")
print("="*70)
print(f"Original sentences:  {len(texts):,}")
print(f"Cleaned sentences:   {len(cleaned):,}")
print(f"Removed sentences:   {len(removed):,}")
print(f"Removal rate:        {len(removed)/len(texts)*100:.2f}%")

# Character counts
original_chars = sum(len(t) for t in texts)
cleaned_chars = sum(len(t) for t in cleaned)
print(f"\nOriginal characters: {original_chars:,}")
print(f"Cleaned characters:  {cleaned_chars:,}")
print(f"Character retention: {cleaned_chars/original_chars*100:.1f}%")

print("\n" + "="*70)
print("NEXT STEP: Update your training script")
print("="*70)
print(f"Change DATA_PATH to: '{OUTPUT_FILE}'")
print("Then run: python train_v2.py")
print("="*70)