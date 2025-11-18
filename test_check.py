with open('egil_saga_am132_norm.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
    chars = sum(len(line) for line in lines)

print(f"Sentences: {len(lines)}")
print(f"Characters: {chars:,}")
print(f"Avg length: {chars/len(lines):.0f} chars/sentence")

if len(lines) >= 500 and chars >= 50000:
    print("✓ EXCELLENT - Great test set size")
elif len(lines) >= 200 and chars >= 20000:
    print("✓ GOOD - Adequate for evaluation")
elif len(lines) >= 100:
    print("⚠ MINIMUM - Consider adding more")
else:
    print("❌ TOO SMALL - Need at least 100 sentences")