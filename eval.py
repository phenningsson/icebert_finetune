from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np

model = AutoModelForMaskedLM.from_pretrained('./icebert-old-icelandic')
tokenizer = AutoTokenizer.from_pretrained('./icebert-old-icelandic')

# Your test text
with open('kombinerad.txt') as f:
    texts = [line.strip() for line in f if line.strip()][:100]

# Calculate perplexity
model.eval()
losses = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

ppl = np.exp(np.mean(losses))
print(f"Perplexity: {ppl:.2f}")